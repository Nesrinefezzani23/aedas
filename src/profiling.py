# =============================================================
#  A.E.D.A.S. — Profiling TFLite par couche
#  src/profiling.py
#
#  Mesure la latence et l'empreinte mémoire de chaque couche
#  du modèle CNN 1D, pour identifier les goulots d'étranglement
#  avant déploiement embarqué.
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import time
import json
from pathlib import Path
from dataclasses import dataclass, field

from config import (
    FEATURES_PATH, MODELS_DIR, N_MFCC,
    MFCC_TIME_STEPS, RANDOM_SEED, set_seed
)

set_seed(RANDOM_SEED)

TFLITE_INT8 = MODELS_DIR / "tflite" / "aedas_int8.tflite"
TFLITE_FP32 = MODELS_DIR / "tflite" / "aedas_fp32.tflite"
PROFILING_DIR = MODELS_DIR / "profiling"


# ── Structures de données ─────────────────────────────────────

@dataclass
class LayerProfile:
    """Profil de performance d'une couche."""
    index:        int
    name:         str
    op_type:      str
    input_shape:  list
    output_shape: list
    input_dtype:  str
    output_dtype: str
    params_count: int = 0
    latency_ms:   float = 0.0
    memory_bytes: int = 0

    @property
    def memory_kb(self): return self.memory_bytes / 1024

    @property
    def latency_pct(self): return 0.0  # calculé après agrégation


@dataclass
class ModelProfile:
    """Profil complet du modèle."""
    model_name:      str
    model_size_kb:   float
    total_params:    int
    layers:          list = field(default_factory=list)
    total_latency_ms:float = 0.0
    total_memory_kb: float = 0.0
    n_runs:          int = 100


# ── Profiling TFLite ──────────────────────────────────────────

class TFLiteProfiler:
    """
    Profiler pour modèles TFLite.
    Analyse couche par couche : shape, dtype, latence, mémoire.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path

        # Interpréteur avec profiling activé
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN
            ),
            experimental_preserve_all_tensors=True,
        )
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.tensor_details = self.interpreter.get_tensor_details()

        print(f"  Modèle chargé : {model_path.name}")
        print(f"  Taille        : {model_path.stat().st_size / 1024:.1f} Ko")
        print(f"  Tenseurs      : {len(self.tensor_details)}")

    def _get_input_sample(self) -> np.ndarray:
        """Génère un échantillon d'entrée compatible avec le modèle."""
        input_shape = self.input_details[0]["shape"]
        input_dtype = self.input_details[0]["dtype"]

        if input_dtype == np.int8:
            return np.random.randint(-128, 127,
                                     size=input_shape,
                                     dtype=np.int8)
        else:
            return np.random.randn(*input_shape).astype(np.float32)

    def profile_tensors(self) -> list[LayerProfile]:
        """Analyse tous les tenseurs du modèle."""
        profiles = []

        for i, tensor in enumerate(self.tensor_details):
            shape = list(tensor["shape"]) if len(tensor["shape"]) > 0 else []

            # Calcul de l'empreinte mémoire du tenseur
            if len(shape) > 0:
                n_elements = 1
                for s in shape:
                    n_elements *= s
                dtype_size = np.dtype(tensor["dtype"]).itemsize
                memory_bytes = n_elements * dtype_size
            else:
                memory_bytes = 0

            profile = LayerProfile(
                index=i,
                name=tensor["name"],
                op_type=self._infer_op_type(tensor["name"]),
                input_shape=shape,
                output_shape=shape,
                input_dtype=str(tensor["dtype"]),
                output_dtype=str(tensor["dtype"]),
                memory_bytes=memory_bytes,
            )
            profiles.append(profile)

        return profiles

    def _infer_op_type(self, name: str) -> str:
        """Infère le type d'opération depuis le nom du tenseur."""
        name_lower = name.lower()
        if "conv" in name_lower:    return "Conv1D"
        if "batch" in name_lower:   return "BatchNorm"
        if "relu" in name_lower:    return "ReLU"
        if "pool" in name_lower:    return "Pooling"
        if "drop" in name_lower:    return "Dropout"
        if "dense" in name_lower:   return "Dense"
        if "permut" in name_lower:  return "Permute"
        if "gap" in name_lower:     return "GAP"
        if "output" in name_lower:  return "Output"
        return "Other"

    def benchmark_inference(self, n_runs: int = 200) -> dict:
        """
        Benchmark d'inférence complet avec statistiques détaillées.
        Warmup de 5 runs pour stabiliser le JIT.
        """
        sample = self._get_input_sample()

        # Warmup
        print(f"  Warmup (5 runs)...")
        for _ in range(5):
            self.interpreter.set_tensor(
                self.input_details[0]["index"], sample
            )
            self.interpreter.invoke()

        # Benchmark
        latencies = []
        print(f"  Benchmark ({n_runs} runs)...")

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.interpreter.set_tensor(
                self.input_details[0]["index"], sample
            )
            self.interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)

        return {
            "mean_ms":  float(latencies.mean()),
            "std_ms":   float(latencies.std()),
            "min_ms":   float(latencies.min()),
            "max_ms":   float(latencies.max()),
            "p50_ms":   float(np.percentile(latencies, 50)),
            "p95_ms":   float(np.percentile(latencies, 95)),
            "p99_ms":   float(np.percentile(latencies, 99)),
            "fps":      float(1000 / latencies.mean()),
            "n_runs":   n_runs,
        }

    def compute_memory_breakdown(self) -> dict:
        """
        Calcule la décomposition mémoire du modèle.
        Distingue poids, activations, et overhead runtime.
        """
        model_size = self.model_path.stat().st_size

        # Taille des tenseurs d'activation (intermédiaires)
        activation_bytes = 0
        weight_bytes     = 0

        for tensor in self.tensor_details:
            shape = tensor["shape"]
            if len(shape) == 0:
                continue
            n_elements = 1
            for s in shape:
                n_elements *= s
            dtype_size    = np.dtype(tensor["dtype"]).itemsize
            tensor_bytes  = n_elements * dtype_size

            # Heuristique : tenseurs avec batch dim = activations
            if len(shape) > 0 and shape[0] in [0, 1]:
                activation_bytes += tensor_bytes
            else:
                weight_bytes += tensor_bytes

        # Overhead runtime TFLite Micro (estimé)
        runtime_overhead = 10 * 1024  # ~10 Ko pour TFLite Micro runtime

        return {
            "model_file_kb":      model_size / 1024,
            "weights_kb":         weight_bytes / 1024,
            "activations_kb":     activation_bytes / 1024,
            "runtime_overhead_kb": runtime_overhead / 1024,
            "total_flash_kb":     model_size / 1024 + runtime_overhead / 1024,
            "total_ram_kb":       activation_bytes / 1024 + runtime_overhead / 1024,
        }


# ── Comparaison FP32 vs INT8 ──────────────────────────────────

def compare_fp32_int8(n_runs: int = 200) -> dict:
    """
    Compare les performances FP32 vs INT8 côte à côte.
    Mesure l'impact réel de la quantification sur la latence.
    """
    print("\n" + "=" * 55)
    print("  Comparaison FP32 vs INT8")
    print("=" * 55)

    results = {}

    for name, path in [("FP32", TFLITE_FP32), ("INT8", TFLITE_INT8)]:
        print(f"\n── {name} ──────────────────────────────────────")
        profiler = TFLiteProfiler(path)
        bench    = profiler.benchmark_inference(n_runs=n_runs)
        memory   = profiler.compute_memory_breakdown()
        results[name] = {"bench": bench, "memory": memory}

        print(f"  Latence moyenne : {bench['mean_ms']:.3f} ms ± {bench['std_ms']:.3f}")
        print(f"  P50 : {bench['p50_ms']:.3f} ms | P95 : {bench['p95_ms']:.3f} ms")
        print(f"  FPS : {bench['fps']:.1f}")
        print(f"  Flash (modèle)  : {memory['model_file_kb']:.1f} Ko")
        print(f"  RAM (activ.)    : {memory['activations_kb']:.1f} Ko")
        print(f"  Total Flash est.: {memory['total_flash_kb']:.1f} Ko")
        print(f"  Total RAM est.  : {memory['total_ram_kb']:.1f} Ko")

    fp32 = results["FP32"]["bench"]
    int8 = results["INT8"]["bench"]
    mem_fp32 = results["FP32"]["memory"]
    mem_int8 = results["INT8"]["memory"]

    speedup     = fp32["mean_ms"] / int8["mean_ms"]
    mem_savings = (1 - mem_int8["model_file_kb"] / mem_fp32["model_file_kb"]) * 100

    print(f"\n{'=' * 55}")
    print(f"  BILAN COMPARATIF")
    print(f"{'=' * 55}")
    print(f"  Accélération INT8 vs FP32  : {speedup:.2f}x")
    print(f"  Réduction mémoire Flash    : {mem_savings:.1f}%")
    print(f"  RAM activations FP32       : {mem_fp32['activations_kb']:.1f} Ko")
    print(f"  RAM activations INT8       : {mem_int8['activations_kb']:.1f} Ko")

    return results


# ── Simulation profil Cortex-M ────────────────────────────────

def estimate_cortex_m_performance(int8_bench: dict,
                                   int8_memory: dict) -> dict:
    """
    Estime les performances sur Cortex-M7 @ 400MHz
    à partir des mesures desktop (x86-64).

    Facteurs de correction empiriques basés sur :
    - Benchmarks EEMBC MLPerf Tiny
    - Papiers TFLite Micro sur STM32H7
    - CMSIS-NN vs implémentation naïve
    """
    # Ratio de performance x86-64 vs Cortex-M7 pour CNN INT8
    # x86 avec AVX512 : ~10-20x plus rapide que Cortex-M7
    # Avec CMSIS-NN (DSP instructions) : facteur ~12x
    PERF_RATIO_CONSERVATIVE = 15.0
    PERF_RATIO_CMSIS_NN     = 10.0

    latency_conservative_ms = int8_bench["mean_ms"] * PERF_RATIO_CONSERVATIVE
    latency_cmsis_ms        = int8_bench["mean_ms"] * PERF_RATIO_CMSIS_NN

    # Cycles ARM @ 400MHz
    cycles_conservative = latency_conservative_ms * 400_000
    cycles_cmsis        = latency_cmsis_ms * 400_000

    # Mémoire sur Cortex-M (Flash = modèle, RAM = activations + stack)
    flash_budget_kb = int8_memory["total_flash_kb"]
    ram_budget_kb   = int8_memory["total_ram_kb"] + 2.0  # +2Ko stack

    cortex_m_profile = {
        "cpu":                   "ARM Cortex-M7 @ 400MHz (ex. STM32H743)",
        "latency_conservative_ms": latency_conservative_ms,
        "latency_cmsis_nn_ms":    latency_cmsis_ms,
        "cycles_conservative":    cycles_conservative,
        "cycles_cmsis_nn":        cycles_cmsis,
        "flash_kb":               flash_budget_kb,
        "ram_kb":                 ram_budget_kb,
        "fits_cortex_m4":         flash_budget_kb < 256 and ram_budget_kb < 64,
        "fits_cortex_m7":         flash_budget_kb < 1024 and ram_budget_kb < 256,
        "realtime_compatible":    latency_cmsis_ms < 500,
    }

    print(f"\n{'=' * 55}")
    print(f"  PROJECTION CORTEX-M7 @ 400MHz")
    print(f"{'=' * 55}")
    print(f"  Latence estimée (sans CMSIS-NN) : {latency_conservative_ms:.2f} ms")
    print(f"  Latence estimée (avec CMSIS-NN) : {latency_cmsis_ms:.2f} ms")
    print(f"  Cycles ARM (CMSIS-NN)           : {cycles_cmsis:,.0f}")
    print(f"  Flash requise                   : {flash_budget_kb:.1f} Ko")
    print(f"  RAM requise                     : {ram_budget_kb:.1f} Ko")
    print(f"  Compatible Cortex-M4 (256K/64K) : {'✅' if cortex_m_profile['fits_cortex_m4'] else '❌'}")
    print(f"  Compatible Cortex-M7 (1M/256K)  : {'✅' if cortex_m_profile['fits_cortex_m7'] else '❌'}")
    print(f"  Temps réel (< 500ms/inférence)  : {'✅' if cortex_m_profile['realtime_compatible'] else '❌'}")

    return cortex_m_profile


# ── Rapport de profiling ──────────────────────────────────────

def generate_profiling_report(results: dict, cortex_profile: dict):
    """Sauvegarde un rapport JSON complet de profiling."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        """Encodeur JSON qui gère les types numpy et bool Python."""
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            if isinstance(obj, np.bool_):    return bool(obj)
            return super().default(obj)

    # Convertir les bool Python natifs aussi
    cortex_clean = {}
    for k, v in cortex_profile.items():
        if isinstance(v, bool):
            cortex_clean[k] = int(v)
        else:
            cortex_clean[k] = v

    report = {
        "model": "A.E.D.A.S. CNN 1D",
        "date":  "2026-07-07",
        "desktop": {
            "fp32": results["FP32"],
            "int8": results["INT8"],
        },
        "embedded_projection": cortex_clean,
    }

    report_path = PROFILING_DIR / "profiling_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Rapport sauvegardé : {report_path}")


# ── Point d'entrée ────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  A.E.D.A.S. — Profiling TFLite (FP32 vs INT8)")
    print("=" * 55)

    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    # Comparaison FP32 vs INT8
    results = compare_fp32_int8(n_runs=200)

    # Projection Cortex-M
    cortex_profile = estimate_cortex_m_performance(
        results["INT8"]["bench"],
        results["INT8"]["memory"],
    )

    # Sauvegarde
    generate_profiling_report(results, cortex_profile)

    print(f"\n✅ Profiling terminé.")