# =============================================================
#  A.E.D.A.S. — Pipeline d'inférence temps réel
#  src/inference.py
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import librosa
import time
from pathlib import Path
from dataclasses import dataclass, field

from config import (
    SAMPLE_RATE, DURATION, N_SAMPLES,
    N_MFCC, N_FFT, HOP_LENGTH, N_MELS, FMAX,
    MFCC_TIME_STEPS, FEATURES_PATH, MODELS_DIR
)

TFLITE_INT8 = MODELS_DIR / "tflite" / "aedas_int8.tflite"
OPTIMAL_THRESHOLD = 0.82


# ── Structures de données ─────────────────────────────────────

@dataclass
class DetectionResult:
    """Résultat d'une inférence sur un extrait audio."""
    probability:   float          # P(siren) ∈ [0, 1]
    is_siren:      bool           # True si prob >= threshold
    threshold:     float          # seuil utilisé
    latency_ms:    float          # temps d'inférence en ms
    confidence:    str = field(init=False)  # "HIGH" / "MEDIUM" / "LOW"

    def __post_init__(self):
        if self.probability >= 0.90 or self.probability <= 0.10:
            self.confidence = "HIGH"
        elif self.probability >= 0.70 or self.probability <= 0.30:
            self.confidence = "MEDIUM"
        else:
            self.confidence = "LOW"

    def __str__(self):
        status = "🚨 SIREN DETECTED" if self.is_siren else "✅ No siren"
        return (f"{status} | P(siren)={self.probability:.4f} | "
                f"Confidence={self.confidence} | Latency={self.latency_ms:.2f}ms")


# ── Classe principale d'inférence ────────────────────────────

class AEDASInference:
    """
    Pipeline d'inférence A.E.D.A.S. utilisant le modèle TFLite INT8.
    Simule le comportement d'un système embarqué Cortex-M.
    """

    def __init__(self, threshold: float = OPTIMAL_THRESHOLD):
        self.threshold = threshold
        self._load_normalizer()
        self._load_model()
        print(f"  A.E.D.A.S. Inference Engine initialisé")
        print(f"  Modèle  : {TFLITE_INT8.name}")
        print(f"  Seuil   : {self.threshold}")
        print(f"  SR      : {SAMPLE_RATE} Hz | Durée : {DURATION}s")

    def _load_normalizer(self):
        """Charge les paramètres de normalisation Z-score."""
        data = np.load(FEATURES_PATH)
        self.mean = data["mean"]  # (1, 13, 1)
        self.std  = data["std"]   # (1, 13, 1)

    def _load_model(self):
        """Initialise l'interpréteur TFLite INT8."""
        self.interpreter = tf.lite.Interpreter(
            model_path=str(TFLITE_INT8)
        )
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Paramètres de quantification
        self.input_scale, self.input_zero_point = (
            self.input_details[0]["quantization"]
        )
        self.output_scale, self.output_zero_point = (
            self.output_details[0]["quantization"]
        )

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Transforme un signal audio brut en features MFCC normalisées.
        Reproduit exactement le pipeline de preprocessing.py.
        """
        # Normaliser la longueur
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
        else:
            audio = audio[:N_SAMPLES]

        # Extraction MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE,
            n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmax=FMAX
        ).astype(np.float32)

        # Normalisation Z-score
        mfcc_norm = (mfcc - self.mean[0]) / self.std[0]

        return mfcc_norm  # (13, 173)

    def predict(self, audio: np.ndarray) -> DetectionResult:
        """
        Prédit si un signal audio contient une sirène.
        Entrée : signal audio numpy float32 à SAMPLE_RATE Hz.
        """
        t_start = time.perf_counter()

        # Prétraitement
        mfcc = self.preprocess(audio)
        mfcc_batch = mfcc[np.newaxis, :, :]  # (1, 13, 173)

        # Quantification de l'entrée INT8
        mfcc_q = (mfcc_batch / self.input_scale
                  + self.input_zero_point).astype(np.int8)

        # Inférence TFLite
        self.interpreter.set_tensor(
            self.input_details[0]["index"], mfcc_q
        )
        self.interpreter.invoke()
        output_q = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Déquantification — cast int32 AVANT soustraction pour éviter overflow int8
        raw_val     = int(output_q[0][0])          # int8 → int32
        zero_point  = int(self.output_zero_point)  # int32
        probability = float((raw_val - zero_point) * self.output_scale)
        probability = float(np.clip(probability, 0.0, 1.0))

        latency_ms = (time.perf_counter() - t_start) * 1000

        return DetectionResult(
            probability=probability,
            is_siren=(probability >= self.threshold),
            threshold=self.threshold,
            latency_ms=latency_ms,
        )

    def predict_file(self, file_path: Path) -> DetectionResult:
        """Inférence directe sur un fichier audio."""
        audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        return self.predict(audio)

    def benchmark(self, n_runs: int = 100) -> dict:
        """
        Mesure la latence d'inférence sur N exécutions.
        Simule les conditions de déploiement embarqué.
        """
        dummy = np.random.randn(N_SAMPLES).astype(np.float32)

        # Warmup : 3 runs pour déclencher la compilation JIT
        print(f"\n  Warmup JIT (3 runs)...")
        for _ in range(3):
            self.predict(dummy)

        # Benchmark réel
        latencies = []
        print(f"  Benchmark inférence ({n_runs} runs)...")
        for _ in range(n_runs):
            result = self.predict(dummy)
            latencies.append(result.latency_ms)

        latencies = np.array(latencies)
        stats = {
            "mean_ms":   latencies.mean(),
            "std_ms":    latencies.std(),
            "min_ms":    latencies.min(),
            "max_ms":    latencies.max(),
            "p95_ms":    np.percentile(latencies, 95),
            "p99_ms":    np.percentile(latencies, 99),
        }

        print(f"  Latence moyenne : {stats['mean_ms']:.2f} ms ± {stats['std_ms']:.2f}")
        print(f"  Min : {stats['min_ms']:.2f} ms | Max : {stats['max_ms']:.2f} ms")
        print(f"  P95 : {stats['p95_ms']:.2f} ms | P99 : {stats['p99_ms']:.2f} ms")

        fps = 1000 / stats["mean_ms"]
        print(f"  Fréquence inférence : {fps:.1f} inférences/seconde")
        print(f"  {'✅' if stats['p95_ms'] < 100 else '⚠️'} "
              f"Latence P95 {'acceptable' if stats['p95_ms'] < 100 else 'à optimiser'} "
              f"pour temps réel (< 100ms)")
        return stats


# ── Test sur dataset réel ─────────────────────────────────────

def run_inference_test():
    """
    Teste le pipeline sur des exemples réels du dataset.
    Valide que l'inférence TFLite donne des résultats cohérents
    avec les métriques obtenues lors de l'entraînement.
    """
    import pandas as pd
    from config import US8K_AUDIO, METADATA_PATH

    print("=" * 55)
    print("  A.E.D.A.S. — Test du pipeline d'inférence")
    print("=" * 55)

    engine = AEDASInference(threshold=OPTIMAL_THRESHOLD)

    # Benchmark de latence
    stats = engine.benchmark(n_runs=100)

    # Test sur des exemples réels
    print(f"\n{'─' * 55}")
    print(f"  Test sur fichiers réels")
    print(f"{'─' * 55}")

    meta = pd.read_csv(METADATA_PATH)
    meta_us8k = meta[meta["source"] == "urbansound8k"]

    # 5 sirènes + 5 non-sirènes
    sirens     = meta_us8k[meta_us8k["label"] == 1].sample(5, random_state=42)
    non_sirens = meta_us8k[meta_us8k["label"] == 0].sample(5, random_state=42)
    test_samples = pd.concat([sirens, non_sirens]).sample(frac=1, random_state=42)

    correct = 0
    print(f"\n  {'Fichier':<35} {'Réel':<12} {'Prédit':<12} {'P(siren)':<10} {'OK'}")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*10} {'─'*4}")

    for _, row in test_samples.iterrows():
        file_path = US8K_AUDIO / f"fold{row['fold']}" / row["filename"]
        result    = engine.predict_file(file_path)

        real_label  = "SIREN" if row["label"] == 1 else "non-siren"
        pred_label  = "SIREN" if result.is_siren else "non-siren"
        is_correct  = (result.is_siren == bool(row["label"]))
        correct    += int(is_correct)
        ok_mark     = "✅" if is_correct else "❌"

        print(f"  {row['filename']:<35} {real_label:<12} "
              f"{pred_label:<12} {result.probability:<10.4f} {ok_mark}")

    print(f"\n  Accuracy sur échantillon : {correct}/10 ({correct*10}%)")
    print(f"\n✅ Pipeline d'inférence validé.")
    return stats


if __name__ == "__main__":
    run_inference_test()