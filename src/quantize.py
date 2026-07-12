# =============================================================
#  A.E.D.A.S. — Quantification TFLite INT8
#  src/quantize.py
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from pathlib import Path

from config import (
    FEATURES_PATH, MODELS_DIR, N_MFCC,
    MFCC_TIME_STEPS, RANDOM_SEED, set_seed
)
from model import build_cnn1d

set_seed(RANDOM_SEED)

TFLITE_DIR  = MODELS_DIR / "tflite"
TFLITE_FP32 = TFLITE_DIR / "aedas_fp32.tflite"
TFLITE_INT8 = TFLITE_DIR / "aedas_int8.tflite"


# ── 1. Chargement du meilleur modèle ─────────────────────────

def load_best_model() -> tf.keras.Model:
    """
    Charge le modèle du fold ayant obtenu la meilleure AUC.
    Fold 9 → AUC 0.9922 d'après les résultats de validation croisée.
    """
    import pandas as pd
    cv_results = pd.read_csv(MODELS_DIR / "cv_results.csv")
    best_fold  = cv_results.loc[cv_results["auc"].idxmax(), "fold"]
    best_fold  = int(best_fold)

    model_path = MODELS_DIR / "folds" / f"fold_{best_fold:02d}_best.keras"
    print(f"  Meilleur fold : {best_fold} (AUC = {cv_results['auc'].max():.4f})")
    print(f"  Chargement : {model_path}")

    model = build_cnn1d()
    model.load_weights(str(model_path))
    return model, best_fold


# ── 2. Dataset de calibration pour la quantification ─────────

def get_calibration_dataset(n_samples: int = 200):
    """
    La quantification INT8 nécessite un dataset de calibration
    pour calculer les plages de valeurs min/max de chaque couche.
    On utilise un sous-ensemble aléatoire des features extraites.
    """
    data     = np.load(FEATURES_PATH)
    features = data["features"]
    labels   = data["labels"]

    # Équilibrage : 50% siren, 50% non-siren
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_each  = n_samples // 2

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.concatenate([
        rng.choice(idx_pos, min(n_each, len(idx_pos)), replace=False),
        rng.choice(idx_neg, n_each, replace=False),
    ])
    rng.shuffle(idx)

    calib_data = features[idx].astype(np.float32)
    print(f"  Dataset calibration : {calib_data.shape} ({n_samples} échantillons)")
    return calib_data


# ── 3. Conversion FP32 ────────────────────────────────────────

def convert_fp32(model: tf.keras.Model) -> bytes:
    """Conversion basique sans quantification — référence FP32."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print(f"  Modèle FP32 : {len(tflite_model) / 1024:.1f} Ko")
    return tflite_model


# ── 4. Conversion INT8 ────────────────────────────────────────

def convert_int8(model: tf.keras.Model, calib_data: np.ndarray) -> bytes:
    """
    Quantification post-entraînement INT8 complète.
    Poids ET activations sont quantifiés en int8.
    Nécessite un dataset de calibration représentatif.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Activation de la quantification INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Dataset de calibration pour les activations
    def representative_dataset():
        for i in range(len(calib_data)):
            sample = calib_data[i:i+1]  # (1, 13, 173)
            yield [sample]

    converter.representative_dataset  = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8

    tflite_model = converter.convert()
    print(f"  Modèle INT8 : {len(tflite_model) / 1024:.1f} Ko")
    return tflite_model


# ── 5. Validation de la précision après quantification ────────

def validate_quantized_model(tflite_path: Path, calib_data: np.ndarray,
                              labels_calib: np.ndarray) -> dict:
    """
    Vérifie que la quantification n'a pas dégradé les performances.
    Compare les prédictions INT8 avec les labels réels.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = (
        input_details[0]["quantization"]
        if input_details[0]["dtype"] == np.int8
        else (1.0, 0)
    )

    predictions = []
    for i in range(len(calib_data)):
        sample = calib_data[i:i+1]

        if input_details[0]["dtype"] == np.int8:
            sample_q = (sample / input_scale + input_zero_point).astype(np.int8)
        else:
            sample_q = sample

        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            out_scale, out_zp = output_details[0]["quantization"]
            prob = float((output[0][0] - out_zp) * out_scale)
        else:
            prob = float(output[0][0])

        predictions.append(prob)

    predictions = np.array(predictions)
    y_pred = (predictions >= 0.5).astype(int)

    acc = (y_pred == labels_calib).mean()
    return {"accuracy": acc, "n_samples": len(calib_data)}


# ── 6. Pipeline principal ─────────────────────────────────────

def run_quantization():
    print("=" * 55)
    print("  A.E.D.A.S. — Quantification TFLite")
    print("=" * 55)

    TFLITE_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement
    print("\n[1/5] Chargement du meilleur modèle...")
    model, best_fold = load_best_model()

    # Dataset de calibration
    print("\n[2/5] Préparation du dataset de calibration...")
    calib_data = get_calibration_dataset(n_samples=200)

    data   = np.load(FEATURES_PATH)
    labels = data["labels"]
    rng    = np.random.default_rng(RANDOM_SEED)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    idx = np.concatenate([
        rng.choice(idx_pos, min(100, len(idx_pos)), replace=False),
        rng.choice(idx_neg, 100, replace=False),
    ])
    labels_calib = labels[idx]

    # Conversion FP32
    print("\n[3/5] Conversion FP32...")
    fp32_model = convert_fp32(model)
    TFLITE_FP32.write_bytes(fp32_model)

    # Conversion INT8
    print("\n[4/5] Conversion INT8 (quantification post-entraînement)...")
    int8_model = convert_int8(model, calib_data)
    TFLITE_INT8.write_bytes(int8_model)

    # Validation
    print("\n[5/5] Validation de la précision post-quantification...")
    metrics_fp32 = validate_quantized_model(TFLITE_FP32, calib_data, labels_calib)
    metrics_int8 = validate_quantized_model(TFLITE_INT8, calib_data, labels_calib)

    # Résumé
    fp32_size = TFLITE_FP32.stat().st_size / 1024
    int8_size = TFLITE_INT8.stat().st_size / 1024
    reduction = (1 - int8_size / fp32_size) * 100

    print(f"\n{'=' * 55}")
    print(f"  RÉSULTATS DE QUANTIFICATION")
    print(f"{'=' * 55}")
    print(f"  Modèle source (Keras FP32) : 161.8 Ko")
    print(f"  TFLite FP32                : {fp32_size:.1f} Ko")
    print(f"  TFLite INT8                : {int8_size:.1f} Ko")
    print(f"  Réduction mémoire          : {reduction:.1f}%")
    print(f"  Accuracy FP32 (calib set)  : {metrics_fp32['accuracy']*100:.2f}%")
    print(f"  Accuracy INT8 (calib set)  : {metrics_int8['accuracy']*100:.2f}%")
    delta = abs(metrics_fp32['accuracy'] - metrics_int8['accuracy']) * 100
    print(f"  Dégradation quantification : {delta:.2f}%")
    print(f"\n  {'✅' if delta < 2.0 else '⚠️'} "
          f"{'Quantification validée' if delta < 2.0 else 'Dégradation trop importante'} "
          f"(seuil acceptable : < 2%)")
    print(f"\n✅ Modèles TFLite sauvegardés dans {TFLITE_DIR}")


if __name__ == "__main__":
    run_quantization()