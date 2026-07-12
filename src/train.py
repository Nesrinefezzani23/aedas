# =============================================================
#  A.E.D.A.S. — Entraînement CNN 1D avec validation croisée
#  src/train.py
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from pathlib import Path
import json, time

from config import (
    FEATURES_PATH, METADATA_PATH, MODELS_DIR,
    REPORTS_DIR, N_FOLDS, RANDOM_SEED,
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    LEARNING_RATE, set_seed
)
from model import build_cnn1d

set_seed(RANDOM_SEED)


# ── 1. Chargement des données ─────────────────────────────────

def load_data():
    print("Chargement des features...")
    data     = np.load(FEATURES_PATH)
    features = data["features"]   # (8932, 13, 173)
    labels   = data["labels"]     # (8932,)
    meta     = pd.read_csv(METADATA_PATH)
    print(f"  Features : {features.shape} | Labels : {labels.shape}")
    return features, labels, meta


# ── 2. Calcul du class weight ─────────────────────────────────

def get_class_weights(labels: np.ndarray) -> dict:
    """
    Compense le déséquilibre 10.8% siren / 89.2% non-siren.
    Keras utilisera ces poids pour pondérer la loss pendant l'entraînement.
    """
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights : {cw}")
    return cw


# ── 3. Callbacks ──────────────────────────────────────────────

def get_callbacks(fold: int, models_dir: Path) -> list:
    models_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / f"fold_{fold:02d}_best.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=4,
            mode="max",
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ── 4. Validation croisée 10-fold ─────────────────────────────

def run_cross_validation(
    features: np.ndarray,
    labels:   np.ndarray,
    meta:     pd.DataFrame,
) -> dict:
    """
    Validation croisée 10-fold en respectant les folds prédéfinis
    d'UrbanSound8K. Les fichiers ESC-50 sont alloués dans le fold
    d'entraînement à chaque itération (jamais en test isolé).
    """

    models_dir = MODELS_DIR / "folds"
    all_results = []

    print("\n" + "=" * 55)
    print("  A.E.D.A.S. — Validation croisée 10-fold")
    print("=" * 55)

    start_total = time.time()

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'─' * 55}")
        print(f"  FOLD {fold:02d} / {N_FOLDS}")
        print(f"{'─' * 55}")

        # ── Découpage train / test ────────────────────────────
        # Test  : fold courant (US8K uniquement)
        # Train : 9 autres folds US8K + tous les fichiers ESC-50
        test_mask  = (meta["source"] == "urbansound8k") & (meta["fold"] == fold)
        train_mask = ~test_mask

        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test  = features[test_mask]
        y_test  = labels[test_mask]

        print(f"  Train : {X_train.shape[0]} | "
              f"Siren train : {y_train.sum()} | "
              f"Test : {X_test.shape[0]} | "
              f"Siren test : {y_test.sum()}")

        # ── Class weights ────────────────────────────────────
        class_weights = get_class_weights(y_train)

        # ── Modèle ───────────────────────────────────────────
        set_seed(RANDOM_SEED + fold)
        model = build_cnn1d()

        # ── Entraînement ─────────────────────────────────────
        start_fold = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=get_callbacks(fold, models_dir),
            verbose=1,
        )
        elapsed = time.time() - start_fold

        # ── Évaluation ───────────────────────────────────────
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        report = classification_report(
            y_test, y_pred,
            target_names=["non-siren", "siren"],
            output_dict=True
        )

        fold_result = {
            "fold":       fold,
            "accuracy":   report["accuracy"],
            "precision":  report["siren"]["precision"],
            "recall":     report["siren"]["recall"],
            "f1":         report["siren"]["f1-score"],
            "auc":        roc_auc_score(y_test, y_pred_proba),
            "epochs_run": len(history.history["loss"]),
            "time_s":     elapsed,
        }

        print(f"\n  Résultats fold {fold:02d} :")
        print(f"    Accuracy  : {fold_result['accuracy']:.4f}")
        print(f"    Precision : {fold_result['precision']:.4f}")
        print(f"    Recall    : {fold_result['recall']:.4f}")
        print(f"    F1-score  : {fold_result['f1']:.4f}")
        print(f"    AUC-ROC   : {fold_result['auc']:.4f}")
        print(f"    Epochs    : {fold_result['epochs_run']} | Temps : {elapsed:.1f}s")

        all_results.append(fold_result)

    # ── Résumé global ─────────────────────────────────────────
    total_time = time.time() - start_total
    df_results = pd.DataFrame(all_results)

    print(f"\n{'=' * 55}")
    print(f"  RÉSULTATS FINAUX — Validation croisée 10-fold")
    print(f"{'=' * 55}")
    print(f"  Accuracy  : {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}")
    print(f"  Precision : {df_results['precision'].mean():.4f} ± {df_results['precision'].std():.4f}")
    print(f"  Recall    : {df_results['recall'].mean():.4f} ± {df_results['recall'].std():.4f}")
    print(f"  F1-score  : {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")
    print(f"  AUC-ROC   : {df_results['auc'].mean():.4f} ± {df_results['auc'].std():.4f}")
    print(f"  Temps total : {total_time/60:.1f} min")

    target_met = df_results['accuracy'].mean() >= 0.90
    print(f"\n  Objectif précision ≥ 90% : {'✅ ATTEINT' if target_met else '⚠️ Non atteint'}")

    # ── Sauvegarde des résultats ──────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = MODELS_DIR / "cv_results.csv"
    df_results.to_csv(results_path, index=False)

    summary = {
        "accuracy_mean":  df_results['accuracy'].mean(),
        "accuracy_std":   df_results['accuracy'].std(),
        "precision_mean": df_results['precision'].mean(),
        "recall_mean":    df_results['recall'].mean(),
        "f1_mean":        df_results['f1'].mean(),
        "auc_mean":       df_results['auc'].mean(),
        "total_time_min": total_time / 60,
        "target_met":     bool(target_met),
    }

    with open(MODELS_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Résultats sauvegardés dans {MODELS_DIR}")
    print(f"✅ Entraînement terminé.")

    return summary


# ── Point d'entrée ────────────────────────────────────────────

if __name__ == "__main__":
    features, labels, meta = load_data()
    summary = run_cross_validation(features, labels, meta)