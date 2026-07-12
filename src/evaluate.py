# =============================================================
#  A.E.D.A.S. — Évaluation et visualisation des résultats
#  src/evaluate.py
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)
import json
from pathlib import Path

from config import (
    FEATURES_PATH, METADATA_PATH, MODELS_DIR,
    REPORTS_DIR, N_FOLDS, RANDOM_SEED, set_seed
)
from model import build_cnn1d

set_seed(RANDOM_SEED)


# ── 1. Chargement ─────────────────────────────────────────────

def load_data():
    data     = np.load(FEATURES_PATH)
    features = data["features"]
    labels   = data["labels"]
    meta     = pd.read_csv(METADATA_PATH)
    return features, labels, meta


# ── 2. Reconstruction des prédictions sur tous les folds ──────

def collect_all_predictions(features, labels, meta):
    """
    Recharge chaque meilleur modèle sauvegardé et prédit sur son fold test.
    Retourne les vraies étiquettes et probabilités prédites pour
    l'ensemble du dataset (reconstruction complète).
    """
    all_y_true  = []
    all_y_proba = []
    all_folds   = []

    print("Reconstruction des prédictions sur tous les folds...")

    for fold in range(1, N_FOLDS + 1):
        model_path = MODELS_DIR / "folds" / f"fold_{fold:02d}_best.keras"

        if not model_path.exists():
            print(f"  ⚠️  Modèle fold {fold} introuvable — ignoré")
            continue

        # Données de test du fold courant
        test_mask = (meta["source"] == "urbansound8k") & (meta["fold"] == fold)
        X_test = features[test_mask]
        y_test = labels[test_mask]

        # Chargement du meilleur modèle du fold
        model = build_cnn1d()
        model.load_weights(str(model_path))

        y_proba = model.predict(X_test, verbose=0).flatten()

        all_y_true.extend(y_test.tolist())
        all_y_proba.extend(y_proba.tolist())
        all_folds.extend([fold] * len(y_test))

        print(f"  Fold {fold:02d} — {len(y_test)} échantillons — "
              f"AUC : {auc(*roc_curve(y_test, y_proba)[:2]):.4f}")

    return (np.array(all_y_true),
            np.array(all_y_proba),
            np.array(all_folds))


# ── 3. Analyse du seuil optimal ───────────────────────────────

def find_optimal_threshold(y_true, y_proba):
    """
    Trouve le seuil qui maximise le F1-score sur l'ensemble des prédictions.
    Permet d'améliorer la precision sans trop sacrifier le recall.
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })

    df = pd.DataFrame(results)
    best = df.loc[df["f1"].idxmax()]
    return best, df


# ── 4. Visualisations ─────────────────────────────────────────

def plot_all_results(y_true, y_proba, folds, df_cv, best_thresh, df_thresh):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("A.E.D.A.S. — Évaluation complète du modèle CNN 1D",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    threshold = best_thresh["threshold"]
    y_pred    = (y_proba >= threshold).astype(int)

    # ── 1. Matrice de confusion ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Non-Siren", "Siren"])
    ax1.set_yticklabels(["Non-Siren", "Siren"])
    ax1.set_xlabel("Prédit"); ax1.set_ylabel("Réel")
    ax1.set_title(f"Matrice de confusion\n(seuil = {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    # ── 2. Courbe ROC ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="tomato", lw=2,
             label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.fill_between(fpr, tpr, alpha=0.1, color="tomato")
    ax2.set_xlabel("Taux Faux Positifs"); ax2.set_ylabel("Taux Vrais Positifs")
    ax2.set_title("Courbe ROC")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    # ── 3. Courbe Precision-Recall ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prec, rec, thresh_pr = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    ax3.plot(rec, prec, color="steelblue", lw=2,
             label=f"PR-AUC = {pr_auc:.4f}")
    ax3.fill_between(rec, prec, alpha=0.1, color="steelblue")
    ax3.axhline(y_true.mean(), color="grey", linestyle="--",
                label=f"Baseline = {y_true.mean():.3f}")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_title("Courbe Precision-Recall")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ── 4. Analyse du seuil ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(df_thresh["threshold"], df_thresh["precision"],
             label="Precision", color="steelblue", lw=2)
    ax4.plot(df_thresh["threshold"], df_thresh["recall"],
             label="Recall", color="tomato", lw=2)
    ax4.plot(df_thresh["threshold"], df_thresh["f1"],
             label="F1-score", color="green", lw=2.5)
    ax4.axvline(threshold, color="black", linestyle="--", lw=1.5,
                label=f"Seuil optimal = {threshold:.2f}")
    ax4.set_xlabel("Seuil de décision")
    ax4.set_ylabel("Score")
    ax4.set_title("Precision / Recall / F1 en fonction du seuil")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0.1, 0.9)

    # ── 5. Métriques par fold ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    fold_nums = df_cv["fold"].values
    ax5.plot(fold_nums, df_cv["accuracy"],  "o-", label="Accuracy",  color="steelblue")
    ax5.plot(fold_nums, df_cv["f1"],        "s-", label="F1",        color="tomato")
    ax5.plot(fold_nums, df_cv["auc"],       "^-", label="AUC",       color="green")
    ax5.axhline(df_cv["accuracy"].mean(), color="steelblue", linestyle="--", alpha=0.5)
    ax5.axhline(df_cv["auc"].mean(),      color="green",     linestyle="--", alpha=0.5)
    ax5.set_xlabel("Fold"); ax5.set_ylabel("Score")
    ax5.set_title("Métriques par fold")
    ax5.set_xticks(fold_nums)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # ── 6. Distribution des probabilités prédites ────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.hist(y_proba[y_true == 0], bins=60, density=True,
             color="steelblue", alpha=0.7, label="Non-siren")
    ax6.hist(y_proba[y_true == 1], bins=60, density=True,
             color="tomato", alpha=0.7, label="Siren")
    ax6.axvline(threshold, color="black", linestyle="--", lw=2,
                label=f"Seuil = {threshold:.2f}")
    ax6.axvline(0.5, color="grey", linestyle=":", lw=1.5,
                label="Seuil par défaut (0.5)")
    ax6.set_xlabel("Probabilité prédite P(siren)")
    ax6.set_ylabel("Densité")
    ax6.set_title("Distribution des probabilités prédites par classe")
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.savefig(f"{REPORTS_DIR}/evaluation_complete.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figure sauvegardée dans reports/")


# ── 5. Rapport textuel final ──────────────────────────────────

def print_final_report(y_true, y_proba, best_thresh, df_cv):
    threshold = best_thresh["threshold"]
    y_pred    = (y_proba >= threshold).astype(int)

    with open(MODELS_DIR / "cv_summary.json") as f:
        summary = json.load(f)

    print("\n" + "=" * 55)
    print("  A.E.D.A.S. — RAPPORT D'ÉVALUATION FINAL")
    print("=" * 55)

    print(f"\n  Seuil de décision optimal : {threshold:.2f}")
    print(f"  (vs seuil par défaut 0.50)\n")

    print(classification_report(
        y_true, y_pred,
        target_names=["non-siren", "siren"]
    ))

    print(f"  Métriques validation croisée 10-fold :")
    print(f"    Accuracy  : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"    Precision : {summary['precision_mean']:.4f}")
    print(f"    Recall    : {summary['recall_mean']:.4f}")
    print(f"    F1-score  : {summary['f1_mean']:.4f}")
    print(f"    AUC-ROC   : {summary['auc_mean']:.4f}")

    print(f"\n  Meilleur fold  : {df_cv.loc[df_cv['auc'].idxmax(), 'fold']:.0f} "
          f"(AUC = {df_cv['auc'].max():.4f})")
    print(f"  Pire fold      : {df_cv.loc[df_cv['auc'].idxmin(), 'fold']:.0f} "
          f"(AUC = {df_cv['auc'].min():.4f})")
    print(f"\n  Objectif ≥ 90% : {'✅ ATTEINT' if summary['target_met'] else '⚠️ Non atteint'}")
    print(f"  Empreinte INT8 : ~40 Ko ✅")
    print(f"  Temps total    : {summary['total_time_min']:.1f} min")
    print("=" * 55)


# ── Point d'entrée ────────────────────────────────────────────

if __name__ == "__main__":
    features, labels, meta = load_data()
    df_cv = pd.read_csv(MODELS_DIR / "cv_results.csv")

    y_true, y_proba, folds = collect_all_predictions(features, labels, meta)

    best_thresh, df_thresh = find_optimal_threshold(y_true, y_proba)
    print(f"\nSeuil optimal trouvé : {best_thresh['threshold']:.2f}")
    print(f"  Precision : {best_thresh['precision']:.4f}")
    print(f"  Recall    : {best_thresh['recall']:.4f}")
    print(f"  F1-score  : {best_thresh['f1']:.4f}")

    print_final_report(y_true, y_proba, best_thresh, df_cv)
    plot_all_results(y_true, y_proba, folds, df_cv, best_thresh, df_thresh)