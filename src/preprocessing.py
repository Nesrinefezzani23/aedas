# =============================================================
#  A.E.D.A.S. — Extraction MFCC à grande échelle
#  src/preprocessing.py
# =============================================================

import numpy as np
import pandas as pd
import librosa
import warnings
from pathlib import Path
from tqdm import tqdm

from config import (
    US8K_AUDIO, US8K_META, ESC50_AUDIO, ESC50_META,
    FEATURES_PATH, METADATA_PATH, DATA_PROCESSED,
    SAMPLE_RATE, DURATION, N_SAMPLES,
    N_MFCC, N_FFT, HOP_LENGTH, N_MELS, FMAX, MFCC_TIME_STEPS,
    LABEL_SIREN, LABEL_NON_SIREN, ESC50_NEGATIVE_CLASSES,
    set_seed, RANDOM_SEED
)

warnings.filterwarnings("ignore")
set_seed(RANDOM_SEED)


# ── 1. Fonction d'extraction MFCC ────────────────────────────

def extract_mfcc(file_path: Path) -> np.ndarray | None:
    """
    Charge un fichier audio et extrait ses MFCC.

    - Resampling systématique à SAMPLE_RATE (22 050 Hz)
    - Padding / truncation pour normaliser à DURATION secondes
    - Retourne une matrice (N_MFCC, MFCC_TIME_STEPS) ou None si erreur
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Normaliser la longueur à exactement N_SAMPLES
        if len(y) < N_SAMPLES:
            # Padding par zéros à droite si trop court
            y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")
        else:
            # Truncation si trop long
            y = y[:N_SAMPLES]

        # Extraction MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmax=FMAX
        )

        # Vérification de la shape attendue
        assert mfcc.shape == (N_MFCC, MFCC_TIME_STEPS), \
            f"Shape inattendue : {mfcc.shape}"

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"\n⚠️  Erreur sur {file_path.name} : {e}")
        return None


# ── 2. Construction du dataset UrbanSound8K ──────────────────

def build_urbansound8k() -> tuple[list, list, list]:
    """
    Parcourt tous les folds d'UrbanSound8K.
    Retourne (features, labels, metadata_rows).
    """
    df = pd.read_csv(US8K_META)
    features, labels, meta = [], [], []

    print(f"\n📁 UrbanSound8K — {len(df)} fichiers")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="US8K"):
        file_path = US8K_AUDIO / f"fold{row['fold']}" / row["slice_file_name"]

        if not file_path.exists():
            print(f"\n⚠️  Fichier manquant : {file_path}")
            continue

        mfcc = extract_mfcc(file_path)
        if mfcc is None:
            continue

        label = LABEL_SIREN if row["class"] == "siren" else LABEL_NON_SIREN

        features.append(mfcc)
        labels.append(label)
        meta.append({
            "source":    "urbansound8k",
            "filename":  row["slice_file_name"],
            "fold":      int(row["fold"]),
            "label":     label,
            "class":     row["class"],
            "salience":  int(row["salience"]),
        })

    return features, labels, meta


# ── 3. Construction du dataset ESC-50 ────────────────────────

def build_esc50() -> tuple[list, list, list]:
    """
    Extrait les classes utiles d'ESC-50 :
    - Positifs  : siren
    - Négatifs  : car_horn, engine, rain, wind
    Retourne (features, labels, metadata_rows).
    """
    df = pd.read_csv(ESC50_META)
    useful_classes = ["siren"] + ESC50_NEGATIVE_CLASSES
    df_filtered = df[df["category"].isin(useful_classes)].reset_index(drop=True)

    features, labels, meta = [], [], []

    print(f"\n📁 ESC-50 — {len(df_filtered)} fichiers sélectionnés")

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="ESC-50"):
        file_path = ESC50_AUDIO / row["filename"]

        if not file_path.exists():
            print(f"\n⚠️  Fichier manquant : {file_path}")
            continue

        mfcc = extract_mfcc(file_path)
        if mfcc is None:
            continue

        label = LABEL_SIREN if row["category"] == "siren" else LABEL_NON_SIREN

        features.append(mfcc)
        labels.append(label)
        meta.append({
            "source":    "esc50",
            "filename":  row["filename"],
            "fold":      int(row["fold"]),
            "label":     label,
            "class":     row["category"],
            "salience":  1,  # ESC-50 = foreground par construction
        })

    return features, labels, meta


# ── 4. Pipeline principal ─────────────────────────────────────

def run_preprocessing() -> None:
    """
    Pipeline complet d'extraction :
    1. UrbanSound8K + ESC-50
    2. Normalisation (z-score par coefficient MFCC)
    3. Sauvegarde dans data/processed/
    """
    print("=" * 55)
    print("  A.E.D.A.S. — Pipeline d'extraction MFCC")
    print("=" * 55)

    # Créer le dossier de sortie si nécessaire
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Extraction
    feat_us8k, lab_us8k, meta_us8k = build_urbansound8k()
    feat_esc,  lab_esc,  meta_esc  = build_esc50()

    # Consolidation
    all_features = np.array(feat_us8k + feat_esc, dtype=np.float32)
    all_labels   = np.array(lab_us8k  + lab_esc,  dtype=np.int32)
    all_meta     = pd.DataFrame(meta_us8k + meta_esc)

    print(f"\n{'=' * 55}")
    print(f"  Extraction terminée")
    print(f"{'=' * 55}")
    print(f"  Shape features : {all_features.shape}")
    print(f"  Shape labels   : {all_labels.shape}")
    print(f"  Positifs (siren)  : {all_labels.sum()}")
    print(f"  Négatifs          : {(all_labels == 0).sum()}")
    print(f"  Ratio pos./total  : {all_labels.mean()*100:.1f}%")

    # ── Normalisation Z-score ─────────────────────────────────
    # Calculée sur l'axe temporel et l'axe samples
    # shape : (N_samples, N_MFCC, T) → mean/std sur axes (0, 2)
    print("\n  Normalisation Z-score par coefficient MFCC...")
    mean = all_features.mean(axis=(0, 2), keepdims=True)  # (1, 13, 1)
    std  = all_features.std(axis=(0, 2),  keepdims=True)  # (1, 13, 1)
    std  = np.where(std == 0, 1e-8, std)                  # éviter division par 0

    features_norm = (all_features - mean) / std

    print(f"  Mean globale (MFCC 0) : {mean[0,0,0]:.4f} → après norm : {features_norm[:,0,:].mean():.6f}")
    print(f"  Std  globale (MFCC 0) : {std[0,0,0]:.4f}  → après norm : {features_norm[:,0,:].std():.6f}")

    # ── Sauvegarde ────────────────────────────────────────────
    print(f"\n  Sauvegarde dans {DATA_PROCESSED}...")

    np.savez_compressed(
        FEATURES_PATH,
        features = features_norm,
        labels   = all_labels,
        mean     = mean,
        std      = std,
    )

    all_meta.to_csv(METADATA_PATH, index=False)

    size_mb = FEATURES_PATH.stat().st_size / 1024 / 1024
    print(f"  features.npz  : {size_mb:.1f} Mo")
    print(f"  metadata.csv  : {METADATA_PATH.stat().st_size / 1024:.1f} Ko")
    print(f"\n✅ Prétraitement terminé.")


# ── Point d'entrée ────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()