# =============================================================
#  A.E.D.A.S. — Configuration centrale du pipeline
#  src/config.py
# =============================================================

from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
REPORTS_DIR     = ROOT_DIR / "reports"
MODELS_DIR      = ROOT_DIR / "models"

US8K_AUDIO      = DATA_RAW / "urbansound8k"
US8K_META       = DATA_RAW / "urbansound8k" / "UrbanSound8K.csv"
ESC50_AUDIO     = DATA_RAW / "esc50" / "audio"
ESC50_META      = DATA_RAW / "esc50" / "meta" / "esc50.csv"

FEATURES_PATH   = DATA_PROCESSED / "features.npz"
METADATA_PATH   = DATA_PROCESSED / "metadata.csv"

# ── Audio ─────────────────────────────────────────────────────
SAMPLE_RATE     = 22050   # Hz — standard librosa, compatible embarqué
DURATION        = 4.0     # secondes — durée fixe de chaque extrait
N_SAMPLES       = int(SAMPLE_RATE * DURATION)  # 88 200 samples

# ── MFCC ──────────────────────────────────────────────────────
N_MFCC          = 13      # coefficients — compact pour embarqué INT8
N_FFT           = 2048    # taille fenêtre FFT (~93ms à 22050Hz)
HOP_LENGTH      = 512     # pas entre fenêtres (~23ms, overlap ~75%)
N_MELS          = 128     # bandes Mel (pour le spectrogramme intermédiaire)
FMAX            = 8000    # fréquence max analysée (Hz) — couvre sirènes

# Dimensions de la matrice MFCC résultante
# shape = (N_MFCC, T) où T = 1 + floor(N_SAMPLES / HOP_LENGTH) = 173
MFCC_TIME_STEPS = 1 + (N_SAMPLES // HOP_LENGTH)  # 173

# ── Classes ───────────────────────────────────────────────────
LABEL_SIREN     = 1
LABEL_NON_SIREN = 0

# Classes retenues comme négatifs depuis ESC-50
ESC50_NEGATIVE_CLASSES = ["car_horn", "engine", "rain", "wind"]

# ── Entraînement ──────────────────────────────────────────────
N_FOLDS         = 10      # validation croisée 10-fold (obligatoire US8K)
RANDOM_SEED     = 42
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
EARLY_STOPPING_PATIENCE = 8

# ── Reproductibilité ──────────────────────────────────────────
import os, random, numpy as np
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fixer toutes les sources d'aléatoire pour la reproductibilité."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass