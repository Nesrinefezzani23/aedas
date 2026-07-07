# =============================================================
#  A.E.D.A.S. — Architecture CNN 1D
#  src/model.py
# =============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import os

# Supprimer les logs TensorFlow verbeux
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import (
    N_MFCC, MFCC_TIME_STEPS,
    LEARNING_RATE, RANDOM_SEED
)


def build_cnn1d(
    input_shape: tuple = (N_MFCC, MFCC_TIME_STEPS),
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = 0.4,
    l2_lambda: float = 1e-4,
) -> keras.Model:
    """
    CNN 1D pour classification binaire siren / non-siren.

    Architecture :
    ┌─────────────────────────────────────────────────────┐
    │  Input  (13, 173)  — MFCC normalisés                │
    │  ↓ Permute → (173, 13)  — Conv1D lit la dim temps   │
    │  ↓ Conv1D(32, k=3) + BN + ReLU                      │
    │  ↓ Conv1D(64, k=3) + BN + ReLU                      │
    │  ↓ MaxPooling1D(2)                                   │
    │  ↓ Dropout(0.4)                                      │
    │  ↓ Conv1D(128, k=3) + BN + ReLU                     │
    │  ↓ GlobalAveragePooling1D  — aggrège la dim temps    │
    │  ↓ Dense(64) + ReLU + Dropout(0.4)                  │
    │  ↓ Dense(1)  + Sigmoid  — sortie binaire             │
    └─────────────────────────────────────────────────────┘

    Choix techniques :
    - Conv1D sur l'axe temporel : chaque frame MFCC est un vecteur
      de 13 features, le CNN apprend les patterns temporels.
    - BatchNormalization : stabilise l'entraînement, réduit le besoin
      de LR fine-tuning.
    - GlobalAveragePooling au lieu de Flatten : réduit l'overfitting
      et l'empreinte mémoire embarquée.
    - L2 regularization : pénalise les grands poids pour la robustesse.
    """
    tf.random.set_seed(RANDOM_SEED)

    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    # (batch, 13, 173) → (batch, 173, 13)
    # Conv1D opère sur la dernière dim comme "features", l'avant-dernière comme "steps"
    x = layers.Permute((2, 1), name="permute")(inputs)

    # ── Bloc 1 ──────────────────────────────────────────────
    x = layers.Conv1D(
        32, kernel_size=3, padding="same",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv1"
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)

    # ── Bloc 2 ──────────────────────────────────────────────
    x = layers.Conv1D(
        64, kernel_size=3, padding="same",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # ── Bloc 3 ──────────────────────────────────────────────
    x = layers.Conv1D(
        128, kernel_size=3, padding="same",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # ── Agrégation temporelle ────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Classifieur ─────────────────────────────────────────
    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="dense1"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AEDAS_CNN1D")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def print_model_summary(model: keras.Model) -> None:
    """Affiche le summary et les métriques d'empreinte mémoire estimée."""
    model.summary()

    total_params = model.count_params()
    size_fp32_kb = total_params * 4 / 1024
    size_int8_kb = total_params * 1 / 1024

    print(f"\n{'─' * 50}")
    print(f"  Paramètres totaux     : {total_params:,}")
    print(f"  Empreinte FP32        : {size_fp32_kb:.1f} Ko")
    print(f"  Empreinte INT8 (est.) : {size_int8_kb:.1f} Ko")
    print(f"  Compatible embarqué   : {'✅ Oui' if size_int8_kb < 512 else '⚠️ À optimiser'}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    model = build_cnn1d()
    print_model_summary(model)