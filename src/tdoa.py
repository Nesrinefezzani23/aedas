# =============================================================
#  A.E.D.A.S. — Estimation de direction par TDOA
#  src/tdoa.py
#
#  TDOA = Time Difference Of Arrival
#  Principe : une sirène distante atteint les microphones
#  à des instants légèrement différents selon sa direction.
#  En mesurant ces délais, on estime l'angle d'approche.
# =============================================================

import numpy as np
from scipy.signal import correlate, correlation_lags
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path

from config import SAMPLE_RATE, REPORTS_DIR


# ── Constantes physiques ──────────────────────────────────────

SPEED_OF_SOUND = 343.0  # m/s à 20°C

# Géométrie des microphones simulés sur le toit du véhicule
# Référentiel : centre du toit, axe X = avant du véhicule
#
#   MIC_FL ──────── MIC_FR
#     |       TOIT      |
#   MIC_RL ──────── MIC_RR
#
# Coordonnées (x, y) en mètres
MICROPHONE_POSITIONS = {
    "FL": np.array([ 0.60,  0.75]),   # Front Left
    "FR": np.array([ 0.60, -0.75]),   # Front Right
    "RL": np.array([-0.60,  0.75]),   # Rear Left
    "RR": np.array([-0.60, -0.75]),   # Rear Right
}

# Paires de microphones pour le calcul TDOA
MIC_PAIRS = [
    ("FL", "FR"),   # axe latéral avant → détecte gauche/droite
    ("RL", "RR"),   # axe latéral arrière
    ("FL", "RL"),   # axe longitudinal gauche → détecte avant/arrière
    ("FR", "RR"),   # axe longitudinal droit
]


# ── Structures de données ─────────────────────────────────────

@dataclass
class TDOAResult:
    """Résultat de l'estimation de direction."""
    angle_deg:      float           # angle estimé en degrés (0=avant, 90=gauche)
    distance_est_m: Optional[float] # distance estimée (si possible)
    tdoa_values:    dict            # délais mesurés par paire de micros
    confidence:     float           # score de confiance [0, 1]
    quadrant:       str             # "FRONT", "REAR", "LEFT", "RIGHT", "FRONT-LEFT", etc.

    def __str__(self):
        return (f"Direction : {self.angle_deg:.1f}° ({self.quadrant}) | "
                f"Confiance : {self.confidence:.2f} | "
                f"Distance est. : {self.distance_est_m:.0f}m"
                if self.distance_est_m else
                f"Direction : {self.angle_deg:.1f}° ({self.quadrant}) | "
                f"Confiance : {self.confidence:.2f}")


# ── Fonctions TDOA ────────────────────────────────────────────

def compute_gcc_phat(sig1: np.ndarray, sig2: np.ndarray,
                     sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """
    GCC-PHAT : Generalized Cross-Correlation with Phase Transform.
    Méthode robuste pour estimer le délai entre deux signaux
    en présence de bruit (supérieure à la cross-corrélation classique).

    Retourne (tdoa_seconds, confidence_score).
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # FFT des deux signaux
    S1 = np.fft.rfft(sig1, n=n_fft)
    S2 = np.fft.rfft(sig2, n=n_fft)

    # Cross-spectre avec pondération PHAT
    # PHAT normalise par la magnitude → insensible à l'amplitude
    cross_spectrum = S1 * np.conj(S2)
    magnitude = np.abs(cross_spectrum)
    magnitude = np.where(magnitude < 1e-10, 1e-10, magnitude)
    gcc_phat  = cross_spectrum / magnitude

    # Corrélation dans le domaine temporel
    gcc = np.fft.irfft(gcc_phat, n=n_fft)
    gcc = np.concatenate([gcc[n_fft//2:], gcc[:n_fft//2]])

    # Indice du maximum = délai estimé
    max_idx  = np.argmax(np.abs(gcc))
    center   = n_fft // 2
    delay_samples = max_idx - center
    tdoa_s   = delay_samples / sr

    # Score de confiance basé sur le pic de corrélation
    peak_val = np.abs(gcc[max_idx])
    mean_val = np.mean(np.abs(gcc))
    confidence = min(peak_val / (mean_val * 10 + 1e-8), 1.0)

    return tdoa_s, float(confidence)


def estimate_direction(mic_signals: dict[str, np.ndarray],
                       sr: int = SAMPLE_RATE) -> TDOAResult:
    """
    Estime la direction d'approche d'une source sonore
    à partir de signaux multi-microphones.

    Paramètres :
        mic_signals : dict {mic_name: signal_array}
                      Signaux audio synchronisés de chaque microphone.
        sr          : fréquence d'échantillonnage

    Retourne un TDOAResult avec l'angle estimé.
    """
    tdoa_values = {}
    confidences = []

    # Calcul TDOA pour chaque paire de microphones
    for mic_a, mic_b in MIC_PAIRS:
        if mic_a not in mic_signals or mic_b not in mic_signals:
            continue

        tdoa_s, conf = compute_gcc_phat(
            mic_signals[mic_a],
            mic_signals[mic_b],
            sr=sr
        )
        tdoa_values[f"{mic_a}-{mic_b}"] = tdoa_s
        confidences.append(conf)

    # Estimation de l'angle par les délais latéral et longitudinal
    # TDOA latéral (FL-FR) → composante gauche/droite
    # TDOA longitudinal (FL-RL) → composante avant/arrière
    tdoa_lat  = -tdoa_values.get("FL-FR", 0.0)   # signe corrigé (convention GCC-PHAT)
    tdoa_long = -tdoa_values.get("FL-RL", 0.0)   # signe corrigé (convention GCC-PHAT)

    # Distance inter-microphone
    d_lat  = np.linalg.norm(
        MICROPHONE_POSITIONS["FL"] - MICROPHONE_POSITIONS["FR"]
    )
    d_long = np.linalg.norm(
        MICROPHONE_POSITIONS["FL"] - MICROPHONE_POSITIONS["RL"]
    )

    # Conversion TDOA → angle (modèle champ lointain)
    sin_lat  = np.clip(tdoa_lat  * SPEED_OF_SOUND / d_lat,  -1, 1)
    sin_long = np.clip(tdoa_long * SPEED_OF_SOUND / d_long, -1, 1)

    angle_lat  = np.degrees(np.arcsin(sin_lat))   # angle par rapport à l'axe Y
    angle_long = np.degrees(np.arcsin(sin_long))  # angle par rapport à l'axe X

    # Angle final : combinaison des deux estimations
    angle_deg = float(np.degrees(np.arctan2(sin_lat, sin_long)))
    if angle_deg < 0:
        angle_deg += 360

    # Quadrant
    quadrant = _get_quadrant(angle_deg)

    # Confiance globale
    conf_global = float(np.mean(confidences)) if confidences else 0.0

    return TDOAResult(
        angle_deg=angle_deg,
        distance_est_m=None,
        tdoa_values=tdoa_values,
        confidence=conf_global,
        quadrant=quadrant,
    )


def _get_quadrant(angle_deg: float) -> str:
    """Convertit un angle en quadrant lisible."""
    a = angle_deg % 360
    if   a < 22.5  or a >= 337.5: return "FRONT"
    elif a < 67.5:                 return "FRONT-LEFT"
    elif a < 112.5:                return "LEFT"
    elif a < 157.5:                return "REAR-LEFT"
    elif a < 202.5:                return "REAR"
    elif a < 247.5:                return "REAR-RIGHT"
    elif a < 292.5:                return "RIGHT"
    else:                          return "FRONT-RIGHT"


# ── Simulation de signaux multi-microphones ───────────────────

def simulate_mic_signals(source_angle_deg: float,
                         source_distance_m: float = 50.0,
                         duration_s: float = 1.0,
                         sr: int = SAMPLE_RATE,
                         snr_db: float = 10.0) -> dict[str, np.ndarray]:
    """
    Génère des signaux simulés pour chaque microphone
    en fonction d'une source sonore à une direction et distance données.

    Modèle : source ponctuelle en champ libre, bruit blanc additif.
    """
    n_samples = int(duration_s * sr)
    angle_rad = np.radians(source_angle_deg)

    # Position de la source
    source_pos = np.array([
        source_distance_m * np.cos(angle_rad),
        source_distance_m * np.sin(angle_rad),
    ])

    # Signal source : mélange de sinusoïdes simulant une sirène
    t = np.linspace(0, duration_s, n_samples)
    # Sirène wail : fréquence modulée entre 700 et 1200 Hz
    freq_mod = 700 + 500 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
    source_signal = np.sin(2 * np.pi * np.cumsum(freq_mod) / sr)

    # Puissance du signal pour calculer le bruit
    signal_power = np.mean(source_signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))

    mic_signals = {}
    for mic_name, mic_pos in MICROPHONE_POSITIONS.items():
        # Distance source → microphone
        dist = np.linalg.norm(source_pos - mic_pos)

        # Délai de propagation
        delay_s       = dist / SPEED_OF_SOUND
        delay_samples = int(delay_s * sr)

        # Signal retardé + atténuation en 1/r
        attenuation = source_distance_m / dist
        signal_delayed = np.zeros(n_samples)
        if delay_samples < n_samples:
            signal_delayed[delay_samples:] = (
                source_signal[:n_samples - delay_samples] * attenuation
            )

        # Ajout de bruit blanc
        noise = np.random.randn(n_samples) * np.sqrt(noise_power)
        mic_signals[mic_name] = (signal_delayed + noise).astype(np.float32)

    return mic_signals


# ── Test et visualisation ─────────────────────────────────────

def run_tdoa_test():
    """
    Teste l'estimation TDOA sur plusieurs angles simulés
    et visualise les résultats.
    """
    print("=" * 55)
    print("  A.E.D.A.S. — Test TDOA multi-microphones")
    print("=" * 55)

    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    results = []

    print(f"\n  {'Angle réel':>12} {'Angle estimé':>14} "
          f"{'Erreur':>8} {'Quadrant':<14} {'Conf.'}")
    print(f"  {'─'*12} {'─'*14} {'─'*8} {'─'*14} {'─'*6}")

    np.random.seed(42)
    for angle_true in test_angles:
        mic_signals = simulate_mic_signals(
            source_angle_deg=angle_true,
            source_distance_m=30.0,
            snr_db=15.0
        )
        result = estimate_direction(mic_signals)
        error  = abs(result.angle_deg - angle_true)
        error  = min(error, 360 - error)  # erreur circulaire

        results.append({
            "angle_true": angle_true,
            "angle_est":  result.angle_deg,
            "error":      error,
            "quadrant":   result.quadrant,
            "confidence": result.confidence,
        })

        print(f"  {angle_true:>10}°  {result.angle_deg:>12.1f}°  "
              f"{error:>6.1f}°  {result.quadrant:<14}  {result.confidence:.2f}")

    mae = np.mean([r["error"] for r in results])
    print(f"\n  Erreur angulaire moyenne (MAE) : {mae:.1f}°")
    print(f"  {'✅' if mae < 30 else '⚠️'} "
          f"{'Estimation acceptable' if mae < 30 else 'A améliorer'} (seuil < 30°)")

    # Visualisation polaire
    _plot_tdoa_results(results)
    return results


def _plot_tdoa_results(results: list):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("A.E.D.A.S. — Estimation TDOA : angles réels vs estimés",
                 fontsize=13, fontweight="bold")

    # Graphe polaire
    ax1 = fig.add_subplot(121, projection="polar")
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)

    for r in results:
        true_rad = np.radians(r["angle_true"])
        est_rad  = np.radians(r["angle_est"])
        ax1.plot([true_rad], [1.0], "bo", markersize=10, zorder=3)
        ax1.plot([est_rad],  [0.8], "r^", markersize=8,  zorder=3)
        ax1.plot([true_rad, est_rad], [1.0, 0.8], "k--", alpha=0.4, lw=1)

    ax1.plot([], [], "bo", label="Réel")
    ax1.plot([], [], "r^", label="Estimé")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax1.set_title("Diagramme polaire", pad=15)

    # Positions microphones
    ax2 = fig.add_subplot(122)
    for name, pos in MICROPHONE_POSITIONS.items():
        ax2.plot(pos[0], pos[1], "ks", markersize=12, zorder=3)
        ax2.annotate(f" {name}", pos, fontsize=10, fontweight="bold")
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax2.axvline(0, color="grey", lw=0.8, linestyle="--")
    ax2.set_xlabel("X (m) — axe avant/arrière")
    ax2.set_ylabel("Y (m) — axe gauche/droite")
    ax2.set_title("Géométrie des microphones (toit véhicule)")
    ax2.set_aspect("equal")
    ax2.annotate("← AVANT →", xy=(0, 1.3), ha="center", fontsize=9, color="steelblue")
    ax2.annotate("← ARRIÈRE →", xy=(0, -1.3), ha="center", fontsize=9, color="steelblue")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/tdoa_estimation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figure TDOA sauvegardée dans reports/")


if __name__ == "__main__":
    run_tdoa_test()