# =============================================================
#  A.E.D.A.S. — Environnement acoustique de cabine
#  src/simulation/acoustic_env.py
#
#  Modélise le bruit de fond réaliste à l'intérieur
#  d'un véhicule en mouvement :
#    - Bruit moteur (fonction de la vitesse)
#    - Bruit de vent (aérodynamique)
#    - Bruit de roulement (surface de route)
#    - Audio de bord (musique, GPS, conversation)
#    - Réverbération cabine (acoustique intérieure)
# =============================================================

import numpy as np
from dataclasses import dataclass
from scipy import signal as scipy_signal
from typing import Optional

# Fréquence d'échantillonnage standard
SAMPLE_RATE = 22050


@dataclass
class CabinConfig:
    """Configuration de l'environnement acoustique de la cabine."""
    vehicle_speed_kmh:  float = 80.0    # vitesse du véhicule hôte
    engine_rpm:         float = 2500.0  # régime moteur
    road_type:          str   = "asphalt"  # asphalt, cobblestone, gravel
    cabin_volume_m3:    float = 3.0     # volume approximatif habitacle
    windows_open:       bool  = False   # vitres ouvertes/fermées
    audio_volume_pct:   float = 0.0     # volume audio de bord (0-100%)
    passengers:         int   = 1       # nombre de passagers


def generate_engine_noise(
    config:    CabinConfig,
    duration_s: float = 4.0,
    sr:        int    = SAMPLE_RATE,
) -> np.ndarray:
    """
    Génère le bruit moteur en fonction du régime RPM.
    Modèle : fondamentale à RPM/60 Hz + harmoniques impaires.
    """
    n_samples = int(duration_s * sr)
    t         = np.linspace(0, duration_s, n_samples)
    noise     = np.zeros(n_samples)

    # Fréquence fondamentale moteur (Hz)
    f0 = config.engine_rpm / 60.0

    # Harmoniques du moteur 4 cylindres (impaires dominantes)
    harmonics = [
        (1.0, 1.0),   # fondamentale
        (2.0, 0.6),   # 2e harmonique
        (3.0, 0.3),   # 3e
        (4.0, 0.15),  # 4e
        (6.0, 0.08),  # 6e
    ]

    for harmonic, amplitude in harmonics:
        # Phase aléatoire pour réalisme
        phase = np.random.uniform(0, 2 * np.pi)
        noise += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + phase)

    # Bruit de fond moteur (bruit blanc filtré passe-bas)
    white = np.random.randn(n_samples)
    b, a  = scipy_signal.butter(4, 500 / (sr/2), btype='low')
    filtered_noise = scipy_signal.lfilter(b, a, white)
    noise += 0.2 * filtered_noise

    # Amplitude fonction de la vitesse et du régime
    amplitude_factor = 0.3 + (config.engine_rpm / 6000.0) * 0.5
    noise *= amplitude_factor

    return noise.astype(np.float32)


def generate_wind_noise(
    config:     CabinConfig,
    duration_s: float = 4.0,
    sr:         int   = SAMPLE_RATE,
) -> np.ndarray:
    """
    Génère le bruit aérodynamique proportionnel à v².
    Le bruit de vent est un bruit large bande avec un pic
    autour de 200-800 Hz selon la vitesse.
    """
    n_samples = int(duration_s * sr)

    # Bruit blanc
    white = np.random.randn(n_samples)

    # Filtrage : bande passante 100-1000 Hz (bruit de vent typique)
    v = config.vehicle_speed_kmh / 3.6  # m/s
    f_center = max(100.0, min(800.0, 50.0 + v * 5.0))

    b_low,  a_low  = scipy_signal.butter(2, f_center * 2 / (sr/2), btype='low')
    b_high, a_high = scipy_signal.butter(2, 80.0 / (sr/2), btype='high')

    noise = scipy_signal.lfilter(b_low,  a_low,  white)
    noise = scipy_signal.lfilter(b_high, a_high, noise)

    # Amplitude ∝ v² (loi aérodynamique)
    amplitude = 0.05 * (v / 30.0) ** 2

    # Vitres ouvertes → 3x plus de bruit de vent
    if config.windows_open:
        amplitude *= 3.0

    return (noise * amplitude).astype(np.float32)


def generate_road_noise(
    config:     CabinConfig,
    duration_s: float = 4.0,
    sr:         int   = SAMPLE_RATE,
) -> np.ndarray:
    """
    Génère le bruit de roulement selon le type de surface.
    """
    n_samples = int(duration_s * sr)

    # Profils spectraux par type de route
    road_profiles = {
        "asphalt":    {"f_low": 50,  "f_high": 400,  "amplitude": 0.08},
        "cobblestone":{"f_low": 30,  "f_high": 800,  "amplitude": 0.25},
        "gravel":     {"f_low": 100, "f_high": 2000, "amplitude": 0.40},
        "highway":    {"f_low": 80,  "f_high": 300,  "amplitude": 0.05},
    }

    profile = road_profiles.get(config.road_type, road_profiles["asphalt"])

    white = np.random.randn(n_samples)
    b, a  = scipy_signal.butter(
        2,
        [profile["f_low"] / (sr/2), profile["f_high"] / (sr/2)],
        btype='band'
    )
    noise = scipy_signal.lfilter(b, a, white)

    # Amplitude ∝ vitesse
    v = config.vehicle_speed_kmh / 3.6
    amplitude = profile["amplitude"] * (v / 30.0)

    return (noise * amplitude).astype(np.float32)


def generate_cabin_reverb(
    signal:    np.ndarray,
    config:    CabinConfig,
    sr:        int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Applique une réverbération de cabine simplifiée.
    Modèle : série de réflexions avec délais et atténuations.
    """
    # Temps de réverbération RT60 typique d'une cabine (0.1-0.3s)
    rt60 = 0.15 if not config.windows_open else 0.05

    # Premières réflexions (early reflections)
    delays_ms    = [8, 15, 23, 31]   # ms
    attenuations = [0.7, 0.5, 0.35, 0.25]

    output = signal.copy()
    for delay_ms, atten in zip(delays_ms, attenuations):
        delay_samples = int(delay_ms * sr / 1000)
        if delay_samples < len(signal):
            output[delay_samples:] += signal[:-delay_samples] * atten

    # Normalisation pour éviter la saturation
    max_amp = np.abs(output).max()
    if max_amp > 1.0:
        output /= max_amp

    return output.astype(np.float32)


def generate_cabin_audio(
    config:     CabinConfig,
    duration_s: float = 4.0,
    sr:         int   = SAMPLE_RATE,
) -> np.ndarray:
    """
    Génère le mix complet de bruit de cabine :
    moteur + vent + roulement + réverbération.
    """
    engine = generate_engine_noise(config, duration_s, sr)
    wind   = generate_wind_noise(config, duration_s, sr)
    road   = generate_road_noise(config, duration_s, sr)

    # Mix des composantes
    cabin_noise = engine + wind + road

    # Réverbération cabine
    cabin_noise = generate_cabin_reverb(cabin_noise, config, sr)

    # Normalisation finale
    max_amp = np.abs(cabin_noise).max()
    if max_amp > 0:
        cabin_noise = cabin_noise / max_amp * 0.6  # laisser de la marge

    return cabin_noise.astype(np.float32)


def mix_siren_with_cabin(
    siren_signal:  np.ndarray,
    cabin_noise:   np.ndarray,
    snr_target_db: float = 10.0,
) -> np.ndarray:
    """
    Mélange le signal de sirène avec le bruit de cabine
    à un SNR cible défini.

    SNR = 10 log10(P_signal / P_bruit)
    → amplitude_bruit = amplitude_signal / 10^(SNR/20)
    """
    n = min(len(siren_signal), len(cabin_noise))
    siren = siren_signal[:n]
    cabin = cabin_noise[:n]

    # Puissances
    p_siren = np.mean(siren**2)
    p_cabin = np.mean(cabin**2)

    if p_cabin < 1e-10:
        return siren.astype(np.float32)

    # Facteur de scaling pour atteindre le SNR cible
    snr_linear = 10 ** (snr_target_db / 10)
    scale = np.sqrt(p_siren / (p_cabin * snr_linear))

    mixed = siren + cabin * scale

    # Normalisation finale
    max_amp = np.abs(mixed).max()
    if max_amp > 0:
        mixed /= max_amp

    return mixed.astype(np.float32)