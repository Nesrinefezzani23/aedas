# =============================================================
#  A.E.D.A.S. — Modélisation effet Doppler
#  src/simulation/doppler.py
#
#  Modélise le changement de fréquence d'une sirène
#  en fonction de la vitesse d'approche/éloignement
#  du véhicule d'urgence par rapport à l'observateur.
#
#  Formule de Doppler :
#    f_obs = f_source × (v_son + v_obs) / (v_son - v_source)
#  où :
#    v_son    = 343 m/s (vitesse du son à 20°C)
#    v_obs    = vitesse de l'observateur (véhicule hôte)
#    v_source = vitesse de la source (véhicule d'urgence)
# =============================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

SPEED_OF_SOUND = 343.0  # m/s à 20°C


@dataclass
class VehicleState:
    """État cinématique d'un véhicule à un instant t."""
    x:         float   # position X en mètres
    y:         float   # position Y en mètres
    vx:        float   # vitesse X en m/s
    vy:        float   # vitesse Y en m/s
    label:     str     # identifiant du véhicule

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.vx**2 + self.vy**2))

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class DopplerResult:
    """Résultat du calcul Doppler pour une paire source/observateur."""
    f_source:     float   # fréquence émise par la source (Hz)
    f_observed:   float   # fréquence perçue par l'observateur (Hz)
    shift_hz:     float   # décalage en Hz
    shift_pct:    float   # décalage en %
    distance_m:   float   # distance source-observateur (m)
    approaching:  bool    # True si la source s'approche
    radial_speed: float   # vitesse radiale relative (m/s)


def compute_doppler(
    source:   VehicleState,
    observer: VehicleState,
    f_source: float = 1000.0,
) -> DopplerResult:
    """
    Calcule l'effet Doppler entre une source sonore et un observateur.

    Paramètres :
        source   : véhicule émettant le son (ambulance, pompiers...)
        observer : véhicule recevant le son (véhicule hôte)
        f_source : fréquence émise en Hz

    Retourne un DopplerResult avec la fréquence perçue et les métriques.
    """
    # Vecteur source → observateur
    rel_pos = observer.position - source.position
    distance = float(np.linalg.norm(rel_pos))

    if distance < 0.1:
        distance = 0.1  # éviter division par zéro

    # Direction radiale normalisée (source → observateur)
    radial_dir = rel_pos / distance

    # Vitesses radiales (projection sur l'axe source-observateur)
    # Positive = s'éloigne de l'observateur
    v_source_radial   = float(np.dot(source.velocity,   radial_dir))
    v_observer_radial = float(np.dot(observer.velocity, radial_dir))

    # Formule de Doppler complète
    # f_obs = f_src × (v_son + v_obs_radial) / (v_son - v_src_radial)
    denominator = SPEED_OF_SOUND - v_source_radial
    if abs(denominator) < 1.0:
        denominator = 1.0  # éviter singularité (source supersonique)

    f_observed = f_source * (SPEED_OF_SOUND + v_observer_radial) / denominator
    f_observed = max(20.0, min(20000.0, f_observed))  # clamp [20Hz, 20kHz]

    shift_hz  = f_observed - f_source
    shift_pct = (shift_hz / f_source) * 100.0

    # La source s'approche si v_source_radial > 0 (se déplace vers l'observateur)
    approaching = v_source_radial > 0

    return DopplerResult(
        f_source=f_source,
        f_observed=f_observed,
        shift_hz=shift_hz,
        shift_pct=shift_pct,
        distance_m=distance,
        approaching=approaching,
        radial_speed=v_source_radial,
    )


def generate_doppler_siren(
    source:      VehicleState,
    observer:    VehicleState,
    duration_s:  float = 4.0,
    sr:          int   = 22050,
    siren_type:  str   = "wail",
    snr_db:      float = 15.0,
) -> np.ndarray:
    """
    Génère un signal audio de sirène avec effet Doppler dynamique.
    La fréquence est mise à jour frame par frame selon la position
    relative des deux véhicules qui évoluent dans le temps.

    Types de sirènes :
        'wail'  : montée/descente lente (ambulance EU)
        'yelp'  : modulation rapide (police US)
        'phaser': décalage de phase (pompiers)

    Retourne un tableau numpy float32 de longueur duration_s × sr.
    """
    n_samples  = int(duration_s * sr)
    t          = np.linspace(0, duration_s, n_samples)
    audio      = np.zeros(n_samples, dtype=np.float32)

    # Simulation frame par frame (mise à jour toutes les 10ms)
    frame_size  = int(sr * 0.01)  # 10ms
    n_frames    = n_samples // frame_size

    # Copie des états pour simulation cinématique
    src_x, src_y = source.x, source.y
    obs_x, obs_y = observer.x, observer.y

    dt = duration_s / n_frames
    phase = 0.0

    for frame_idx in range(n_frames):
        start = frame_idx * frame_size
        end   = min(start + frame_size, n_samples)
        frame_t = t[start:end] - t[start]

        # État actuel des véhicules
        src_state = VehicleState(
            x=src_x, y=src_y,
            vx=source.vx, vy=source.vy,
            label=source.label
        )
        obs_state = VehicleState(
            x=obs_x, y=obs_y,
            vx=observer.vx, vy=observer.vy,
            label=observer.label
        )

        # Calcul Doppler pour ce frame
        doppler = compute_doppler(src_state, obs_state, f_source=900.0)

        # Atténuation en 1/r²
        attenuation = min(1.0, 30.0 / max(doppler.distance_m, 1.0))

        # Génération du signal selon le type de sirène
        f_mod = doppler.f_observed

        if siren_type == "wail":
            # Modulation lente 0.5 Hz entre f_base et f_base×1.3
            mod_t = frame_idx * dt
            f_inst = f_mod * (1.0 + 0.15 * np.sin(2 * np.pi * 0.5 * mod_t))

        elif siren_type == "yelp":
            # Modulation rapide 4 Hz
            mod_t = frame_idx * dt
            f_inst = f_mod * (1.0 + 0.20 * np.sin(2 * np.pi * 4.0 * mod_t))

        elif siren_type == "phaser":
            # Décalage progressif
            mod_t = frame_idx * dt
            f_inst = f_mod * (1.0 + 0.10 * np.sin(2 * np.pi * 2.0 * mod_t))

        else:
            f_inst = f_mod

        # Signal sinusoïdal avec continuité de phase
        signal = np.sin(2 * np.pi * f_inst * frame_t + phase)
        phase += 2 * np.pi * f_inst * (frame_t[-1] - frame_t[0])

        # Ajout des harmoniques (richesse spectrale de la sirène)
        signal += 0.3 * np.sin(4 * np.pi * f_inst * frame_t + 2*phase)
        signal += 0.1 * np.sin(6 * np.pi * f_inst * frame_t + 3*phase)

        # Application atténuation
        audio[start:end] = signal[:end-start] * attenuation

        # Mise à jour positions
        src_x += source.vx * dt
        src_y += source.vy * dt
        obs_x += observer.vx * dt
        obs_y += observer.vy * dt

    # Normalisation + ajout bruit
    max_amp = np.abs(audio).max()
    if max_amp > 0:
        audio /= max_amp

    # Bruit de fond calibré selon SNR
    signal_power = np.mean(audio**2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    audio += np.random.randn(n_samples).astype(np.float32) * np.sqrt(noise_power)

    return audio.astype(np.float32)