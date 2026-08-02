# =============================================================
#  A.E.D.A.S. — Scénarios de simulation complets
#  src/simulation/scenario.py
#
#  Définit des scénarios réalistes d'approche de véhicules
#  d'urgence avec cinématique complète, effet Doppler,
#  bruit de cabine et estimation TDOA.
# =============================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from simulation.doppler import (
    VehicleState, DopplerResult,
    compute_doppler, generate_doppler_siren
)
from simulation.acoustic_env import (
    CabinConfig, generate_cabin_audio, mix_siren_with_cabin
)

SAMPLE_RATE = 22050


@dataclass
class ScenarioResult:
    """Résultat complet d'un scénario de simulation."""
    name:           str
    description:    str
    audio_mixed:    np.ndarray      # signal final (sirène + cabine)
    audio_siren:    np.ndarray      # sirène seule (référence)
    audio_cabin:    np.ndarray      # bruit cabine seul
    doppler_frames: list            # DopplerResult par frame
    true_label:     int             # 1=siren, 0=no_siren
    snr_db:         float
    duration_s:     float


def scenario_approach_front(
    speed_emergency_kmh: float = 60.0,
    speed_host_kmh:      float = 50.0,
    initial_distance_m:  float = 150.0,
    snr_db:              float = 10.0,
    siren_type:          str   = "wail",
) -> ScenarioResult:
    """
    Scénario 1 : Approche frontale.
    Le véhicule d'urgence arrive en face du véhicule hôte,
    les deux roulent dans des directions opposées.

    Configuration :
        - Hôte : roule vers l'est (+X) à speed_host_kmh
        - Urgence : roule vers l'ouest (-X) depuis X=initial_distance
    """
    v_emergency = speed_emergency_kmh / 3.6  # m/s
    v_host      = speed_host_kmh / 3.6

    source = VehicleState(
        x=initial_distance_m, y=0.0,
        vx=-v_emergency, vy=0.0,
        label="emergency"
    )
    observer = VehicleState(
        x=0.0, y=0.0,
        vx=v_host, vy=0.0,
        label="host"
    )

    return _run_scenario(
        name="Approche Frontale",
        description=f"Urgence {speed_emergency_kmh}km/h face à hôte {speed_host_kmh}km/h, d={initial_distance_m}m",
        source=source,
        observer=observer,
        snr_db=snr_db,
        siren_type=siren_type,
    )


def scenario_overtake(
    speed_emergency_kmh: float = 100.0,
    speed_host_kmh:      float = 50.0,
    lateral_offset_m:    float = 3.5,
    snr_db:              float = 8.0,
    siren_type:          str   = "yelp",
) -> ScenarioResult:
    """
    Scénario 2 : Dépassement par derrière.
    Le véhicule d'urgence double le véhicule hôte par la gauche.

    Configuration :
        - Hôte : à l'origine, roule vers le nord (+Y)
        - Urgence : derrière (-Y) avec décalage latéral, plus rapide
    """
    v_emergency = speed_emergency_kmh / 3.6
    v_host      = speed_host_kmh / 3.6

    source = VehicleState(
        x=lateral_offset_m, y=-80.0,
        vx=0.0, vy=v_emergency,
        label="emergency"
    )
    observer = VehicleState(
        x=0.0, y=0.0,
        vx=0.0, vy=v_host,
        label="host"
    )

    return _run_scenario(
        name="Dépassement par derrière",
        description=f"Urgence {speed_emergency_kmh}km/h dépasse hôte {speed_host_kmh}km/h, décalage={lateral_offset_m}m",
        source=source,
        observer=observer,
        snr_db=snr_db,
        siren_type=siren_type,
    )


def scenario_intersection(
    speed_emergency_kmh: float = 70.0,
    speed_host_kmh:      float = 40.0,
    distance_intersection_m: float = 100.0,
    snr_db:              float = 12.0,
    siren_type:          str   = "phaser",
) -> ScenarioResult:
    """
    Scénario 3 : Croisement à une intersection.
    Le véhicule d'urgence arrive perpendiculairement.

    Configuration :
        - Hôte : roule vers le nord (+Y)
        - Urgence : arrive de l'est (+X → -X), perpendiculaire
    """
    v_emergency = speed_emergency_kmh / 3.6
    v_host      = speed_host_kmh / 3.6

    source = VehicleState(
        x=distance_intersection_m, y=0.0,
        vx=-v_emergency, vy=0.0,
        label="emergency"
    )
    observer = VehicleState(
        x=0.0, y=-distance_intersection_m,
        vx=0.0, vy=v_host,
        label="host"
    )

    return _run_scenario(
        name="Intersection perpendiculaire",
        description=f"Urgence {speed_emergency_kmh}km/h perpendiculaire, d={distance_intersection_m}m",
        source=source,
        observer=observer,
        snr_db=snr_db,
        siren_type=siren_type,
    )


def scenario_multi_emergency(
    snr_db: float = 8.0,
) -> tuple[np.ndarray, list]:
    """
    Scénario 4 : Deux véhicules d'urgence simultanément.
    Teste la robustesse du système en présence de sources multiples.
    """
    # Ambulance depuis le front
    source1 = VehicleState(x=120.0, y=0.0,  vx=-50/3.6, vy=0.0, label="ambulance")
    # Police depuis le flanc gauche
    source2 = VehicleState(x=0.0, y=80.0, vx=0.0, vy=-40/3.6, label="police")
    observer = VehicleState(x=0.0, y=0.0,   vx=40/3.6, vy=0.0, label="host")

    cabin_cfg = CabinConfig(
        vehicle_speed_kmh=80.0,
        engine_rpm=2500.0,
        road_type="asphalt",
    )

    # Génération des deux sirènes
    siren1 = generate_doppler_siren(source1, observer, siren_type="wail",  snr_db=30.0)
    siren2 = generate_doppler_siren(source2, observer, siren_type="yelp",  snr_db=30.0)
    cabin  = generate_cabin_audio(cabin_cfg)

    # Mix final
    combined_siren = siren1 * 0.6 + siren2 * 0.5
    mixed = mix_siren_with_cabin(combined_siren, cabin, snr_db)

    doppler1 = compute_doppler(source1, observer, 900.0)
    doppler2 = compute_doppler(source2, observer, 900.0)

    return mixed, [doppler1, doppler2]


def _run_scenario(
    name:       str,
    description:str,
    source:     VehicleState,
    observer:   VehicleState,
    snr_db:     float,
    siren_type: str,
    duration_s: float = 4.0,
    sr:         int   = SAMPLE_RATE,
) -> ScenarioResult:
    """Pipeline interne d'exécution d'un scénario."""
    # Configuration cabine réaliste
    cabin_cfg = CabinConfig(
        vehicle_speed_kmh=np.linalg.norm(observer.velocity) * 3.6,
        engine_rpm=2000.0 + np.linalg.norm(observer.velocity) * 20,
        road_type="asphalt",
        windows_open=False,
    )

    # Génération des signaux
    siren_signal = generate_doppler_siren(
        source, observer,
        duration_s=duration_s, sr=sr,
        siren_type=siren_type, snr_db=30.0,
    )
    cabin_signal = generate_cabin_audio(cabin_cfg, duration_s, sr)
    mixed_signal = mix_siren_with_cabin(siren_signal, cabin_signal, snr_db)

    # Calcul Doppler pour plusieurs instants
    doppler_frames = []
    n_frames = 10
    for i in range(n_frames):
        t = i * duration_s / n_frames
        src_t = VehicleState(
            x=source.x   + source.vx   * t,
            y=source.y   + source.vy   * t,
            vx=source.vx, vy=source.vy,
            label=source.label,
        )
        obs_t = VehicleState(
            x=observer.x + observer.vx * t,
            y=observer.y + observer.vy * t,
            vx=observer.vx, vy=observer.vy,
            label=observer.label,
        )
        doppler_frames.append(compute_doppler(src_t, obs_t, 900.0))

    return ScenarioResult(
        name=name,
        description=description,
        audio_mixed=mixed_signal,
        audio_siren=siren_signal,
        audio_cabin=cabin_signal,
        doppler_frames=doppler_frames,
        true_label=1,
        snr_db=snr_db,
        duration_s=duration_s,
    )