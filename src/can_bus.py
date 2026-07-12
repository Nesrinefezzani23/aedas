# =============================================================
#  A.E.D.A.S. — Simulation Bus CAN Virtuel
#  src/can_bus.py
#
#  Simule les messages CAN qu'enverrait le système A.E.D.A.S.
#  à l'ECU du véhicule lors d'une détection de sirène.
#
#  IDs CAN utilisés :
#    0x1A0 — Détection sirène (probabilité + statut)
#    0x1A1 — Direction TDOA (angle + quadrant)
#    0x1A2 — Commande volume audio (baisse automatique)
#    0x1A3 — Alerte tableau de bord (HMI)
# =============================================================

import struct
import time
import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

from inference import DetectionResult
from tdoa import TDOAResult


# ── IDs et constantes CAN ─────────────────────────────────────

class CANMessageID(IntEnum):
    SIREN_DETECTION  = 0x1A0
    TDOA_DIRECTION   = 0x1A1
    AUDIO_VOLUME_CMD = 0x1A2
    HMI_ALERT        = 0x1A3


AUDIO_VOLUME_NORMAL  = 100   # % volume normal
AUDIO_VOLUME_REDUCED = 30    # % volume lors d'une détection
ALERT_DURATION_S     = 5.0   # durée d'affichage alerte HMI


# ── Structure d'un message CAN simulé ────────────────────────

@dataclass
class CANMessage:
    """Représente un message CAN virtuel."""
    msg_id:    int
    data:      bytes
    timestamp: float = field(default_factory=time.time)
    dlc:       int   = field(init=False)  # Data Length Code

    def __post_init__(self):
        self.dlc = len(self.data)
        assert self.dlc <= 8, "CAN classique : max 8 octets par message"

    def __str__(self):
        ts  = datetime.datetime.fromtimestamp(self.timestamp)
        hex_data = " ".join(f"{b:02X}" for b in self.data)
        return (f"[{ts.strftime('%H:%M:%S.%f')[:-3]}] "
                f"ID=0x{self.msg_id:03X} DLC={self.dlc} "
                f"DATA=[{hex_data}]")

    @property
    def id_name(self) -> str:
        try:
            return CANMessageID(self.msg_id).name
        except ValueError:
            return "UNKNOWN"


# ── Encodeurs de messages ─────────────────────────────────────

def encode_siren_detection(result: DetectionResult) -> CANMessage:
    """
    Message 0x1A0 — Détection sirène
    Byte 0   : statut détection (0x01 = siren, 0x00 = clear)
    Byte 1   : probabilité encodée sur 1 octet (0-255 = 0-100%)
    Byte 2   : seuil utilisé encodé (0-255)
    Byte 3   : confidence (0=LOW, 1=MEDIUM, 2=HIGH)
    Bytes 4-7: timestamp relatif (uint32, millisecondes)
    """
    status      = 0x01 if result.is_siren else 0x00
    prob_byte   = int(result.probability * 255)
    thresh_byte = int(result.threshold   * 255)
    conf_map    = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    conf_byte   = conf_map.get(result.confidence, 0)
    ts_ms       = int((time.time() % 1000) * 1000) & 0xFFFFFFFF

    data = struct.pack(">BBBBL",
                       status, prob_byte, thresh_byte,
                       conf_byte, ts_ms)
    return CANMessage(msg_id=CANMessageID.SIREN_DETECTION, data=data)


def encode_tdoa_direction(tdoa: TDOAResult) -> CANMessage:
    """
    Message 0x1A1 — Direction TDOA
    Byte 0-1 : angle en degrés × 10 encodé uint16 (0-3600 → 0.0°-360.0°)
    Byte 2   : quadrant code (0=FRONT, 1=FL, 2=LEFT, 3=RL, 4=REAR, 5=RR, 6=RIGHT, 7=FR)
    Byte 3   : confidence × 255 (uint8)
    Bytes 4-5: réservés
    Bytes 6-7: réservés
    """
    angle_encoded = int(tdoa.angle_deg * 10) & 0xFFFF
    quadrant_map  = {
        "FRONT": 0, "FRONT-LEFT": 1, "LEFT": 2, "REAR-LEFT": 3,
        "REAR":  4, "REAR-RIGHT": 5, "RIGHT": 6, "FRONT-RIGHT": 7
    }
    quad_byte = quadrant_map.get(tdoa.quadrant, 0xFF)
    conf_byte = int(tdoa.confidence * 255)

    data = struct.pack(">HBBBB", angle_encoded, quad_byte,
                       conf_byte, 0x00, 0x00)
    return CANMessage(msg_id=CANMessageID.TDOA_DIRECTION, data=data)


def encode_audio_volume(volume_percent: int) -> CANMessage:
    """
    Message 0x1A2 — Commande volume audio
    Byte 0 : volume cible (0-100%)
    Byte 1 : mode (0x00=normal, 0x01=reduced_siren, 0x02=muted)
    Bytes 2-7 : réservés
    """
    volume_byte = max(0, min(100, volume_percent))
    mode_byte   = (0x01 if volume_percent < AUDIO_VOLUME_NORMAL
                   else 0x00)
    data = struct.pack(">BB6s", volume_byte, mode_byte, b'\x00' * 6)
    return CANMessage(msg_id=CANMessageID.AUDIO_VOLUME_CMD, data=data)


def encode_hmi_alert(is_active: bool, angle_deg: float,
                     quadrant: str) -> CANMessage:
    """
    Message 0x1A3 — Alerte tableau de bord HMI
    Byte 0   : alerte active (0x01) ou inactive (0x00)
    Byte 1-2 : angle × 10 (uint16)
    Byte 3   : quadrant code
    Bytes 4-7: réservés
    """
    status        = 0x01 if is_active else 0x00
    angle_encoded = int(angle_deg * 10) & 0xFFFF
    quadrant_map  = {
        "FRONT": 0, "FRONT-LEFT": 1, "LEFT": 2, "REAR-LEFT": 3,
        "REAR":  4, "REAR-RIGHT": 5, "RIGHT": 6, "FRONT-RIGHT": 7
    }
    quad_byte = quadrant_map.get(quadrant, 0xFF)

    data = struct.pack(">BHBB3s", status, angle_encoded,
                       quad_byte, 0x00, b'\x00' * 3)
    return CANMessage(msg_id=CANMessageID.HMI_ALERT, data=data)


# ── Bus CAN virtuel ───────────────────────────────────────────

class VirtualCANBus:
    """
    Simule un bus CAN en mémoire.
    Enregistre tous les messages transmis avec leur timestamp,
    simule le comportement d'un contrôleur CAN embarqué.
    """

    def __init__(self, bitrate: int = 500_000):
        self.bitrate  = bitrate
        self.log      = []
        self.stats    = {"sent": 0, "detection_msgs": 0,
                         "tdoa_msgs": 0, "volume_msgs": 0, "hmi_msgs": 0}
        print(f"  Bus CAN virtuel initialisé — {bitrate/1000:.0f} kbps")

    def send(self, msg: CANMessage, verbose: bool = True) -> None:
        """Envoie (simule) un message CAN sur le bus."""
        self.log.append(msg)
        self.stats["sent"] += 1

        # Compteurs par type
        if   msg.msg_id == CANMessageID.SIREN_DETECTION:  self.stats["detection_msgs"] += 1
        elif msg.msg_id == CANMessageID.TDOA_DIRECTION:   self.stats["tdoa_msgs"] += 1
        elif msg.msg_id == CANMessageID.AUDIO_VOLUME_CMD: self.stats["volume_msgs"] += 1
        elif msg.msg_id == CANMessageID.HMI_ALERT:        self.stats["hmi_msgs"] += 1

        if verbose:
            print(f"  CAN TX ▶ [{msg.id_name:<18}] {msg}")

    def send_detection_event(self,
                              detection: DetectionResult,
                              tdoa: Optional[TDOAResult] = None) -> None:
        """
        Envoie la séquence complète de messages CAN
        lors d'une détection ou d'un retour à l'état normal.
        """
        # 1. Message de détection
        self.send(encode_siren_detection(detection))

        if detection.is_siren:
            # 2. Direction TDOA (si disponible)
            if tdoa:
                self.send(encode_tdoa_direction(tdoa))

            # 3. Baisse du volume audio
            self.send(encode_audio_volume(AUDIO_VOLUME_REDUCED))

            # 4. Activation alerte HMI
            angle   = tdoa.angle_deg if tdoa else 0.0
            quadrant = tdoa.quadrant if tdoa else "UNKNOWN"
            self.send(encode_hmi_alert(True, angle, quadrant))

        else:
            # Retour à l'état normal
            self.send(encode_audio_volume(AUDIO_VOLUME_NORMAL))
            self.send(encode_hmi_alert(False, 0.0, "FRONT"))

    def print_stats(self) -> None:
        print(f"\n  {'─'*45}")
        print(f"  Statistiques Bus CAN Virtuel")
        print(f"  {'─'*45}")
        print(f"  Messages totaux envoyés : {self.stats['sent']}")
        print(f"  └─ Détection            : {self.stats['detection_msgs']}")
        print(f"  └─ Direction TDOA       : {self.stats['tdoa_msgs']}")
        print(f"  └─ Volume audio         : {self.stats['volume_msgs']}")
        print(f"  └─ Alerte HMI           : {self.stats['hmi_msgs']}")
        print(f"  Bitrate simulé          : {self.bitrate/1000:.0f} kbps")


# ── Test du bus CAN ───────────────────────────────────────────

def run_can_test():
    print("=" * 55)
    print("  A.E.D.A.S. — Test Bus CAN Virtuel")
    print("=" * 55)

    bus = VirtualCANBus(bitrate=500_000)

    # Scénario 1 : détection de sirène avec direction
    print(f"\n  Scénario 1 : Détection sirène (FRONT-LEFT, 45°)")
    print(f"  {'─'*50}")

    detection_siren = DetectionResult(
        probability=0.9961,
        is_siren=True,
        threshold=0.82,
        latency_ms=7.2,
    )
    tdoa_result = TDOAResult(
        angle_deg=45.0,
        distance_est_m=None,
        tdoa_values={"FL-FR": -0.001, "FL-RL": -0.001},
        confidence=1.0,
        quadrant="FRONT-LEFT",
    )
    bus.send_detection_event(detection_siren, tdoa_result)

    # Scénario 2 : fin de détection
    print(f"\n  Scénario 2 : Fin de détection (retour état normal)")
    print(f"  {'─'*50}")

    detection_clear = DetectionResult(
        probability=0.02,
        is_siren=False,
        threshold=0.82,
        latency_ms=6.8,
    )
    bus.send_detection_event(detection_clear)

    # Scénario 3 : séquence rapide (simulation temps réel)
    print(f"\n  Scénario 3 : Séquence temps réel (10 frames)")
    print(f"  {'─'*50}")

    import numpy as np
    probs = [0.01, 0.05, 0.72, 0.91, 0.99, 0.98, 0.95, 0.88, 0.45, 0.03]
    angles = [0, 0, 35, 42, 45, 47, 50, 52, 0, 0]

    for i, (prob, angle) in enumerate(zip(probs, angles)):
        det = DetectionResult(
            probability=prob,
            is_siren=(prob >= 0.82),
            threshold=0.82,
            latency_ms=7.0,
        )
        tdoa = TDOAResult(
            angle_deg=float(angle),
            distance_est_m=None,
            tdoa_values={},
            confidence=0.95,
            quadrant="FRONT-LEFT" if angle > 0 else "FRONT",
        ) if prob >= 0.82 else None

        status = "🚨 SIREN" if det.is_siren else "  clear "
        print(f"  Frame {i+1:02d} | P={prob:.2f} | {status} | "
              f"Angle={angle}°", end="")
        if det.is_siren:
            bus.send(encode_siren_detection(det), verbose=False)
            if tdoa:
                bus.send(encode_tdoa_direction(tdoa), verbose=False)
        print()

    bus.print_stats()
    print(f"\n✅ Bus CAN virtuel validé.")
    return bus


if __name__ == "__main__":
    run_can_test()