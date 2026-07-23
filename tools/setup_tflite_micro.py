# =============================================================
#  A.E.D.A.S. — Setup TFLite Micro pour simulation QEMU
#  tools/setup_tflite_micro.py
#
#  Télécharge et prépare TFLite Micro + CMSIS-NN
#  pour compilation bare-metal Cortex-M7
# =============================================================

import subprocess
import sys
import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

ROOT_DIR  = Path(__file__).parent.parent
TOOLS_DIR = ROOT_DIR / "tools" / "qemu_sim"
TFLM_DIR  = TOOLS_DIR / "tflite-micro"
CMSIS_DIR = TOOLS_DIR / "CMSIS"

def download_file(url: str, dest: Path, desc: str):
    """Télécharge un fichier avec barre de progression."""
    print(f"  Téléchargement {desc}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {desc} : {min(pct, 100)}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=progress)
    print()

def setup_tflite_micro():
    """Clone TFLite Micro et extrait les sources nécessaires."""
    print("=" * 55)
    print("  A.E.D.A.S. — Setup TFLite Micro")
    print("=" * 55)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    # ── TFLite Micro via pip (plus simple que le clone git) ──
    print("\n[1/3] Installation tflite-micro Python package...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "tflite-micro", "--quiet"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✅ tflite-micro installé")
    else:
        print(f"  ⚠️  tflite-micro non disponible via pip")
        print(f"  → On utilisera TFLite runtime standard")

    # ── CMSIS headers pour Cortex-M7 ─────────────────────────
    print("\n[2/3] Téléchargement CMSIS headers...")
    cmsis_zip = TOOLS_DIR / "CMSIS.zip"
    cmsis_url = (
        "https://github.com/ARM-software/CMSIS_5/archive/"
        "refs/tags/5.9.0.zip"
    )
    try:
        download_file(cmsis_url, cmsis_zip, "CMSIS 5.9.0")
        with zipfile.ZipFile(str(cmsis_zip), 'r') as z:
            # Extraire seulement les headers Core
            members = [m for m in z.namelist()
                      if "CMSIS/Core/Include" in m]
            z.extractall(str(TOOLS_DIR), members=members[:20])
        print(f"  ✅ CMSIS headers extraits")
    except Exception as e:
        print(f"  ⚠️  CMSIS download failed : {e}")
        print(f"  → On créera des stubs manuellement")

    # ── Génération du modèle C array ─────────────────────────
    print("\n[3/3] Conversion modèle TFLite → C array...")
    _generate_model_header()
    print("  ✅ Modèle converti en C header")

    print(f"\n✅ Setup terminé dans {TOOLS_DIR}")


def _generate_model_header():
    """
    Convertit le modèle TFLite INT8 en tableau C.
    C'est la méthode standard pour embarquer un modèle
    dans le firmware d'un microcontrôleur.
    """
    model_path  = ROOT_DIR / "models" / "tflite" / "aedas_int8.tflite"
    header_path = TOOLS_DIR / "aedas_model.h"

    model_bytes = model_path.read_bytes()
    n_bytes     = len(model_bytes)

    lines = [
        "// A.E.D.A.S. — Modèle TFLite INT8 embarqué",
        "// Généré automatiquement par tools/setup_tflite_micro.py",
        "// NE PAS MODIFIER MANUELLEMENT",
        "",
        "#ifndef AEDAS_MODEL_H",
        "#define AEDAS_MODEL_H",
        "",
        "#include <stdint.h>",
        "",
        f"// Taille du modèle : {n_bytes} octets ({n_bytes/1024:.1f} Ko)",
        f"#define AEDAS_MODEL_SIZE {n_bytes}",
        "",
        "// Alignement 16 octets requis par TFLite Micro",
        "alignas(16) const uint8_t aedas_model_data[] = {",
    ]

    # Bytes du modèle en hex, 16 par ligne
    hex_lines = []
    for i in range(0, n_bytes, 16):
        chunk = model_bytes[i:i+16]
        hex_str = ", ".join(f"0x{b:02X}" for b in chunk)
        hex_lines.append(f"  {hex_str},")
    lines.extend(hex_lines)
    lines.append("};")
    lines.append("")
    lines.append("#endif // AEDAS_MODEL_H")

    header_path.write_text("\n".join(lines))
    print(f"  Modèle : {n_bytes} octets → {header_path.name}")
    print(f"  Lignes C générées : {len(hex_lines)}")


if __name__ == "__main__":
    setup_tflite_micro()