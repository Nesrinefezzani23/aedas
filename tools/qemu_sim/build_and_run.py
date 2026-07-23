# =============================================================
#  A.E.D.A.S. — Build & Run simulation QEMU Cortex-M7
#  tools/qemu_sim/build_and_run.py
# =============================================================

import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
BUILD_DIR = TOOLS_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)

QEMU_EXE = r"C:\Program Files\qemu\qemu-system-arm.exe"
ELF_FILE = BUILD_DIR / "aedas.elf"
MAP_FILE = BUILD_DIR / "aedas.map"

print("=" * 50)
print("  A.E.D.A.S. — Compilation ARM Cortex-M7")
print("=" * 50)

# ── Compilation ──────────────────────────────────────────────
print("\n[1/3] Compilation arm-none-eabi-gcc...")

gcc_cmd = [
    "arm-none-eabi-gcc",
    "-mcpu=cortex-m3",
    "-mthumb",
    "-O2",
    "-ffunction-sections",
    "-fdata-sections",
    "-Wall",
    "-Wno-unused-variable",
    "-std=c11",
    f"-T{TOOLS_DIR / 'cortex_m7.ld'}",
    "-Wl,--gc-sections",
    f"-Wl,-Map,{MAP_FILE}",
    str(TOOLS_DIR / "startup.c"),
    str(TOOLS_DIR / "main.c"),
    "-o", str(ELF_FILE),
]

result = subprocess.run(gcc_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("ERREUR de compilation :")
    print(result.stderr)
    sys.exit(1)

if result.stderr:
    print("Warnings :", result.stderr)

print(f"  Compilation reussie -> {ELF_FILE.name}")

# ── Taille du binaire ─────────────────────────────────────────
print("\n[2/3] Taille du binaire...")
size_result = subprocess.run(
    ["arm-none-eabi-size", str(ELF_FILE)],
    capture_output=True, text=True
)
print(size_result.stdout)

# ── Lancement QEMU ────────────────────────────────────────────
print("\n[3/3] Lancement QEMU Cortex-M7...")
print("  Ctrl+A puis X pour quitter\n")

qemu_cmd = [
    QEMU_EXE,
    "-machine", "lm3s6965evb",
    "-cpu",     "cortex-m3",
    "-nographic",
    "-serial",  "mon:stdio",
    "-kernel",  str(ELF_FILE),
    "-m",       "16M",
]

subprocess.run(qemu_cmd)