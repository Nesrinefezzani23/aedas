# =============================================================
#  A.E.D.A.S. — Build & Run simulation QEMU Cortex-M7
#  tools/qemu_sim/build_and_run.ps1
# =============================================================

$TOOLS_DIR = $PSScriptRoot
$BUILD_DIR = "$TOOLS_DIR\build"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  A.E.D.A.S. — Compilation ARM Cortex-M7" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null

# ── Compilation ──────────────────────────────────────────────
Write-Host "`n[1/3] Compilation arm-none-eabi-gcc..." -ForegroundColor Yellow

arm-none-eabi-gcc `
    -mcpu=cortex-m7 `
    -mthumb `
    -mfpu=fpv5-d16 `
    -mfloat-abi=hard `
    -O2 `
    -ffunction-sections `
    -fdata-sections `
    -Wall `
    -Wno-unused-variable `
    -std=c11 `
    -T "$TOOLS_DIR\cortex_m7.ld" `
    -Wl,--gc-sections `
    "-Wl,-Map,$BUILD_DIR\aedas.map" `
    "$TOOLS_DIR\startup.c" `
    "$TOOLS_DIR\main.c" `
    -o "$BUILD_DIR\aedas.elf"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur de compilation" -ForegroundColor Red
    exit 1
}
Write-Host "  Compilation reussie -> aedas.elf" -ForegroundColor Green

# ── Taille du binaire ─────────────────────────────────────────
Write-Host "`n[2/3] Taille du binaire..." -ForegroundColor Yellow
arm-none-eabi-size "$BUILD_DIR\aedas.elf"

# ── Lancement QEMU ────────────────────────────────────────────
Write-Host "`n[3/3] Lancement QEMU Cortex-M7..." -ForegroundColor Yellow
Write-Host "  Ctrl+A puis X pour quitter`n" -ForegroundColor Gray

& "C:\Program Files\qemu\qemu-system-arm.exe" `
    -machine mps2-an500 `
    -cpu cortex-m7 `
    -nographic `
    -serial mon:stdio `
    -kernel "$BUILD_DIR\aedas.elf" `
    -m 16M