#!/usr/bin/env bash
## @file 02_linearity_g8.sh
## @brief Check centered-response linearity at the largest scanned opacity.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly OUT_DIR="results/squared_v1/full"
readonly -a SPECTRA=(flat diff mcdiff)

mkdir -p "$OUT_DIR"

for spectrum in "${SPECTRA[@]}"; do
    echo "Checking linearity: spectrum=${spectrum}, g=8"
    python run_scan.py \
        --spectrum "$spectrum" \
        --g 8 \
        --harmonics 2 3 4 \
        --Nx 64 \
        --Nphi 64 \
        --tau_max 4.0 \
        --dt 0.02 \
        --linearity-check \
        --out "$OUT_DIR"
done

grep -R '"linearity_dev"' -A4 "$OUT_DIR" || true
