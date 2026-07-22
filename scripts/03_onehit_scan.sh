#!/usr/bin/env bash
## @file 03_onehit_scan.sh
## @brief Generate corrected dilute one-hit reference data.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly OUT_DIR="results/squared_v1/onehit"
readonly -a SPECTRA=(flat diff mcdiff)
readonly -a OPACITIES=(0.01 0.02 0.03 0.06 0.1)

mkdir -p "$OUT_DIR"

for spectrum in "${SPECTRA[@]}"; do
    for g in "${OPACITIES[@]}"; do
        echo "Running one-hit reference: spectrum=${spectrum}, g=${g}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$g" \
            --harmonics 2 3 4 \
            --Nx 64 \
            --Nphi 64 \
            --tau_max 4.0 \
            --dt 0.02 \
            --one-hit \
            --out "$OUT_DIR"
    done
done

echo "Completed $(find "$OUT_DIR" -type f -name '*.json' | wc -l) JSON records."
