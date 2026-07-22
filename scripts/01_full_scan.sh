#!/usr/bin/env bash
## @file 01_full_scan.sh
## @brief Run the corrected exploratory full-evolution opacity scan.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly OUT_DIR="results/squared_v1/full"
readonly -a SPECTRA=(flat diff mcdiff)
readonly -a OPACITIES=(0.03 0.06 0.1 0.2 0.5 1 2 4 8)

mkdir -p "$OUT_DIR"

for spectrum in "${SPECTRA[@]}"; do
    for g in "${OPACITIES[@]}"; do
        echo "Running full evolution: spectrum=${spectrum}, g=${g}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$g" \
            --harmonics 2 3 4 \
            --Nx 64 \
            --Nphi 64 \
            --tau_max 4.0 \
            --dt 0.02 \
            --out "$OUT_DIR"
    done
done

echo "Completed $(find "$OUT_DIR" -type f -name '*.json' | wc -l) JSON records."
