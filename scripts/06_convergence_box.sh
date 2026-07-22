#!/usr/bin/env bash
## @file 06_convergence_box.sh
## @brief Test finite-box sensitivity of the corrected n=4 response.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly BASE_DIR="results/squared_v1/convergence"
readonly -a L_VALUES=(8 10 12)

for box_size in "${L_VALUES[@]}"; do
    out_dir="${BASE_DIR}/L${box_size}"
    echo "Testing half-box size: L=${box_size}"
    python run_scan.py \
        --spectrum diff \
        --g 0.03 \
        --harmonics 4 \
        --Nx 96 \
        --Nphi 64 \
        --L "$box_size" \
        --tau_max 4.0 \
        --dt 0.02 \
        --out "$out_dir"
done
