#!/usr/bin/env bash
## @file 05_convergence_dt.sh
## @brief Test timestep convergence of the corrected n=4 response.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly BASE_DIR="results/squared_v1/convergence"
readonly -a DT_VALUES=(0.04 0.02 0.01 0.005)

for dt in "${DT_VALUES[@]}"; do
    out_dir="${BASE_DIR}/dt${dt}"
    echo "Testing timestep: dt=${dt}"
    python run_scan.py \
        --spectrum diff \
        --g 0.03 \
        --harmonics 4 \
        --Nx 64 \
        --Nphi 64 \
        --tau_max 4.0 \
        --dt "$dt" \
        --out "$out_dir"
done
