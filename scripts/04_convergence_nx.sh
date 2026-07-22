#!/usr/bin/env bash
## @file 04_convergence_nx.sh
## @brief Test spatial-grid convergence of the corrected n=4 response.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly BASE_DIR="results/squared_v1/convergence"
readonly -a NX_VALUES=(32 48 64 96 128)

for nx in "${NX_VALUES[@]}"; do
    out_dir="${BASE_DIR}/Nx${nx}"
    echo "Testing spatial resolution: Nx=${nx}"
    python run_scan.py \
        --spectrum diff \
        --g 0.03 \
        --harmonics 4 \
        --Nx "$nx" \
        --Nphi 64 \
        --tau_max 4.0 \
        --dt 0.02 \
        --out "$out_dir"
done
