#!/usr/bin/env bash
## @file 10_plot_tau0.sh
## @brief Plot the baseline fixed-g and matched-opacity tau0 scans.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly FIXED_ROOT="${FIXED_ROOT:-results/squared_v1/tau0/fixed_g/Nx64_Nphi64_dt0.005}"
readonly MATCHED_ROOT="${MATCHED_ROOT:-results/squared_v1/tau0/matched_opacity/Nx64_Nphi64_dt0.005}"
readonly OUT_DIR="${OUT_DIR:-plots/tau0}"

plotter="${PROJECT_ROOT}/plot_tau0_results.py"
if [[ ! -f "$plotter" ]]; then
    plotter="${PROJECT_ROOT}/scripts/plot_tau0_results.py"
fi
if [[ ! -f "$plotter" ]]; then
    echo "Cannot find plot_tau0_results.py in the project root or scripts/." >&2
    exit 1
fi

python "$plotter" \
    --fixed-root "$FIXED_ROOT" \
    --matched-root "$MATCHED_ROOT" \
    --out "$OUT_DIR"
