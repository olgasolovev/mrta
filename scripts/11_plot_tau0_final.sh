#!/usr/bin/env bash
## @file 11_plot_tau0_final.sh
## @brief Generate final matched-opacity tau0 validation figures.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly RESULTS_ROOT="${RESULTS_ROOT:-results/squared_v1/tau0/matched_opacity/Nx64_Nphi64_dt0.005}"
readonly OUT_DIR="${OUT_DIR:-plots/tau0_final}"
readonly REFERENCE_TAU0="${REFERENCE_TAU0:-0.10}"

plotter="${PROJECT_ROOT}/plot_tau0_final.py"
if [[ ! -f "$plotter" ]]; then
    plotter="${PROJECT_ROOT}/scripts/plot_tau0_final.py"
fi
if [[ ! -f "$plotter" ]]; then
    echo "Cannot find plot_tau0_final.py in the project root or scripts/." >&2
    exit 1
fi

python "$plotter" \
    --results "$RESULTS_ROOT" \
    --out "$OUT_DIR" \
    --reference-tau0 "$REFERENCE_TAU0"
