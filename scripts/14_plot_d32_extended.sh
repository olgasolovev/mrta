#!/usr/bin/env bash
## @file 14_plot_d32_extended.sh
## @brief Merge baseline and extended records into the D_32 completion plot.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly BASELINE_ROOT="${BASELINE_ROOT:-results/squared_v1/full}"
readonly EXTENDED_ROOT="${EXTENDED_ROOT:-results/squared_v1/extended_g/full}"
readonly OUT_DIR="${OUT_DIR:-plots/extended_g}"

plotter="${PROJECT_ROOT}/plot_d32_extended.py"
if [[ ! -f "$plotter" ]]; then
    plotter="${PROJECT_ROOT}/scripts/plot_d32_extended.py"
fi
if [[ ! -f "$plotter" ]]; then
    echo "Cannot find plot_d32_extended.py in the project root or scripts/." >&2
    exit 1
fi

python "$plotter" \
    --results "$BASELINE_ROOT" "$EXTENDED_ROOT" \
    --out "$OUT_DIR"
