#!/usr/bin/env bash
## @file 07_plot_validated.sh
## @brief Plot only the corrected full and one-hit production records.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly SOURCE_ROOT="results/squared_v1"
readonly OUT_DIR="plots/validated"

full_count="$(find "$SOURCE_ROOT/full" -type f -name '*.json' | wc -l)"
onehit_count="$(find "$SOURCE_ROOT/onehit" -type f -name '*.json' | wc -l)"

if [[ "$full_count" -ne 27 ]]; then
    echo "Expected 27 full records, found ${full_count}. Run 01_full_scan.sh first." >&2
    exit 1
fi

if [[ "$onehit_count" -ne 15 ]]; then
    echo "Expected 15 one-hit records, found ${onehit_count}. Run 03_onehit_scan.sh first." >&2
    exit 1
fi

## @brief Build a temporary input tree that excludes convergence records.
plot_input="$(mktemp -d)"
trap 'rm -rf -- "$plot_input"' EXIT
mkdir -p "$plot_input/full" "$plot_input/onehit"
cp -a "$SOURCE_ROOT/full/." "$plot_input/full/"
cp -a "$SOURCE_ROOT/onehit/." "$plot_input/onehit/"

python plot_merged_results.py \
    --results "$plot_input" \
    --out "$OUT_DIR" \
    --harmonics 2 3 4

echo "Validated plots written to ${OUT_DIR}."
