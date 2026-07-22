#!/usr/bin/env bash
## @file 13_extended_g_validation.sh
## @brief Validate linearity and timestep convergence at the g=64 endpoint.
##
## Runs the production timestep and a factor-two refinement in separate trees.
## Both calculations include the centered-response linearity check.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly G="${G:-64}"
readonly NX="${NX:-64}"
readonly NPHI="${NPHI:-64}"
readonly TAU0="${TAU0:-0.1}"
readonly TAU_MAX="${TAU_MAX:-4.0}"
readonly BASE_DIR="${BASE_DIR:-results/squared_v1/extended_g/convergence/g${G}}"
readonly -a SPECTRA=(flat diff mcdiff)

production_dt="$($PYTHON_BIN - "$G" <<'PY'
import sys
print(f"{0.16 / float(sys.argv[1]):.12g}")
PY
)"
refined_dt="$($PYTHON_BIN - "$production_dt" <<'PY'
import sys
print(f"{0.5 * float(sys.argv[1]):.12g}")
PY
)"

for dt in "$production_dt" "$refined_dt"; do
    out_dir="${BASE_DIR}/dt${dt}"
    for spectrum in "${SPECTRA[@]}"; do
        echo "Endpoint validation: spectrum=${spectrum}, g=${G}, dt=${dt}"
        "$PYTHON_BIN" run_scan.py \
            --spectrum "$spectrum" \
            --g "$G" \
            --harmonics 2 3 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --tau0 "$TAU0" \
            --tau_max "$TAU_MAX" \
            --dt "$dt" \
            --linearity-check \
            --out "$out_dir"
    done
done

echo "Endpoint validation complete: ${BASE_DIR}"
