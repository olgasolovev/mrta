#!/usr/bin/env bash
## @file 12_extended_g_scan.sh
## @brief Extend the full MRTA n=2,3 response scan through g=64.
##
## The timestep keeps g*dt=0.16, matching the established production point
## (g,dt)=(8,0.02).  Existing JSON records are skipped unless FORCE=1.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly NX="${NX:-64}"
readonly NPHI="${NPHI:-64}"
readonly TAU0="${TAU0:-0.1}"
readonly TAU_MAX="${TAU_MAX:-4.0}"
readonly OUT_DIR="${OUT_DIR:-results/squared_v1/extended_g/full}"
readonly FORCE="${FORCE:-0}"
readonly -a SPECTRA=(flat diff mcdiff)
readonly -a G_VALUES=(16 24 32 48 64)

mkdir -p "$OUT_DIR"

## @brief Return the production timestep associated with an opacity.
## @param $1 Opacity parameter g.
## @return Prints dt=0.16/g.
time_step_for_g() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
print(f"{0.16 / float(sys.argv[1]):.12g}")
PY
}

for g in "${G_VALUES[@]}"; do
    dt="$(time_step_for_g "$g")"
    for spectrum in "${SPECTRA[@]}"; do
        output="${OUT_DIR}/${spectrum}_g${g}.json"
        if [[ -f "$output" && "$FORCE" != "1" ]]; then
            echo "Skipping existing record: ${output}"
            continue
        fi

        echo "Extended scan: spectrum=${spectrum}, g=${g}, dt=${dt}"
        "$PYTHON_BIN" run_scan.py \
            --spectrum "$spectrum" \
            --g "$g" \
            --harmonics 2 3 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --tau0 "$TAU0" \
            --tau_max "$TAU_MAX" \
            --dt "$dt" \
            --out "$OUT_DIR"
    done
done

echo "Extended opacity scan complete: ${OUT_DIR}"
