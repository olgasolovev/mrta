#!/usr/bin/env bash
## @file 15_asymptotic_g_scan.sh
## @brief Extend the MRTA D_32 scan to g=96 and 128, optionally g=192.
##
## @details
## Computes the full-solver n=2 and n=3 responses for the flat, diff, and
## mcdiff collision spectra.  The timestep follows dt=0.16/g, preserving the
## production resolution g*dt=0.16 used from g=8 through g=64.
##
## Existing JSON records are skipped by default so interrupted runs can be
## resumed.  Set FORCE=1 to recompute them.  Set INCLUDE_192=1 to add g=192.
## Linearity checks are enabled by default; set LINEARITY_CHECK=0 for a faster
## exploratory scan.
##
## @code{.sh}
## ./scripts/15_asymptotic_g_scan.sh
## INCLUDE_192=1 ./scripts/15_asymptotic_g_scan.sh
## LINEARITY_CHECK=0 ./scripts/15_asymptotic_g_scan.sh
## @endcode

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly NX="${NX:-64}"
readonly NPHI="${NPHI:-64}"
readonly L="${L:-10.0}"
readonly TAU0="${TAU0:-0.1}"
readonly TAU_MAX="${TAU_MAX:-4.0}"
readonly OUT_DIR="${OUT_DIR:-results/squared_v1/extended_g/full}"
readonly FORCE="${FORCE:-0}"
readonly INCLUDE_192="${INCLUDE_192:-0}"
readonly LINEARITY_CHECK="${LINEARITY_CHECK:-1}"
readonly -a SPECTRA=(flat diff mcdiff)

G_VALUES=(96 128)
if [[ "$INCLUDE_192" == "1" ]]; then
    G_VALUES+=(192)
fi
readonly -a G_VALUES

LINEARITY_ARGS=()
if [[ "$LINEARITY_CHECK" == "1" ]]; then
    LINEARITY_ARGS+=(--linearity-check)
fi
readonly -a LINEARITY_ARGS

mkdir -p "$OUT_DIR"

## @brief Calculate the timestep that maintains g*dt=0.16.
## @param $1 Code opacity parameter g.
## @return Prints the corresponding timestep.
time_step_for_g() {
    "$PYTHON_BIN" - "$1" <<'PY'
import math
import sys

g = float(sys.argv[1])
if not math.isfinite(g) or g <= 0.0:
    raise SystemExit(f"g must be positive and finite, received {g!r}")
print(f"{0.16 / g:.12g}")
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

        echo "Asymptotic scan: spectrum=${spectrum}, g=${g}, dt=${dt}"
        "$PYTHON_BIN" run_scan.py \
            --spectrum "$spectrum" \
            --g "$g" \
            --harmonics 2 3 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --L "$L" \
            --tau0 "$TAU0" \
            --tau_max "$TAU_MAX" \
            --dt "$dt" \
            "${LINEARITY_ARGS[@]}" \
            --out "$OUT_DIR"
    done
done

echo "Asymptotic opacity scan complete: ${OUT_DIR}"
echo "Regenerate D_32 with: ./bin/mrta-plot-extended-g"
