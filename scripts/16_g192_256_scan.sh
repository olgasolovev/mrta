#!/usr/bin/env bash
## @file 16_g192_256_scan.sh
## @brief Run the final high-opacity D_32 points at g=192 and optionally 256.
##
## @details
## The default run computes g=192 with dt=0.000833333333.  Set
## INCLUDE_256=1 to additionally compute g=256 with dt=0.000625.  Both points
## use the full solver for harmonics n=2,3 and the flat, diff, and mcdiff
## spectra.  Results append to the existing extended-g production directory.
##
## Existing JSON records are skipped unless FORCE=1.  Linearity checks are
## enabled by default and may be disabled with LINEARITY_CHECK=0.
##
## @code{.sh}
## ./scripts/16_g192_256_scan.sh
## INCLUDE_256=1 ./scripts/16_g192_256_scan.sh
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
readonly INCLUDE_256="${INCLUDE_256:-0}"
readonly LINEARITY_CHECK="${LINEARITY_CHECK:-1}"
readonly FORCE="${FORCE:-0}"
readonly -a SPECTRA=(flat diff mcdiff)

G_VALUES=(192)
if [[ "$INCLUDE_256" == "1" ]]; then
    G_VALUES+=(256)
fi
readonly -a G_VALUES

LINEARITY_ARGS=()
if [[ "$LINEARITY_CHECK" == "1" ]]; then
    LINEARITY_ARGS+=(--linearity-check)
fi
readonly -a LINEARITY_ARGS

mkdir -p "$OUT_DIR"

## @brief Return the validated production timestep for a requested opacity.
## @param $1 Code opacity g; supported values are 192 and 256.
## @return Prints the corresponding timestep.
time_step_for_g() {
    case "$1" in
        192) printf '%s\n' '0.000833333333' ;;
        256) printf '%s\n' '0.000625' ;;
        *)
            echo "Unsupported opacity: $1" >&2
            return 1
            ;;
    esac
}

for g in "${G_VALUES[@]}"; do
    dt="$(time_step_for_g "$g")"

    for spectrum in "${SPECTRA[@]}"; do
        output="${OUT_DIR}/${spectrum}_g${g}.json"
        if [[ -f "$output" && "$FORCE" != "1" ]]; then
            echo "Skipping existing record: ${output}"
            continue
        fi

        echo "High-opacity scan: spectrum=${spectrum}, g=${g}, dt=${dt}"
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

echo "High-opacity scan complete: ${OUT_DIR}"
echo "Regenerate the completion plot with: ./bin/mrta-plot-extended-g"
