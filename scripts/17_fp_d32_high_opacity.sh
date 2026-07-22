#!/usr/bin/env bash
## @file 17_fp_d32_high_opacity.sh
## @brief Extend the non-separable FP D32 scan to higher opacities.
## @details
## Runs the FP (red-curve) response and its matched flat-kernel denominator
## for harmonics n = 2 and n = 3.  The time step is bounded by
## Delta tau <= 0.12/g and adjusted so that the evolution ends exactly at
## tau_max.  Existing non-empty JSON records are skipped, making the scan
## safe to resume after interruption.
##
## Environment variables:
##   G_VALUES            Space-separated opacity grid.
##                       Default: "12 16 24 32".
##   INCLUDE_48_64       Set to 1 to append g = 48 and 64.
##   VALIDATE_ENDPOINT   Set to 1 for a half-dt convergence rerun at the
##                       largest requested opacity.
##   PYTHON_BIN          Python interpreter. Default: python.
##   OUT_DIR             Production output directory. Default: results_fp.

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly OUT_DIR="${OUT_DIR:-results_fp}"
readonly TAU0="0.1"
readonly TAU_MAX="6.0"
readonly DT_COEFFICIENT="0.12"
readonly DT_MAX="0.03"

g_values="${G_VALUES:-12 16 24 32}"
if [[ "${INCLUDE_48_64:-0}" == "1" ]]; then
    g_values+=" 48 64"
fi
read -r -a OPACITIES <<< "$g_values"
readonly -a OPACITIES
readonly -a KERNELS=(flat fp)

mkdir -p "$OUT_DIR"

## @brief Calculate a step size that respects the high-opacity bound and
## reaches tau_max exactly after an integer number of steps.
## @param $1 Opacity parameter g.
## @return Time step printed to standard output.
time_step_for_g() {
    local g="$1"
    awk -v g="$g" -v tau0="$TAU0" -v taumax="$TAU_MAX" \
        -v coefficient="$DT_COEFFICIENT" -v dtmax="$DT_MAX" '
        BEGIN {
            span = taumax - tau0
            cap = coefficient / g
            if (cap > dtmax) cap = dtmax
            nsteps = int(span / cap)
            if (nsteps * cap < span) nsteps++
            printf "%.12g", span / nsteps
        }'
}

## @brief Run one matched kernel/opacity record unless it already exists.
## @param $1 Kernel name: flat or fp.
## @param $2 Opacity parameter g.
## @param $3 Time step.
## @param $4 Output directory.
run_point() {
    local kernel="$1"
    local g="$2"
    local dt="$3"
    local out_dir="$4"
    local output_file="${out_dir}/${kernel}_g${g}.json"

    if [[ -s "$output_file" ]]; then
        echo "Skipping existing record: $output_file"
        return
    fi

    echo "Running D32 point: kernel=${kernel}, g=${g}, dt=${dt}"
    "$PYTHON_BIN" run_scan_fp.py \
        --kernel "$kernel" \
        --g "$g" \
        --harmonics 2 3 \
        --Nx 80 \
        --Nphi 32 \
        --Np 20 \
        --pmax 12.0 \
        --L 9.0 \
        --tau0 "$TAU0" \
        --tau_max "$TAU_MAX" \
        --dt "$dt" \
        --out "$out_dir"
}

for g in "${OPACITIES[@]}"; do
    dt="$(time_step_for_g "$g")"
    for kernel in "${KERNELS[@]}"; do
        run_point "$kernel" "$g" "$dt" "$OUT_DIR"
    done
done

if [[ "${VALIDATE_ENDPOINT:-0}" == "1" ]]; then
    readonly endpoint="${OPACITIES[${#OPACITIES[@]} - 1]}"
    production_dt="$(time_step_for_g "$endpoint")"
    half_dt="$(awk -v dt="$production_dt" 'BEGIN { printf "%.12g", dt / 2.0 }')"
    readonly validation_dir="${OUT_DIR}/convergence/g${endpoint}/dt_half"
    mkdir -p "$validation_dir"

    echo "Running endpoint half-dt validation: g=${endpoint}, dt=${half_dt}"
    for kernel in "${KERNELS[@]}"; do
        run_point "$kernel" "$endpoint" "$half_dt" "$validation_dir"
    done
fi

echo "High-opacity FP D32 scan complete."
echo "Production records: $OUT_DIR/{flat,fp}_g*.json"
