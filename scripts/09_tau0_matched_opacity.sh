#!/usr/bin/env bash
## @file 09_tau0_matched_opacity.sh
## @brief Scan tau0 at opacity matched through the flat one-hit kappa_2.
##
## The flat-spectrum one-hit response is linear in g.  At every tau0 this
## script first measures kappa_2 at a probe coupling, then rescales g so that
## kappa_2 matches the reference point (tau0=0.10, g=0.03).  It subsequently
## runs full and one-hit n=4 responses for all requested spectra.
##
## Override numerical settings through the environment, for example:
## @code{.sh}
## DT=0.0025 ./scripts/09_tau0_matched_opacity.sh
## @endcode

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly PROBE_G="${PROBE_G:-0.03}"
readonly REFERENCE_TAU0="${REFERENCE_TAU0:-0.10}"
readonly DT="${DT:-0.005}"
readonly NX="${NX:-64}"
readonly NPHI="${NPHI:-64}"
readonly TAU_MAX="${TAU_MAX:-4.0}"
readonly RUN_TAG="Nx${NX}_Nphi${NPHI}_dt${DT}"
readonly OUT_ROOT="${OUT_ROOT:-results/squared_v1/tau0/matched_opacity/${RUN_TAG}}"
readonly CALIBRATION_ROOT="${OUT_ROOT}/calibration"
readonly MAPPING_FILE="${OUT_ROOT}/matched_g.csv"
readonly -a TAU0_VALUES=(0.025 0.05 0.10 0.20)
readonly -a SPECTRA=(flat diff mcdiff)

mkdir -p "$CALIBRATION_ROOT"

## @brief Produce the one-hit flat-spectrum kappa_2 calibration records.
for tau0 in "${TAU0_VALUES[@]}"; do
    echo "Calibrating opacity: tau0=${tau0}, probe g=${PROBE_G}"
    python run_scan.py \
        --spectrum flat \
        --g "$PROBE_G" \
        --harmonics 2 \
        --Nx "$NX" \
        --Nphi "$NPHI" \
        --tau0 "$tau0" \
        --tau_max "$TAU_MAX" \
        --dt "$DT" \
        --one-hit \
        --linearity-check \
        --out "${CALIBRATION_ROOT}/tau0_${tau0}"
done

readonly REFERENCE_JSON="${CALIBRATION_ROOT}/tau0_${REFERENCE_TAU0}/flat_g${PROBE_G}_1hit.json"
if [[ ! -f "$REFERENCE_JSON" ]]; then
    echo "Missing reference calibration record: ${REFERENCE_JSON}" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
printf 'tau0,probe_g,matched_g,target_kappa2,probe_kappa2\n' > "$MAPPING_FILE"

## @brief Match g and run the n=4 response at each initialization time.
for tau0 in "${TAU0_VALUES[@]}"; do
    sample_json="${CALIBRATION_ROOT}/tau0_${tau0}/flat_g${PROBE_G}_1hit.json"
    if [[ ! -f "$sample_json" ]]; then
        echo "Missing calibration record: ${sample_json}" >&2
        exit 1
    fi

    calibration="$(${PYTHON:-python} - "$REFERENCE_JSON" "$sample_json" "$PROBE_G" <<'PY'
import json
import math
import sys

reference_path, sample_path, probe_text = sys.argv[1:]
with open(reference_path, encoding="utf-8") as stream:
    target = float(json.load(stream)["kappa"]["2"])
with open(sample_path, encoding="utf-8") as stream:
    sample = float(json.load(stream)["kappa"]["2"])

probe = float(probe_text)
if not math.isfinite(sample) or abs(sample) < 1.0e-30:
    raise SystemExit(f"Invalid probe kappa_2: {sample!r}")

matched = probe * target / sample
if not math.isfinite(matched) or matched <= 0.0:
    raise SystemExit(f"Invalid matched coupling: {matched!r}")

print(f"{matched:.16g},{target:.16e},{sample:.16e}")
PY
)"

    IFS=',' read -r matched_g target_kappa2 probe_kappa2 <<< "$calibration"
    printf '%s,%s,%s,%s,%s\n' \
        "$tau0" "$PROBE_G" "$matched_g" "$target_kappa2" "$probe_kappa2" \
        >> "$MAPPING_FILE"

    echo "Matched opacity: tau0=${tau0}, g=${matched_g}"
    tau_dir="${OUT_ROOT}/tau0_${tau0}"

    for spectrum in "${SPECTRA[@]}"; do
        echo "Matched full run: tau0=${tau0}, spectrum=${spectrum}, g=${matched_g}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$matched_g" \
            --harmonics 4 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --tau0 "$tau0" \
            --tau_max "$TAU_MAX" \
            --dt "$DT" \
            --linearity-check \
            --out "${tau_dir}/full"

        echo "Matched one-hit run: tau0=${tau0}, spectrum=${spectrum}, g=${matched_g}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$matched_g" \
            --harmonics 4 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --tau0 "$tau0" \
            --tau_max "$TAU_MAX" \
            --dt "$DT" \
            --one-hit \
            --linearity-check \
            --out "${tau_dir}/onehit"
    done
done

echo "Matched-opacity tau0 scan complete: ${OUT_ROOT}"
echo "Calibration mapping: ${MAPPING_FILE}"
