#!/usr/bin/env bash
## @file 08_tau0_fixed_g.sh
## @brief Measure the explicit initialization-time dependence at fixed code g.
##
## Runs both the full evolution and its one-hit reference for n=4 at each
## requested initialization time.  Override numerical settings through the
## environment, for example:
## @code{.sh}
## DT=0.0025 ./scripts/08_tau0_fixed_g.sh
## @endcode

set -euo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

readonly G="${G:-0.03}"
readonly DT="${DT:-0.005}"
readonly NX="${NX:-64}"
readonly NPHI="${NPHI:-64}"
readonly TAU_MAX="${TAU_MAX:-4.0}"
readonly RUN_TAG="Nx${NX}_Nphi${NPHI}_dt${DT}"
readonly OUT_ROOT="${OUT_ROOT:-results/squared_v1/tau0/fixed_g/${RUN_TAG}}"
readonly -a TAU0_VALUES=(0.025 0.05 0.10 0.20)
readonly -a SPECTRA=(flat diff mcdiff)

for tau0 in "${TAU0_VALUES[@]}"; do
    tau_dir="${OUT_ROOT}/tau0_${tau0}"

    for spectrum in "${SPECTRA[@]}"; do
        echo "Fixed-g full run: tau0=${tau0}, spectrum=${spectrum}, g=${G}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$G" \
            --harmonics 4 \
            --Nx "$NX" \
            --Nphi "$NPHI" \
            --tau0 "$tau0" \
            --tau_max "$TAU_MAX" \
            --dt "$DT" \
            --linearity-check \
            --out "${tau_dir}/full"

        echo "Fixed-g one-hit run: tau0=${tau0}, spectrum=${spectrum}, g=${G}"
        python run_scan.py \
            --spectrum "$spectrum" \
            --g "$G" \
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

echo "Fixed-g tau0 scan complete: ${OUT_ROOT}"
