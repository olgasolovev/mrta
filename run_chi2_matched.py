"""!
@brief Matched-opacity (chi_2) verification runs.

For each infrared configuration, run the solver at code opacity
g = chi_2 / u,  u = gamma_2^eff(pmin)/gamma_2^eff(0),
so that the physical opacity is held fixed at chi_2 = 2 and 4.  The
decisive test: kappa_3(chi_2 = 2) > 0 > kappa_3(chi_2 = 4) for every
configuration -- the zero crossing expressed directly in the matched
variable, chi_2^* in (2, 4), with no rescaling argument required.
Also records kappa_2(chi_2 = 2) for the D32(chi_2 = 2) collapse test.

Resumable: results_fp/chi2_<tag>.json.
"""

import json
import os
import time

from run_scan_fp import scan_point
from mrta.fp_kernel import onehit_rate_thermal_cut

OUT = "results_fp"

## (tag, pmin, Np, pmax, bc_inner)
CONFIGS = [
    ("a015",  0.15, 20, 12.0, "noflux"),
    ("a030",  0.30, 20, 12.0, "noflux"),
    ("a060",  0.60, 19, 12.0, "noflux"),
]


def jload(tag):
    p = os.path.join(OUT, tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def jsave(rec, tag):
    json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(f"== wrote {tag}.json", flush=True)


def main():
    t00 = time.time()
    flat2 = jload("flat_g2")["kappa"]      # flat baseline: chi_2 = g exactly
    for tag, pmin, Np, pmax, bc in CONFIGS:
        jtag = f"chi2_{tag}"
        rec = jload(jtag) or {"tag": tag, "pmin": pmin, "Np": Np,
                              "pmax": pmax, "bc_inner": bc}
        if rec.get("done"):
            continue
        u = onehit_rate_thermal_cut(2, pmin) / onehit_rate_thermal_cut(2, 0.0)
        rec["u"] = u
        for chi, harm in ((2.0, [2, 3]), (4.0, [3])):
            for n in harm:
                key = f"k{n}_chi{chi:g}"
                if key in rec:
                    continue
                r = scan_point("fp", chi / u, [n], Np=Np, pmax=pmax,
                               pmin=pmin, bc_inner=bc)
                rec[key] = float(r["kappa"][str(n)])
                jsave(rec, jtag)
        rec["D32_chi2"] = (rec["k3_chi2"] / rec["k2_chi2"]) \
            / (flat2["3"] / flat2["2"])
        rec["bracket_chi2"] = bool(rec["k3_chi2"] > 0.0 > rec["k3_chi4"])
        rec["done"] = True
        jsave(rec, jtag)
        print(f"  -> {tag}: u = {u:.4f}, D32(chi2=2) = {rec['D32_chi2']:+.4f},"
              f" k3(chi2=2) = {rec['k3_chi2']:+.3e},"
              f" k3(chi2=4) = {rec['k3_chi4']:+.3e},"
              f" chi2* in (2,4): {rec['bracket_chi2']}", flush=True)
    print(f"CHI2 STUDY DONE in {(time.time()-t00)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
