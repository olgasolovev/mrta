"""!
@brief Infrared-robustness study for the FP benchmark.

For each configuration (pmin, Np, pmax, bc_inner) this driver computes:
  * gamma_{2,0} and X = eta_FP/eta_flat on a fine 1D grid with the SAME
    cutoff and inner BC (cheap, spectral columns of the table);
  * the closed-form predictions onehit_rate_thermal_cut / d32_direct_cut /
    visc_pred_cut (analytic columns);
  * full solver runs kappa_2(g=2), kappa_3(g=2), kappa_3(g=4) -- and, if the
    sign of kappa_3 does not flip between g=2 and g=4, adaptively kappa_3 at
    g=1 or g=8 to re-bracket the zero crossing g_3*;
  * D32(g=2) against the (p-grid independent) flat baseline.

Resumable: one JSON per configuration in results_fp/ir_<tag>.json.
The flat baseline kappa's are reused from results_fp/flat_g*.json (the
p-integrated engine has no p grid, so they are common to all rows).
"""

import json
import os
import time

import numpy as np

from run_scan_fp import scan_point
from mrta import fp_kernel as fp

OUT = "results_fp"

## (tag, pmin, Np, pmax, bc_inner)
CONFIGS = [
    ("base",   0.00, 20, 12.0, "noflux"),    # reuses fp_g2 / fp_g4
    ("a015",   0.15, 20, 12.0, "noflux"),
    ("a030",   0.30, 20, 12.0, "noflux"),
    ("a030d",  0.30, 20, 12.0, "dirichlet"),
    ("a060",   0.60, 19, 12.0, "noflux"),
    ("Np14",   0.00, 14, 12.0, "noflux"),
    ("pmax10", 0.00, 20, 10.0, "noflux"),
    ("hiNp",   0.00, 26, 14.0, "noflux"),    # reuses fp_g2_hiNp for g=2
]

GRID_G = [2.0, 4.0]          # primary bracket probes for kappa_3
FALLBACK = {"low": 1.0, "high": 8.0}


def jload(tag):
    path = os.path.join(OUT, tag + ".json")
    return json.load(open(path)) if os.path.exists(path) else None


def jsave(rec, tag):
    json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(f"== wrote {tag}.json", flush=True)


def reuse_kappa(tag, cfgtag, g, n):
    """Pull kappa_n(g) from pre-existing production JSONs where the grids
    coincide (base and hiNp), else None."""
    m = {("base", 2.0): "fp_g2", ("base", 4.0): "fp_g4",
         ("base", 1.0): "fp_g1", ("base", 8.0): "fp_g8",
         ("hiNp", 2.0): "fp_g2_hiNp"}
    src = m.get((cfgtag, g))
    if src is None:
        return None
    r = jload(src)
    if r is None:
        return None
    return r["kappa"].get(str(n))


def get_kappa(cfg, g, n, rec):
    tag, pmin, Np, pmax, bc = cfg
    key = f"k{n}_g{g:g}"
    if key in rec:
        return rec[key]
    v = reuse_kappa(tag, tag, g, n)
    if v is None:
        r = scan_point("fp", g, [n], Np=Np, pmax=pmax, pmin=pmin,
                       bc_inner=bc)   # default delta2 -> production amplitudes
        v = r["kappa"][str(n)]
    rec[key] = v
    return v


def bracket_crossing(cfg, rec):
    """Return (g_lo, g_hi) with sign change of kappa_3, extending the probe
    set adaptively if needed."""
    k2 = get_kappa(cfg, 2.0, 3, rec)
    k4 = get_kappa(cfg, 4.0, 3, rec)
    if k2 > 0 > k4:
        return 2.0, 4.0
    if k2 <= 0:                       # crossing moved below 2
        k1 = get_kappa(cfg, 1.0, 3, rec)
        return (1.0, 2.0) if k1 > 0 else (None, 2.0)
    # k4 >= 0: crossing moved above 4
    k8 = get_kappa(cfg, 8.0, 3, rec)
    return (4.0, 8.0) if k8 < 0 else (4.0, None)


def main():
    t00 = time.time()
    flat2 = jload("flat_g2")["kappa"]
    for cfg in CONFIGS:
        tag, pmin, Np, pmax, bc = cfg
        jtag = f"ir_{tag}"
        rec = jload(jtag) or {"tag": tag, "pmin": pmin, "Np": Np,
                              "pmax": pmax, "bc_inner": bc}
        if rec.get("done"):
            continue
        # cheap spectral / analytic columns
        if "X" not in rec:
            rec["gamma20"] = float(fp.spectrum_1d(
                2, Np=6000, pmax=400.0, pmin=pmin, bc_inner=bc, nev=1)[0])
            rec["X"] = float(fp.viscosity_enhancement(
                Np=8000, pmax=400.0, pmin=pmin, bc_inner=bc))
            rec["X_pred"] = float(fp.visc_pred_cut(pmin))
            rec["g2eff_pred"] = float(fp.onehit_rate_thermal_cut(2, pmin))
            rec["D32dir_pred"] = float(fp.d32_direct_cut(pmin))
            jsave(rec, jtag)
        # solver columns
        k2 = get_kappa(cfg, 2.0, 2, rec); jsave(rec, jtag)
        k3 = get_kappa(cfg, 2.0, 3, rec); jsave(rec, jtag)
        rec["D32_g2"] = (k3 / k2) / (flat2["3"] / flat2["2"])
        lo, hi = bracket_crossing(cfg, rec)
        rec["g3star_lo"], rec["g3star_hi"] = lo, hi
        rec["done"] = True
        jsave(rec, jtag)
        print(f"  -> {tag}: D32(2) = {rec['D32_g2']:+.4f}, "
              f"g3* in ({lo}, {hi}), X = {rec['X']:.4f}", flush=True)
    print(f"IR STUDY DONE in {(time.time()-t00)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
