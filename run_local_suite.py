"""!
@file run_local_suite.py
@brief Local production suite for the FP benchmark (resumable: existing JSONs
are skipped, so the script can be re-invoked until it prints SUITE DONE)."""

import json
import os
import time

import numpy as np

from run_scan_fp import scan_point

OUT = "results_fp"
os.makedirs(OUT, exist_ok=True)
GS = [0.5, 1.0, 2.0, 4.0, 8.0]


def have(tag):
    return os.path.exists(os.path.join(OUT, tag + ".json"))


def save(rec, tag):
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"== wrote {tag}.json", flush=True)


def point(tag, *a, **kw):
    if have(tag):
        return
    save(scan_point(*a, **kw), tag)


t00 = time.time()

# --- 1) scalar baselines
for kern in ("flat", "diff"):
    point(f"{kern}_g1_1hit", kern, 1.0, [2, 3, 4], one_hit=True)
    for g in GS:
        point(f"{kern}_g{g:g}", kern, g, [2, 3])

# --- 2) FP one-hit anchors
point("fp_g1_1hit", "fp", 1.0, [2, 3, 4], one_hit=True)

# --- 3) FP opacity scan
for g in GS:
    point(f"fp_g{g:g}", "fp", g, [2, 3])

# --- 4) alpha = 1/3 one-hit anchors
for kern in ("flat", "fp"):
    point(f"{kern}_g1_1hit_a0.333", kern, 1.0, [2, 3], one_hit=True,
          alpha=1.0 / 3.0)

# --- 4b) flat one-hit through the momentum-resolved pipeline with the
#         direct/sub split (for the attribution table)
if not have("flatP_g1_1hit"):
    from mrta.solver_p import ParamsP, kappa_n_p
    rec = {"kernel": "flatP", "kappa": {}, "eps": {},
           "kappa_direct": {}, "kappa_sub": {}}
    for n, d in ((2, 0.04), (3, 0.01)):
        par = ParamsP(g=1.0, kernel="flat")
        kap, eps, r = kappa_n_p(par, n, d, one_hit=True, store_every=10**9)
        rec["kappa"][str(n)] = kap
        rec["eps"][str(n)] = eps
        rec["kappa_direct"][str(n)] = r["Vn_onehit_direct"][n - 1].real / eps
        rec["kappa_sub"][str(n)] = r["Vn_onehit_sub"][n - 1].real / eps
        print(f"  [flatP 1hit] n={n}: {kap:+.6e}", flush=True)
    save(rec, "flatP_g1_1hit")

# --- 5) hardening diagnostic
from mrta.solver_p import ParamsP, SolverP  # noqa: E402

if not os.path.exists(os.path.join(OUT, "hardening_g2_n2.npz")):
    par = ParamsP(g=2.0, beta=0.04, n_ecc=2)
    r = SolverP(par).run(store_every=10, track_p=2)
    np.savez(os.path.join(OUT, "hardening_g2_n2.npz"),
             tau=r["tau"], mean_p=r["mean_p"], Vn=r["Vn"])
    print("== wrote hardening_g2_n2.npz", flush=True)

if not os.path.exists(os.path.join(OUT, "hardening_g0_n2.npz")):
    par = ParamsP(g=0.0, beta=0.04, n_ecc=2)
    r = SolverP(par).run(store_every=10, track_p=2)
    np.savez(os.path.join(OUT, "hardening_g0_n2.npz"),
             tau=r["tau"], mean_p=r["mean_p"])
    print("== wrote hardening_g0_n2.npz", flush=True)

# --- 6) resolution checks
point("fp_g2_hiNp", "fp", 2.0, [2, 3], Np=26, pmax=14.0)
point("fp_g2_hiNx", "fp", 2.0, [2, 3], Nx=96)

print(f"SUITE DONE in {(time.time()-t00)/60:.1f} min", flush=True)
