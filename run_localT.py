"""!
@file run_localT.py
@brief Reduced local-temperature check: frozen bath T = T0 vs conformal
T(x,tau), at g = 1, 2, 3, 4 (alpha = 1), n = 2, 3, plus the hardening
diagnostic.  Resumable."""
import json, os, time
import numpy as np
from run_scan_fp import scan_point
from mrta.solver_p import ParamsP, SolverP, kappa_n_p
from dataclasses import asdict

OUT = "results_fp"
def have(t): return os.path.exists(os.path.join(OUT, t + ".json"))

def point_local(tag, g, n):
    if have(tag):
        return
    par = ParamsP(g=g, local_T=True, Np=32)
    d = 0.04 if n == 2 else 0.01
    kap, eps, _ = kappa_n_p(par, n, d, store_every=10**9)
    json.dump({"kernel": "fp_localT", "g": g, "kappa": {str(n): float(kap)},
               "eps": {str(n): float(eps)}},
              open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(f"== wrote {tag}.json  kappa{n} = {kap:+.4e}", flush=True)

def main():
    t0 = time.time()
    # frozen g=3 completion
    if not have("fp_g3"):
        rec = scan_point("fp", 3.0, [2, 3])
        json.dump(rec, open(os.path.join(OUT, "fp_g3.json"), "w"), indent=1)
        print("== wrote fp_g3.json", flush=True)
    for g in (1.0, 2.0, 3.0, 4.0):
        point_local(f"lT_fp_n3_g{g:g}", g, 3)
    for g in (1.0, 2.0, 3.0, 4.0):
        point_local(f"lT_fp_n2_g{g:g}", g, 2)
    if not os.path.exists(os.path.join(OUT, "hardening_lT_g2_n2.npz")):
        par = ParamsP(g=2.0, beta=0.04, n_ecc=2, local_T=True, Np=32)
        r = SolverP(par).run(store_every=10, track_p=2)
        np.savez(os.path.join(OUT, "hardening_lT_g2_n2.npz"),
                 tau=r["tau"], mean_p=r["mean_p"])
        print("== wrote hardening_lT_g2_n2.npz", flush=True)

    # floor sensitivity (g=2, n=3): T-floor 0.5 vs default 0.6
    if not have("lT_fp_n3_g2_floor05"):
        par = ParamsP(g=2.0, local_T=True, Np=32, local_T_floor=0.5)
        kap, eps, _ = kappa_n_p(par, 3, 0.01, store_every=10**9)
        json.dump({"kappa": {"3": float(kap)}},
                  open(os.path.join(OUT, "lT_fp_n3_g2_floor05.json"), "w"))
        print(f"== wrote lT_fp_n3_g2_floor05.json  kappa3 = {kap:+.4e}",
              flush=True)
    # combined local-T + conformal exponent alpha = 1/3 (mini-scan)
    for g in (0.2, 0.5, 2.0):
        tag = f"lT_a13_fp_n3_g{g:g}"
        if have(tag):
            continue
        par = ParamsP(g=g, local_T=True, Np=32, alpha=1.0/3.0)
        kap, eps, _ = kappa_n_p(par, 3, 0.01, store_every=10**9)
        json.dump({"kappa": {"3": float(kap)}},
                  open(os.path.join(OUT, tag + ".json"), "w"))
        print(f"== wrote {tag}.json  kappa3 = {kap:+.4e}", flush=True)
    print(f"LOCAL-T CHECK DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
