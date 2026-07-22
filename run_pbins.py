"""!
@file run_pbins.py
@brief Momentum-differential response: kappa_n in p-bins [0,2), [2,4), [4,8) T0
for the FP kernel and the flat baseline (same momentum-resolved pipeline),
at g = 0.5, 2, 8.  Resumable."""
import json, os, time
from dataclasses import asdict
import numpy as np
from mrta.solver_p import ParamsP, SolverP
from mrta.solver import eccentricity

OUT = "results_fp"
EDGES = [0.0, 2.0, 4.0, 8.0]
def have(t): return os.path.exists(os.path.join(OUT, t + ".json"))

def binned_kappa(kernel, g, n):
    d = 0.04 if n == 2 else 0.01
    par = ParamsP(g=g, kernel=kernel, beta=d, n_ecc=n)
    sol = SolverP(par)
    eps = eccentricity(sol.E0, sol.X, sol.Y, n, sol.dA).real
    sol.run(store_every=10**9)
    Vb = sol.Vn_binned(EDGES)
    return [float(Vb[b, n - 1].real / eps) for b in range(3)]

def main():
    t0 = time.time()
    for g in (2.0, 8.0, 0.5):
        for kern in ("fp", "flat"):
            for n in (3, 2):
                tag = f"pb_{kern}_n{n}_g{g:g}"
                if have(tag):
                    continue
                kb = binned_kappa(kern, g, n)
                json.dump({"kernel": kern, "g": g, "n": n, "edges": EDGES,
                           "kappa_bins": kb},
                          open(os.path.join(OUT, tag + ".json"), "w"))
                print(f"== {tag}: " + " ".join(f"{v:+.3e}" for v in kb),
                      flush=True)
    print(f"PBIN SCAN DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
