"""!
@file run_gainproj.py
@brief Conservation-projection check: current prescription (l = 0, 1 frozen)
vs C_proj = -(1-P)Lambda(1-P) with active projected l = 0, +-1 radial
dynamics, at g = 1, 2, 3, 4, n = 2, 3.  Resumable."""
import json, os, time
from mrta.solver_p import ParamsP, kappa_n_p

OUT = "results_fp"
def have(t): return os.path.exists(os.path.join(OUT, t + ".json"))

def main():
    t0 = time.time()
    for g in (2.0, 3.0, 1.0, 4.0):        # crossing-critical first
        for n in (3, 2):
            tag = f"gp_fp_n{n}_g{g:g}"
            if have(tag):
                continue
            par = ParamsP(g=g, gain_proj=True)
            d = 0.04 if n == 2 else 0.01
            kap, eps, _ = kappa_n_p(par, n, d, store_every=10**9)
            json.dump({"kappa": {str(n): float(kap)}},
                      open(os.path.join(OUT, tag + ".json"), "w"))
            print(f"== wrote {tag}.json  kappa{n} = {kap:+.4e}", flush=True)
    print(f"GAIN-PROJ CHECK DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
