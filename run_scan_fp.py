"""!
@brief Production scan driver for the non-separable (leading-log FP)
benchmark: kappa_n(opacity) for the operator-valued kernel and the scalar
baselines on matched numerics.

@details
Designed like run_scan.py for Slurm job arrays: one (kernel, g) point per
invocation, one JSON per point.  Engine selection:

  * kernel == 'fp'          -> momentum-resolved SolverP (operator kernel);
  * scalar kernels          -> the p-integrated Solver (exact in p, fast);
                               gate V0p of tests/test_fp.py certifies that both
                               engines agree on identical grids.

For the FP kernel the one-hit accumulator stores the direct and
equilibrium-subtraction pieces separately (dilute-anchor attribution).
kappa extraction for the FP engine is single-sided (rotation covariance,
verified to 1e-13 in gate V4p); the scalar engine uses the original
centered difference.

Examples
--------
    python run_scan_fp.py --kernel fp   --g 2.0 --out results_fp/
    python run_scan_fp.py --kernel flat --g 2.0 --out results_fp/
    python run_scan_fp.py --kernel fp   --g 1.0 --one-hit --harmonics 2 3 4

Slurm array:
    GS=(0.5 1 2 4 8)
    python run_scan_fp.py --kernel $KERN --g ${GS[$SLURM_ARRAY_TASK_ID]} \
        --out results_fp/
"""

import argparse
import json
import os
import time

from mrta.solver import Params, kappa_n
from mrta.solver_p import ParamsP, kappa_n_p


def scan_point(kernel, g, harmonics, one_hit=False, alpha=1.0, delta2=0.04,
               Nx=80, Nphi=32, Np=20, pmax=12.0, pmin=0.0,
               bc_inner="noflux", L=9.0, tau0=0.1,
               tau_max=6.0, dt=0.03, bjorken=True, two_sided=False,
               verbose=True):
    """Compute kappa_n for all requested harmonics at one (kernel, g)."""
    deltas = {2: delta2, 3: delta2 / 4.0, 4: delta2 / 20.0}
    rec = {"kernel": kernel, "g": g, "alpha": alpha, "one_hit": one_hit,
           "Nx": Nx, "Nphi": Nphi, "Np": Np, "pmax": pmax, "pmin": pmin,
           "bc_inner": bc_inner, "L": L,
           "tau0": tau0, "tau_max": tau_max, "dt": dt, "bjorken": bjorken,
           "kappa": {}, "eps": {}, "kappa_direct": {}, "kappa_sub": {}}
    for n in harmonics:
        t0 = time.time()
        d = deltas.get(n, delta2 / 20.0)
        if kernel == "fp":
            par = ParamsP(Nx=Nx, Nphi=Nphi, Np=Np, pmax=pmax, pmin=pmin,
                          bc_inner=bc_inner, L=L,
                          tau0=tau0, tau_max=tau_max, dt=dt, g=g,
                          alpha=alpha, bjorken=bjorken, kernel="fp")
            kap, eps, r = kappa_n_p(par, n, d, one_hit=one_hit,
                                    two_sided=two_sided, store_every=10**9)
            if one_hit:
                rec["kappa_direct"][str(n)] = \
                    r["Vn_onehit_direct"][n - 1].real / eps
                rec["kappa_sub"][str(n)] = \
                    r["Vn_onehit_sub"][n - 1].real / eps
        else:
            par = Params(Nx=Nx, Nphi=Nphi, L=L, tau0=tau0, tau_max=tau_max,
                         dt=dt, g=g, alpha=alpha, bjorken=bjorken,
                         spectrum=kernel)
            kap, eps, _ = kappa_n(par, n, d, one_hit=one_hit,
                                  store_every=10**9)
        rec["kappa"][str(n)] = kap
        rec["eps"][str(n)] = eps
        if verbose:
            print(f"  [{kernel} g={g:g}{' 1hit' if one_hit else ''}] n={n}: "
                  f"kappa = {kap:+.6e} ({time.time()-t0:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="fp",
                    choices=["fp", "flat", "diff", "mcdiff"])
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--harmonics", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--delta", type=float, default=0.04,
                    help="beta_2 amplitude; n=3,4 scaled down as in the paper")
    ap.add_argument("--Nx", type=int, default=80)
    ap.add_argument("--Nphi", type=int, default=32)
    ap.add_argument("--Np", type=int, default=20)
    ap.add_argument("--pmax", type=float, default=12.0)
    ap.add_argument("--L", type=float, default=9.0)
    ap.add_argument("--tau0", type=float, default=0.1)
    ap.add_argument("--tau_max", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--no-bjorken", action="store_true")
    ap.add_argument("--one-hit", action="store_true")
    ap.add_argument("--two-sided", action="store_true",
                    help="centered difference also for the FP engine")
    ap.add_argument("--out", default="results_fp")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rec = scan_point(args.kernel, args.g, args.harmonics,
                     one_hit=args.one_hit, alpha=args.alpha,
                     delta2=args.delta, Nx=args.Nx, Nphi=args.Nphi,
                     Np=args.Np, pmax=args.pmax, L=args.L, tau0=args.tau0,
                     tau_max=args.tau_max, dt=args.dt,
                     bjorken=not args.no_bjorken, two_sided=args.two_sided)
    tag = f"{args.kernel}_g{args.g:g}" \
        + ("_1hit" if args.one_hit else "") \
        + (f"_a{args.alpha:g}" if args.alpha != 1.0 else "")
    path = os.path.join(args.out, tag + ".json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
