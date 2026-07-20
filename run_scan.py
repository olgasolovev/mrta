"""!
@brief Production scan driver: kappa_n(opacity) for a chosen kernel spectrum.

@details
Designed for Slurm job arrays: each invocation computes one (spectrum, g)
point for all requested harmonics and writes a JSON file, so the scan is
embarrassingly parallel and restartable.

Examples
--------
Single point:
    python run_scan.py --spectrum diff --g 1.0 --out results/

Slurm array (one g-value per task):
    GS=(0.03 0.06 0.12 0.25 0.5 1 2 4 8 16 32 64)
    python run_scan.py --spectrum $SPEC --g ${GS[$SLURM_ARRAY_TASK_ID]} --out results/

Collect into the money plot afterwards with your own plotting script:
kappa ratios (k3/k2, k4/k2) vs g, one curve per spectrum.
"""

import argparse
import json
import os
import time

from mrta.solver import Params, kappa_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", default="flat",
                    choices=["flat", "diff", "mcdiff", "mixed"])
    ap.add_argument("--b_over_a", type=float, default=1.0)
    ap.add_argument("--g", type=float, required=True, help="ghat_2 rate scale (opacity knob)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--harmonics", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--delta", type=float, default=0.04, help="perturbation amplitude beta_n")
    ap.add_argument("--Nx", type=int, default=192)
    ap.add_argument("--Nphi", type=int, default=128)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--tau0", type=float, default=0.1)
    ap.add_argument("--tau_max", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--no-bjorken", action="store_true")
    ap.add_argument("--one-hit", action="store_true",
                    help="accumulate the one-hit master formula instead of solving")
    ap.add_argument("--linearity-check", action="store_true",
                    help="repeat with delta/2 and report the deviation")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = Params(Nx=args.Nx, Nphi=args.Nphi, L=args.L, tau0=args.tau0,
                  tau_max=args.tau_max, dt=args.dt, g=args.g,
                  alpha=args.alpha, bjorken=not args.no_bjorken,
                  spectrum=args.spectrum, b_over_a=args.b_over_a)

    record = {"spectrum": args.spectrum, "b_over_a": args.b_over_a,
              "g": args.g, "alpha": args.alpha,
              "bjorken": not args.no_bjorken, "one_hit": args.one_hit,
              "Nx": args.Nx, "Nphi": args.Nphi, "L": args.L,
              "tau0": args.tau0, "tau_max": args.tau_max, "dt": args.dt,
              "delta": args.delta, "kappa": {}, "eps": {},
              "linearity_dev": {}}

    for n in args.harmonics:
        t0 = time.time()
        kap, eps, _ = kappa_n(base, n, args.delta, one_hit=args.one_hit,
                              store_every=10**9)
        record["kappa"][str(n)] = kap
        record["eps"][str(n)] = eps
        msg = f"  n={n}: kappa = {kap:+.6e}  (eps = {eps:+.4e}, {time.time()-t0:.0f}s)"
        if args.linearity_check:
            kap2, _, _ = kappa_n(base, n, args.delta / 2, one_hit=args.one_hit,
                                 store_every=10**9)
            dev = abs(kap2 / kap - 1.0)
            record["linearity_dev"][str(n)] = dev
            msg += f"  linearity dev = {dev:.2%}"
        print(msg, flush=True)

    tag = f"{args.spectrum}_g{args.g:g}" + ("_1hit" if args.one_hit else "")
    if args.spectrum == "mixed":
        tag += f"_ba{args.b_over_a:g}"
    path = os.path.join(args.out, tag + ".json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
