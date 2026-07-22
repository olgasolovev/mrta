"""!
@brief Conformal-rate (alpha = 1/3) scan: r(x,tau) propto e^{1/3}.

At alpha = 1 the odd-harmonic direct one-hit term vanishes (Prop. 3), so
kappa_3 is maximally exposed to the Landau subtraction; this scan tests
whether the FP kernel's triangular suppression / zero crossing survives
when the direct channel is restored at the conformal exponent.

Order of execution (resumable, one JSON per point):
  1. scalar baselines flat & diff, n = 2, 3, all g (fast p-integrated engine);
  2. FP kappa_3 at g = 0.5, 1, 2, 2.5, 3, 3.5, 4, 6, 8  (the essential
     signed scan requested);
  3. FP kappa_2 at g = 0.5, 1, 2, 3, 4, 6, 8 (for D32).
"""

import json
import os
import time

from run_scan_fp import scan_point

OUT = "results_fp"
ALPHA = 1.0 / 3.0
G3 = [0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0]
G3X = [0.1, 0.2, 0.35]   # bracket the early crossing
G2 = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def have(tag):
    return os.path.exists(os.path.join(OUT, tag + ".json"))


def point(tag, kernel, g, harmonics):
    if have(tag):
        return
    rec = scan_point(kernel, g, harmonics, alpha=ALPHA)
    json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(f"== wrote {tag}.json", flush=True)


def main():
    t0 = time.time()
    for kern in ("flat", "diff"):
        for g in G3:
            point(f"a13_{kern}_g{g:g}", kern, g, [2, 3])
    for g in G3X + G3:
        point(f"a13_fp_n3_g{g:g}", "fp", g, [3])
    for g in G2:
        point(f"a13_fp_n2_g{g:g}", "fp", g, [2])
    print(f"ALPHA13 SCAN DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
