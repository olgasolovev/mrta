"""!
@brief Validation battery for the mode-resolved transport solver.

@details
Gates (see project plan):
  V1  pure advection vs exact free-streaming solution
  V2  homogeneous relaxation: harmonic decay rates e^{-ghat_l g t}
  V3  exact conservation of energy and momentum in a coupled run
  V4  dilute limit: full solver vs one-hit master formula, and the
      ratio theorem  (k3/k2)_spec / (k3/k2)_flat -> gamma_3/gamma_2

Run:  python tests_validation.py [--fast]
"""

import sys
import time
import numpy as np

from mrta.solver import Params, Solver, gamma_hat, kappa_n
from mrta.analytic import (gaussian_profile, free_stream_exact,
                           ratio_theorem)

FAST = "--fast" in sys.argv
PASS, FAIL = "\x1b[32mPASS\x1b[0m", "\x1b[31mFAIL\x1b[0m"
_failures = []


def check(name, ok, detail=""):
    print(f"[{PASS if ok else FAIL}] {name}   {detail}")
    if not ok:
        _failures.append(name)


# ----------------------------------------------------------------------------
## @brief V1 -- pure advection
# ----------------------------------------------------------------------------

def v1():
    p = Params(Nx=96 if FAST else 128, Nphi=32 if FAST else 64,
               g=0.0, tau0=0.0, tau_max=2.0, dt=0.05, beta=0.05, n_ecc=2)
    sol = Solver(p)
    t0 = time.time()
    ## @brief advect only (g = 0 makes collide a no-op through zero rate, but skip it
    # entirely to test the advection module in isolation)
    nsteps = int(round((p.tau_max - p.tau0) / p.dt))
    for _ in range(2 * nsteps):          #< 2 half steps per step
        sol.advect_half()
    exact = free_stream_exact(
        lambda XX, YY: gaussian_profile(XX, YY, R=p.R, Etot=p.Etot,
                                        n=p.n_ecc, beta=p.beta),
        sol.X, sol.Y, sol.phi, p.tau_max)
    err = np.max(np.abs(sol.Phi - exact)) / np.max(exact)
    check("V1 advection vs exact free streaming",
          err < 1e-8, f"max rel err = {err:.2e}  ({time.time()-t0:.1f}s)")


# ----------------------------------------------------------------------------
## @brief V2 -- homogeneous relaxation rates
# ----------------------------------------------------------------------------

def v2():
    for spec in ("flat", "diff", "mcdiff"):
        p = Params(Nx=16, Nphi=64, g=0.8, alpha=0.0, bjorken=False,
                   tau0=0.0, tau_max=1.0, dt=0.02, spectrum=spec)
        sol = Solver(p)
        ## @brief uniform anisotropic state: e_ref density with l = 2 and l = 3 content
        amp2, amp3 = 0.10, 0.08
        base = sol.e_ref / (2.0 * np.pi)
        sol.Phi = base * (1.0
                          + 2 * amp2 * np.cos(2 * sol.phi)
                          + 2 * amp3 * np.cos(3 * sol.phi))[None, None, :] \
            * np.ones((p.Nx, p.Nx, 1))
        r = sol.run(store_every=10**9)
        gh = gamma_hat(np.arange(4), spec)
        got2 = np.abs(r["Vn"][-1, 1]) / amp2
        got3 = np.abs(r["Vn"][-1, 2]) / amp3
        exp2 = np.exp(-gh[2] * p.g * p.tau_max)
        exp3 = np.exp(-gh[3] * p.g * p.tau_max)
        ok = abs(got2 / exp2 - 1) < 1e-10 and abs(got3 / exp3 - 1) < 1e-10
        check(f"V2 homogeneous decay [{spec}]", ok,
              f"l=2: {got2:.6e} vs {exp2:.6e} | l=3: {got3:.6e} vs {exp3:.6e}")


# ----------------------------------------------------------------------------
## @brief V3 -- conservation in a coupled run
# ----------------------------------------------------------------------------

def v3():
    p = Params(Nx=96 if FAST else 128, Nphi=32 if FAST else 64,
               g=2.0, tau_max=3.0, dt=0.02, beta=0.06, n_ecc=2,
               spectrum="diff")
    sol = Solver(p)
    r = sol.run(store_every=10**9)
    dE = abs(r["E"][-1] / r["E"][0] - 1.0)
    Pmax = max(abs(r["Px"]).max(), abs(r["Py"]).max()) / r["E"][0]
    ok = dE < 1e-12 and Pmax < 1e-12
    check("V3 exact conservation", ok,
          f"|dE/E| = {dE:.2e},  max|P|/E = {Pmax:.2e}")


# ----------------------------------------------------------------------------
## @brief V4 -- dilute limit: one-hit agreement and the ratio theorem
# ----------------------------------------------------------------------------

def v4():
    common = dict(Nx=96 if FAST else 128, Nphi=48 if FAST else 64,
                  tau0=0.1, tau_max=4.0 if FAST else 5.0, dt=0.025,
                  alpha=1.0, bjorken=True)
    ## @brief NB: the dilute gate must sit deep in the linear regime.  The odd-n
    # response at alpha = 1 is selection-rule suppressed at leading order
    # (Sec. 7 of the derivation note), so its relative finite-opacity
    # corrections are parametrically enhanced (~50 * g empirically); at
    # g = 2e-4 the full solver agrees with the one-hit master formula to <1%.
    g_dilute = 2e-4
    delta = 0.05
    kappas = {}
    for spec in ("flat", "diff"):
        for n in (2, 3):
            par = Params(g=g_dilute, spectrum=spec, **common)
            kap_full, eps, _ = kappa_n(par, n, delta, store_every=10**9)
            kap_1hit, _, _ = kappa_n(par, n, delta, one_hit=True,
                                     store_every=10**9)
            kappas[(spec, n, "full")] = kap_full
            kappas[(spec, n, "1hit")] = kap_1hit
            rel = abs(kap_full / kap_1hit - 1.0)
            check(f"V4a full vs one-hit [{spec}, n={n}]", rel < 0.05,
                  f"kappa_full = {kap_full:+.4e}, kappa_1hit = {kap_1hit:+.4e},"
                  f" rel dev = {rel:.1%}")

    for tag in ("full", "1hit"):
        dr = (kappas[("diff", 3, tag)] / kappas[("diff", 2, tag)]) \
            / (kappas[("flat", 3, tag)] / kappas[("flat", 2, tag)])
        target = ratio_theorem(3, 2, "diff")      #< 9/4
        tol = 0.08 if tag == "full" else 1e-10
        check(f"V4b ratio theorem [{tag}]", abs(dr / target - 1.0) < tol,
              f"double ratio = {dr:.4f}  (target gamma3/gamma2 = {target:.4f})")


if __name__ == "__main__":
    t0 = time.time()
    v1()
    v2()
    v3()
    v4()
    print(f"\n{len(_failures)} failure(s); total {time.time()-t0:.1f}s")
    sys.exit(1 if _failures else 0)
