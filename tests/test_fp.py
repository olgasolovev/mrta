"""!
@brief Validation gates for the non-separable (leading-log FP) extension.

Gates
-----
V0p  scalar-kernel equivalence: SolverP('flat') reproduces the p-integrated
     Solver('flat') observables on the same grid (the momentum-resolved
     code contains the MRTA solver as an exact special case).
V1p  operator eigenmodes: a spatially uniform state seeded with a single
     radial eigenmode of Lambda_2 decays at exactly exp(-g gamma_{2,k} t)
     (machine test of the collision substep), and the fine-grid 1D spectrum
     reproduces the closed-form Rydberg tower
     gamma_{l,k} = (kappa/4T^2)[1 - (2l+2k+1)^{-2}].
V2p  exact conservation of E, P in a coupled eccentric run.
V3p  one-hit master formula: full solver -> one-hit accumulator as g -> 0.
V4p  dilute anchors: the DIRECT one-hit double ratio equals the discrete
     thermal-shape prediction gamma_3^eff/gamma_2^eff (continuum: 9/4),
     and V_n(-beta) = -V_n(+beta) (rotation covariance).

Run:  pytest tests/test_fp.py [--fast]
      python tests/test_fp.py [--fast]
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import numpy as np

from mrta.solver import Params, Solver
from mrta.solver_p import ParamsP, SolverP, kappa_n_p
from mrta import fp_kernel as fp


def banner(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def gate_V0p(fast):
    """SolverP with scalar 'flat' kernel vs p-integrated Solver."""
    kw = dict(Nx=64, Nphi=32, L=8.0, tau0=0.1, tau_max=2.0, dt=0.04,
              g=1.5, alpha=1.0, bjorken=True, beta=0.05, n_ecc=2)
    s1 = Solver(Params(**kw))
    r1 = s1.run(store_every=10**9)
    s2 = SolverP(ParamsP(**kw, kernel="flat", Np=16, pmax=12.0))
    r2 = s2.run(store_every=10**9)
    d = np.abs(r1["Vn"][-1] - r2["Vn"][-1]).max()
    return banner("V0p scalar-kernel equivalence",
                  d < 5e-4, f"max |dV_n| = {d:.2e}")


def gate_V1p(fast):
    """Eigenmode decay + Rydberg tower."""
    # (a) fine-grid 1D spectrum vs closed form
    ok = True
    msg = []
    for l in (2, 3):
        num = fp.spectrum_1d(l, Np=4000, pmax=400.0, nev=2)
        for k in (0, 1):
            ex = fp.rydberg_gamma(l, k)
            rel = abs(num[k] / ex - 1.0)
            ok &= rel < 5e-4
            msg.append(f"g{l}{k}:{rel:.1e}")
    # (b) machine test of the collision substep on a uniform state
    p = ParamsP(Nx=16, Nphi=32, Np=20, pmax=12.0, L=8.0, g=0.8, alpha=0.0,
                bjorken=False, tau0=0.0, tau_max=1.0, dt=0.05)
    sol = SolverP(p)
    k_mode = 3                       # an interior radial eigenmode of Lambda_2
    psi = sol.K.R[2][:, k_mode]
    gam = sol.K.Gam[2][k_mode]
    amp0 = 1e-3
    base = sol.e_ref / (2.0 * np.pi) * sol.h0
    sol.F[:] = base[None, None, :, None] \
        + amp0 * psi[None, None, :, None] * (2.0 * np.cos(2.0 * sol.phi))[None, None, None, :]
    r = sol.run(store_every=10**9)
    v2 = r["Vn"][-1, 1].real
    v2_exact = (amp0 * psi.sum() * sol.dp * 2.0 * np.pi / sol.e_ref) \
        * np.exp(-p.g * gam * (p.tau_max - p.tau0))
    rel = abs(v2 / v2_exact - 1.0)
    ok &= rel < 1e-10
    msg.append(f"substep:{rel:.1e}")
    return banner("V1p eigenmodes (Rydberg + substep)", ok, " ".join(msg))


def gate_V2p(fast):
    """Exact conservation in a coupled eccentric run."""
    p = ParamsP(Nx=64, Nphi=32, Np=20, L=8.0, g=4.0, beta=0.06,
                tau_max=2.0 if fast else 4.0)
    r = SolverP(p).run(store_every=10**9)
    dE = abs(r["E"][-1] / r["E"][0] - 1.0)
    P = max(abs(r["Px"][-1]), abs(r["Py"][-1]))
    return banner("V2p conservation", dE < 1e-12 and P < 1e-12,
                  f"dE/E = {dE:.1e}, |P| = {P:.1e}")


def gate_V3p(fast):
    """Full solver -> one-hit accumulator as g -> 0."""
    base = ParamsP(Nx=64, Nphi=32, Np=20, L=9.0, g=0.02, beta=0.04,
                   tau_max=4.0 if fast else 6.0)
    k_full, eps, _ = kappa_n_p(base, 2, 0.04, one_hit=False,
                               store_every=10**9)
    k_1hit, _, _ = kappa_n_p(base, 2, 0.04, one_hit=True,
                             store_every=10**9)
    rel = abs(k_full / (base.g * k_1hit / base.g) / 1.0 - k_1hit / k_1hit)  # noqa
    rel = abs(k_full / k_1hit - 1.0)
    return banner("V3p one-hit master formula", rel < 0.02,
                  f"|full/one-hit - 1| = {rel:.2%} at g = {base.g}")


def gate_V4p(fast):
    """Dilute anchors: direct-term double ratio + rotation covariance."""
    base = ParamsP(Nx=64 if fast else 80, Nphi=32, Np=20, L=9.0, g=1.0,
                   beta=0.04, tau_max=4.0 if fast else 6.0)
    res = {}
    for n in (2, 3):
        _, eps, r = kappa_n_p(base, n, 0.04 if n == 2 else 0.01,
                              one_hit=True, store_every=10**9)
        res[n] = dict(direct=r["Vn_onehit_direct"][n - 1].real / eps,
                      total=r["Vn_onehit"][n - 1].real / eps)
    # flat baseline on the same grid, same code path
    resf = {}
    for n in (2, 3):
        pp = ParamsP(**{**base.__dict__, "kernel": "flat"})
        _, eps, r = kappa_n_p(pp, n, 0.04 if n == 2 else 0.01,
                              one_hit=True, store_every=10**9)
        resf[n] = dict(direct=r["Vn_onehit_direct"][n - 1].real / eps,
                       total=r["Vn_onehit"][n - 1].real / eps)
    D32_dir = (res[3]["direct"] / res[2]["direct"]) \
        / (resf[3]["direct"] / resf[2]["direct"])
    D32_tot = (res[3]["total"] / res[2]["total"]) \
        / (resf[3]["total"] / resf[2]["total"])
    # discrete thermal-shape prediction for the direct term
    K = fp.factorize(3, base.Np, base.pmax)
    h0 = K.p**2 * np.exp(-K.p)
    g2 = ((K.R[2] * K.Gam[2][None, :]) @ (K.L[2] @ h0)).sum() / h0.sum()
    g3 = ((K.R[3] * K.Gam[3][None, :]) @ (K.L[3] @ h0)).sum() / h0.sum()
    pred = g3 / g2
    rel = abs(D32_dir / pred - 1.0)
    # rotation covariance
    kap_p, eps_p, _ = kappa_n_p(ParamsP(**{**base.__dict__, "tau_max": 1.0}),
                                3, 0.01, one_hit=True, store_every=10**9)
    kap_pm, _, _ = kappa_n_p(ParamsP(**{**base.__dict__, "tau_max": 1.0}),
                             3, 0.01, one_hit=True, two_sided=True,
                             store_every=10**9)
    sym = abs(kap_pm / kap_p - 1.0)
    ok = rel < 5e-3 and sym < 1e-8
    return banner("V4p dilute anchors", ok,
                  f"D32(direct) = {D32_dir:.4f} vs pred {pred:.4f} "
                  f"(rel {rel:.1e}); D32(total,1hit) = {D32_tot:.4f}; "
                  f"+-beta sym = {sym:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    gates = [gate_V1p, gate_V0p, gate_V2p, gate_V3p, gate_V4p]
    ok = all([g(a.fast) for g in gates])
    print("ALL GATES PASS" if ok else "SOME GATES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# pytest wrappers (also runnable as ``python tests/test_fp.py [--fast]``)
# ---------------------------------------------------------------------------

def _fast(request=None):
    """!
    @brief Read the --fast flag from a pytest request or fall back to False.
    """
    if request is None:
        return False
    try:
        return request.config.getoption("--fast")
    except Exception:
        return False


def test_V0p_scalar_kernel_equivalence(request):
    """!@brief SolverP('flat') reproduces the p-integrated Solver('flat')."""
    assert gate_V0p(_fast(request))


def test_V1p_eigenmodes(request):
    """!@brief Rydberg tower + machine-precision collision-substep decay."""
    assert gate_V1p(_fast(request))


def test_V2p_conservation(request):
    """!@brief Exact E and P conservation in coupled eccentric runs."""
    assert gate_V2p(_fast(request))


def test_V3p_one_hit_master_formula(request):
    """!@brief Full solver reproduces the one-hit accumulator as g -> 0."""
    assert gate_V3p(_fast(request))


def test_V4p_dilute_anchors(request):
    """!@brief Dilute anchors and rotation covariance."""
    assert gate_V4p(_fast(request))
