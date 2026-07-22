"""!
@file fp_kernel.py
@brief mrta.fp_kernel -- linearized leading-log QCD (Fokker-Planck) collision
kernel: the non-separable microscopic benchmark for MRTA.

@details
Model
-----
2D massless partons, momentum-resolved distribution f(x, phi, p, tau).
The leading-logarithmic limit of the linearized AMY collision operator is
drag + diffusion in momentum space (Landau / Fokker-Planck form),

    C_FP[f] = kappa  grad_p . [ T grad_p f  +  phat f ],

with bath temperature T (frozen to the reference T0, see the note) and
diffusion constant kappa (the 2D stand-in for qhat/2).  In polar momentum
coordinates the operator is diagonal in angular harmonics l but acts as a
RADIAL DIFFERENTIAL OPERATOR per harmonic:

    (C_FP)_l  =  kappa [ d^2/dp^2 + (1/p + 1/T) d/dp + 1/(pT) - l^2/p^2 ] f_l
              == -Lambda_l f_l .

The angular part carries the momentum-dependent rate

    gamma_l(p) = kappa l^2 / p^2                                    (*)

which entangles the harmonic index with |p|: this kernel is manifestly
OUTSIDE the MRTA ansatz gamma_l(x,tau) = ghat_l W(x,tau).

Exact spectrum (Coulomb map)
----------------------------
The route is the classical unitary equivalence between Fokker-Planck
operators and Schrodinger Hamiltonians (Risken, The Fokker-Planck
Equation, 1989), recently deployed in relativistic kinetic theory by
L. Gavassino [PRD 114, 014018 (2026), arXiv:2601.19474; PRL 137, 022302
(2026), arXiv:2601.19464], who obtained hydrogenic quasinormal spectra
for the hydrodynamic sector.  What is specific to this module is the
l-RESOLVED application to the momentum-conservation-constrained angular
blocks (l >= 2) of the transverse-plane kernel -- the anisotropy sector
probed by flow observables.
Lambda_l is self-adjoint w.r.t. the weight w(p) = p exp(p/T) on (0, inf).
The Liouville transform psi = w^{-1/2} chi maps it EXACTLY onto the radial
Coulomb Hamiltonian

    Lambda_l  ~  kappa [ -d^2/dp^2 + (l^2 - 1/4)/p^2 - 1/(2 T p) + 1/(4T^2) ],

i.e. angular momentum lambda = l - 1/2, charge Z = 1/(2T), energy offset
kappa/(4T^2) (the pure-drag relaxation floor).  Hence a Rydberg tower of
bound states below a continuum edge:

    gamma_{l,k} = (kappa / 4T^2) [ 1 - 1/(2l + 2k + 1)^2 ],   k = 0,1,2,...
    continuum:    gamma >= kappa / 4T^2.

Eigenfunctions: psi_{l,k}(p) ~ p^l exp(-p/2T - p/(4 T nu)) L_k^{(2l)}(p/(2 T nu)),
nu = l + k + 1/2 (generalized Laguerre).

One-hit closed form
-------------------
The l-dependence of Lambda_l is EXACTLY l^2 * (kappa/p^2).  Therefore, for any
source p-shape h(p) in channel n, the energy-weighted one-hit effective rate

    gamma_n^eff[h] = <p^2 | Lambda_n | h> / <p^2 | h>
                   = kappa [ (n^2 - 1) I0[h] + I1[h]/T ] / I2[h],
    Im[h] = int dp p^m h(p),

is of the mixed-MRTA form A[h] + n^2 B[h].  For the thermal shape
h = exp(-p/T):  A = 0 and gamma_n^eff = kappa n^2 / (2 T^2)  -- the pure
angular-diffusion (l^2) anchor.

Units in this module: T0 = 1, kappa = 1/2, so that gamma_2^eff(thermal) = 1
and the opacity knob g of the solver is normalized exactly as for the
'flat'/'diff' spectra of mrta.solver (ghat_2 = 1).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


KAPPA = 0.5      #: diffusion constant (units T0 = 1); gamma_2^eff(thermal) = 1
TBATH = 1.0      #: frozen bath temperature (reference/central temperature)


# ----------------------------------------------------------------------------
# exact analytic results
# ----------------------------------------------------------------------------

def rydberg_gamma(l: int, k: int, kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief Exact bound-state eigenvalue of the harmonic-l block.
    @param l  angular harmonic (l >= 2 for the anisotropy sector)
    @param k  radial quantum number (k = 0, 1, ...)
    @param kappa  momentum-diffusion coefficient
    @param T  bath temperature
    @return gamma_{l,k} = (kappa/4T^2)[1 - (2l+2k+1)^{-2}] (Coulomb/Rydberg tower)
    """
    nu2 = float(2 * l + 2 * k + 1) ** 2
    return kappa / (4.0 * T * T) * (1.0 - 1.0 / nu2)


def continuum_edge(kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief Drag-set edge of the essential spectrum.
    @return kappa/(4 T^2)
    """
    return kappa / (4.0 * T * T)


def onehit_rate(n: int, h, p, kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief Closed-form one-hit effective rate gamma_n^eff[h] for source shape
    h(p) (f-space) sampled on grid p:  kappa[(n^2-1) I0 + I1/T] / I2.
    """
    I0 = np.trapezoid(h, p)
    I1 = np.trapezoid(p * h, p)
    I2 = np.trapezoid(p * p * h, p)
    return kappa * ((n * n - 1.0) * I0 + I1 / T) / I2


def thermal_shape(p, T: float = TBATH):
    """!
    @brief f-space thermal shape exp(-p/T), unnormalized.
    @param p  momentum grid
    @return exp(-p/T)
    """
    return np.exp(-p / T)


# ----------------------------------------------------------------------------
# discrete operator: conservative Sturm-Liouville finite volume
# ----------------------------------------------------------------------------

def pgrid(Np: int, pmax: float, pmin: float = 0.0):
    """!
    @brief Cell-centered uniform grid p_i = pmin + (i+1/2) h on (pmin, pmax].
    @param Np  number of cells
    @param pmax  outer truncation
    @param pmin  explicit infrared cutoff (0 = none)
    @return (p, h): cell centers and spacing
    """
    h = (pmax - pmin) / Np
    return pmin + (np.arange(Np) + 0.5) * h, h


def operator_tridiag(l: int, Np: int, pmax: float,
                     kappa: float = KAPPA, T: float = TBATH,
                     pmin: float = 0.0, bc_inner: str = "noflux"):
    """!
    @brief Symmetric tridiagonal representation of Lambda_l (weight-space).

    @details
    Finite volume with fluxes P = kappa p e^{p/T} at faces.  BCs: no-flux at
    p = 0 (exact, P(0) = 0); Dirichlet (absorbing) at the truncation face
    p = pmax, which removes the spurious surface state that a reflecting
    truncation binds through the exponentially growing weight, and renders
    the operator strictly positive.  Exponentials are cancelled analytically
    so arbitrarily large pmax is safe:
        offdiag_i = -kappa p_{i+1/2} / (h^2 sqrt(p_i p_{i+1}))
                    (the e^{p/T} factors cancel exactly),
        diag_i    = kappa [ p_{i+1/2} e^{+h/2T} + p_{i-1/2} e^{-h/2T} ] / (p_i h^2)
                    - kappa (1/(p_i T) - l^2/p_i^2).
    Infrared generalization: the grid may start at an explicit cutoff
    pmin > 0, with inner boundary condition bc_inner in {"noflux",
    "dirichlet"} (reflecting / absorbing wall at pmin).  For pmin = 0 the
    two coincide with the exact natural condition, since the flux P(0) = 0.
    Returns (diag, offdiag, p, h).
    """
    p, h = pgrid(Np, pmax, pmin)
    pf = pmin + np.arange(1, Np) * h                # interior faces p_{i+1/2}
    off = -kappa * pf / (h * h * np.sqrt(p[:-1] * p[1:]))
    up = np.zeros(Np)
    dn = np.zeros(Np)
    up[:-1] = kappa * pf * np.exp(+0.5 * h / T) / (p[:-1] * h * h)
    dn[1:] = kappa * pf * np.exp(-0.5 * h / T) / (p[1:] * h * h)
    # Dirichlet ghost at the outer face: flux = P_out (0 - psi_{N-1})/(h/2)
    up[-1] = 2.0 * kappa * pmax * np.exp(+0.5 * h / T) / (p[-1] * h * h)
    if bc_inner == "dirichlet" and pmin > 0.0:
        dn[0] += 2.0 * kappa * pmin * np.exp(-0.5 * h / T) / (p[0] * h * h)
    elif bc_inner != "noflux" and bc_inner != "dirichlet":
        raise ValueError(f"unknown bc_inner {bc_inner!r}")
    diag = up + dn - kappa * (1.0 / (p * T) - l * l / (p * p))
    return diag, off, p, h


def spectrum_1d(l: int, Np: int = 4000, pmax: float = 200.0, nev: int = 6,
                kappa: float = KAPPA, T: float = TBATH,
                pmin: float = 0.0, bc_inner: str = "noflux"):
    """!
    @brief Lowest nev eigenvalues of Lambda_l on a fine 1D grid (validation
    against the closed-form Rydberg tower).
    @param l  angular harmonic
    @param Np, pmax, pmin, bc_inner  grid / boundary parameters
    @param nev  number of eigenvalues returned
    @return ascending eigenvalues (length nev)
    """
    from scipy.linalg import eigh_tridiagonal
    d, e, p, h = operator_tridiag(l, Np, pmax, kappa, T, pmin, bc_inner)
    vals = eigh_tridiagonal(d, e, select="i", select_range=(0, nev - 1),
                            eigvals_only=True)
    return vals


# ----------------------------------------------------------------------------
# eigen-factorization for the momentum-resolved solver (F = p^2 f space)
# ----------------------------------------------------------------------------

@dataclass
class KernelFactorization:
    """Per-harmonic factorization  Lambda~_l = R_l diag(Gam_l) L_l  in F-space."""
    p: np.ndarray            #: cell centers
    dp: float                #: cell width
    Gam: list                #: [lmax+1] arrays of eigenvalues (None for l<2)
    R: list                  #: right transforms, F-space
    L: list                  #: left transforms,  F-space (L @ R = 1)
    m: list                  #: measurement rows m_l = 1^T Lambda~_l (for one-hit)


def factorize(lmax: int, Np: int, pmax: float,
              kappa: float = KAPPA, T: float = TBATH,
              pmin: float = 0.0,
              bc_inner: str = "noflux",
              gain_proj: bool = False) -> KernelFactorization:
    """!
    @brief Diagonalize Lambda_l for l = 2..lmax on the solver p-grid and
    return F-space transforms.

    @details
    Weight-space symmetric S_l = D A_l D^{-1}, D = diag(sqrt(w_i)),
    w = p e^{p/T}; eigh(S_l) = U Gam U^T.  In F = p^2 f space:
        R_l = P2 D^{-1} U,   L_l = U^T D P2^{-1},   P2 = diag(p^2).
    exp of the collision substep is then  F <- Feq + R e^{-r Gam dt} L (F-Feq).
    """
    Gam, R, L, M = [None, None], [None, None], [None, None], [None, None]
    p, h = pgrid(Np, pmax, pmin)
    # D and P2 combine to p^2 / sqrt(p e^{p/T}) = p^{3/2} e^{-p/2T}: bounded.
    d_inv_p2 = p ** 1.5 * np.exp(-0.5 * p / T)       # P2 @ D^{-1} (diagonal)
    d_p2inv = 1.0 / d_inv_p2                         # D @ P2^{-1}
    if gain_proj:
        # Conservation-projected gain sector: in the l = 0, +-1 blocks apply
        # (1 - P) Lambda_l (1 - P) with P the weighted projection onto the
        # conserved-moment direction z(p) = p e^{-p/T} (simultaneously the
        # dT and du zero-mode shape).  In symmetric coordinates the moment
        # functional is <v, chi> with v = p^{3/2} e^{-p/2T} = d_inv_p2.
        # For l >= 2 the projector has no support and Lambda is unchanged.
        v = d_inv_p2 / np.linalg.norm(d_inv_p2)
        Gam, R, L, M = [], [], [], []
        for l in (0, 1):
            d, e, _, _ = operator_tridiag(l, Np, pmax, kappa, T,
                                          pmin, bc_inner)
            S = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
            S = S - np.outer(v, v @ S)
            S = S - np.outer(S @ v, v)
            S = 0.5 * (S + S.T)
            gam, U = np.linalg.eigh(S)
            gam = np.clip(gam, 0.0, None)        # zero mode: clip -1e-16 -> 0
            Rl = d_inv_p2[:, None] * U
            Ll = U.T * d_p2inv[None, :]
            Gam.append(gam); R.append(np.ascontiguousarray(Rl))
            L.append(np.ascontiguousarray(Ll))
            M.append(np.ascontiguousarray((Rl * gam[None, :]) @ Ll).sum(axis=0))
    for l in range(2, lmax + 1):
        d, e, _, _ = operator_tridiag(l, Np, pmax, kappa, T, pmin, bc_inner)
        S = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        gam, U = np.linalg.eigh(S)
        Rl = d_inv_p2[:, None] * U
        Ll = U.T * d_p2inv[None, :]
        Gam.append(gam)
        R.append(np.ascontiguousarray(Rl))
        L.append(np.ascontiguousarray(Ll))
        M.append(np.ascontiguousarray((Rl * gam[None, :]) @ Ll).sum(axis=0))
    return KernelFactorization(p=p, dp=h, Gam=Gam, R=R, L=L, m=M)


# ----------------------------------------------------------------------------
# transport-coefficient functionals
# ----------------------------------------------------------------------------

def viscosity_enhancement_exact() -> float:
    """!
    @brief EXACT Chapman-Enskog result for the FP kernel.

    @details
    Using Lambda_l (p^a e^{-p/T}) = kappa e^{-p/T} [ (l^2-a^2) p^{a-2}
    + (a/T) p^{a-1} ], the choice a = l collapses the first term:

        Lambda_l ( p^l e^{-p/T} ) = (l kappa / T) p^{l-1} e^{-p/T}.

    The l = 2 gradient source has f-space shape s = p e^{-p/T}, so the CE
    solution is exact and closed-form:  delta f_2 = (T/2kappa) p^2 e^{-p/T}.
    The stress moment then gives

        eta_FP / eta_flat |_{matched one-hit norm}
            = gammabar <p^2|Lambda_2^{-1}|s> / <p^2|s>
            = (2kappa/T^2)(T/2kappa) Gamma(5) T^5 / (Gamma(4) T^4) = 4  exactly.

    Equivalently: the viscosity-weighted effective l=2 eigenvalue is
    gamma_2^visc = kappa/(2T^2), a factor 4 below the response-weighted
    one-hit rate 2 kappa/T^2 -- the momentum entanglement makes 'the'
    l = 2 eigenvalue an observable-dependent notion.
    """
    return 4.0


def viscosity_enhancement(Np: int = 6000, pmax: float = 300.0,
                          kappa: float = KAPPA, T: float = TBATH,
                          pmin: float = 0.0,
                          bc_inner: str = "noflux") -> float:
    """!
    @brief Shear-viscosity enhancement of the FP kernel over flat RTA at
    matched dilute (one-hit) normalization.

    @details
    Chapman-Enskog: the l=2 gradient source has f-space shape
    s(p) = p e^{-p/T} (boost mode of the thermal state), the stress is the
    p^2 moment, so
        eta_FP / eta_flat = gammabar * <p^2 | Lambda_2^{-1} | s> / <p^2 | s>,
    with gammabar = 2 kappa / T^2 the one-hit thermal l=2 rate (the rate to
    which the flat kernel is normalized).  Since spec(Lambda_2) reaches down
    to gamma_{2,0} = 0.96 kappa/4T^2 << gammabar, the inverse average is
    dominated by the soft end of the spectrum and the ratio exceeds one.
    """
    from scipy.linalg import solveh_banded
    d, e, p, h = operator_tridiag(2, Np, pmax, kappa, T, pmin, bc_inner)
    w = np.exp(np.log(p) + np.clip(p / T, None, 600.0))    # p e^{p/T}
    sqw = np.sqrt(p) * np.exp(0.5 * p / T)
    s = p * np.exp(-p / T)                                  # CE source, f-space
    # solve Lambda_2 psi = s  via weight-space symmetric system
    rhs = sqw * s
    ab = np.zeros((2, Np))
    ab[0, 1:] = e
    ab[1, :] = d
    chi = solveh_banded(ab, rhs, lower=False)
    psi = chi / sqw
    # normalization: matched one-hit thermal l=2 rate ON THE SAME cut grid
    gammabar = onehit_rate_thermal_cut(2, pmin, kappa, T)
    return gammabar * np.sum(p * p * psi) / np.sum(p * p * s)


# ----------------------------------------------------------------------------
# infrared robustness: closed-form predictions with an explicit cutoff
# ----------------------------------------------------------------------------

def onehit_rate_thermal_cut(n: int, pmin: float,
                            kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief EXACT thermal-shape one-hit rate with an infrared cutoff a = pmin.

    @details
    The energy measure p^2 dp cancels the 1/p^2 singularity of the angular
    rate exactly, so the response functional is IR-finite and the cutoff
    enters only through incomplete moments of the thermal shape:
        I0 = T e^{-a/T},  I1 = T(a+T)e^{-a/T},  I2 = T(a^2+2aT+2T^2)e^{-a/T},
        gamma_n^eff(a) = kappa [ (n^2-1) I0 + I1/T ] / I2
                       = kappa (n^2 T + a) / (a^2 + 2 a T + 2 T^2).
    Small-a expansion: gamma_n^eff(a) = (kappa n^2/2T^2)
    [1 - (1 - 1/n^2)(a/T) + O(a^2)] -- a LINEAR, per-channel shift.
    """
    a = pmin
    return kappa * (n * n * T + a) / (a * a + 2.0 * a * T + 2.0 * T * T)


def d32_direct_cut(pmin: float, T: float = TBATH) -> float:
    """!
    @brief Exact direct-channel dilute double ratio with IR cutoff:
    D32(a) = gamma_3^eff(a)/gamma_2^eff(a) = (9T+a)/(4T+a), i.e.
    D32(a)/(9/4) = (1 + x/9)/(1 + x/4), x = a/T:
    a -14% * x/ ... linear drift, about -3.9% at x = 0.3.
    """
    return (9.0 * T + pmin) / (4.0 * T + pmin)


def visc_pred_cut(pmin: float, kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief Analytic prediction for the viscosity ratio with IR cutoff.

    @details
    The exact CE solution delta f_2 = (T/2kappa) p^2 e^{-p/T} vanishes
    quadratically at p -> 0, so an inner wall at a modifies it only through
    a homogeneous-solution correction of relative size O(a^4).  To that
    accuracy,
        X(a) = gammabar(a) * (T/2kappa) * Gamma(5, a/T)/Gamma(4, a/T) * T,
    with gammabar(a) = onehit_rate_thermal_cut(2, a) the matched-opacity
    normalization on the same cut domain, and Gamma the upper incomplete
    gamma function.  Small-a: X(a) = 4 [1 - (3/4)(a/T) + O(a^2)]: the whole
    linear drift is the (deliberate) renormalization of the opacity unit,
    not of the transport physics; at fixed physical normalization the ratio
    is 4 + O(a^4), i.e. UNCHANGED analytically.
    """
    from scipy.special import gammaincc, gamma as G
    x = pmin / T
    R43 = T * (gammaincc(5, x) * G(5)) / (gammaincc(4, x) * G(4))
    gbar = onehit_rate_thermal_cut(2, pmin, kappa, T)
    return gbar * (T / (2.0 * kappa)) * R43


def chi2(g: float, pmin: float = 0.0,
         kappa: float = KAPPA, T: float = TBATH) -> float:
    """!
    @brief Matched (physical) opacity variable chi_2.
    @param g  raw code opacity knob
    @param pmin  infrared cutoff of the momentum grid
    @return chi_2 = g * gamma_2^eff(pmin) / gamma_2^eff(0)

    @details
    The code knob g multiplies the collision operator, whose physical
    strength in the l = 2 channel depends on the infrared regulator through
    gamma_2^eff(pmin).  The regulator-invariant opacity is the accumulated
    l = 2 relaxation weight along the reference trajectory,
        chi_2 = int_{tau0}^{tauf} dtau gammabar_2^ref(tau)
              propto g * gamma_2^eff(pmin),
    the geometry factor being kernel- and regulator-independent.  In the
    normalization of this module (gamma_2^eff(0) = 1 at kappa = T^2/2) the
    dimensionless matched knob is simply
        chi_2 = g * gamma_2^eff(pmin) / gamma_2^eff(0),
    with gamma_2^eff given in closed form by onehit_rate_thermal_cut.
    For pmin = 0, chi_2 = g identically: the production scans are already
    expressed in the physical variable.
    """
    return g * onehit_rate_thermal_cut(2, pmin, kappa, T) \
        / onehit_rate_thermal_cut(2, 0.0, kappa, T)
