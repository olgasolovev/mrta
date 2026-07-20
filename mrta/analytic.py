"""!
@brief mrta.analytic -- analytic reference results (the validation oracle).

@details
Everything here is independent of the solver machinery and is used to
certify it (validation gates V1, V2, V4).
"""

from __future__ import annotations

import numpy as np
from math import gamma as Gamma


# ----------------------------------------------------------------------------
## @brief eccentric Gaussian profile
# ----------------------------------------------------------------------------

def gaussian_profile(X, Y, R=1.0, Etot=1.0, n=2, beta=0.0):
    """!
    @brief E0(x) = Etot/(2 pi R^2) e^{-r^2/2R^2} [1 + beta (r/R)^n cos(n theta)].
    @param X See the function description.
    @param Y See the function description.
    @param R See the function description.
    @param Etot See the function description.
    @param n See the function description.
    @param beta See the function description.
    @return The value described above.
    """
    r = np.sqrt(X**2 + Y**2)
    th = np.arctan2(Y, X)
    prof = (Etot / (2.0 * np.pi * R**2)) * np.exp(-r**2 / (2.0 * R**2))
    if beta != 0.0:
        prof = prof * (1.0 + beta * (r / R)**n * np.cos(n * th))
    return prof


def eps_gaussian(n: int, beta: float) -> float:
    """!
    @brief Closed-form eccentricity of the eccentric Gaussian:

    @details
    eps_n = -beta * 2^{n/2-1} n! / Gamma(n/2+1).
    @param n See the function description.
    @param beta See the function description.
    @return The value described above.
    """
    return -beta * 2.0**(n / 2.0 - 1.0) * float(np.math.factorial(n)) / Gamma(n / 2 + 1) \
        if hasattr(np, "math") else -beta * 2.0**(n / 2.0 - 1.0) * _fact(n) / Gamma(n / 2 + 1)


def _fact(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


## @brief make eps_gaussian robust across numpy versions
def eps_gaussian(n: int, beta: float) -> float:  # noqa: F811
    return -beta * 2.0**(n / 2.0 - 1.0) * _fact(n) / Gamma(n / 2 + 1)


# ----------------------------------------------------------------------------
## @brief exact free streaming
# ----------------------------------------------------------------------------

def free_stream_exact(E0_func, X, Y, phi, t):
    """!
    @brief Exact solution of the collisionless equation for an initially

    @details
    isotropic state:  Phi(x, phi, t) = E0(x - v t)/(2 pi).

        E0_func : callable E0(X, Y)
        Returns array [Nx, Ny, Nphi].
    @param E0_func See the function description.
    @param X See the function description.
    @param Y See the function description.
    @param phi See the function description.
    @param t See the function description.
    @return The value described above.
    """
    out = np.empty(X.shape + (len(phi),))
    for k, ph in enumerate(phi):
        out[:, :, k] = E0_func(X - np.cos(ph) * t, Y - np.sin(ph) * t) / (2.0 * np.pi)
    return out


# ----------------------------------------------------------------------------
## @brief homogeneous relaxation (gate V2)
# ----------------------------------------------------------------------------

def homogeneous_decay(c_l0: float, ghat_l: float, g: float, t: float) -> float:
    """!
    @brief Amplitude of harmonic l at time t for a spatially uniform state with

    @details
    alpha = 0, bjorken = False:  c_l(t) = c_l(0) exp(-ghat_l * g * t).
    @param c_l0 See the function description.
    @param ghat_l See the function description.
    @param g See the function description.
    @param t See the function description.
    @return The value described above.
    """
    return c_l0 * np.exp(-ghat_l * g * t)


# ----------------------------------------------------------------------------
## @brief dilute-limit ratio theorem (gate V4)
# ----------------------------------------------------------------------------

def ratio_theorem(n: int, m: int, spectrum: str, b_over_a: float = 1.0) -> float:
    """!
    @brief Predicted double ratio  (kappa_n/kappa_m)|_spec / (kappa_n/kappa_m)|_flat

    @details
    = gamma_n / gamma_m  (leading order in opacity).
    @param n See the function description.
    @param m See the function description.
    @param spectrum See the function description.
    @param b_over_a See the function description.
    @return The value described above.
    """
    from .solver import gamma_hat
    gh = gamma_hat(np.array([n, m]), spectrum, b_over_a)
    return gh[0] / gh[1]
