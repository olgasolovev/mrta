"""!
@brief mrta.solver -- 2D mode-resolved transport solver.

@details
Model (energy-weighted, |p|-integrated distribution Phi(x, phi, tau)):

    d_tau Phi + v(phi) . grad_x Phi
        = - r(x,tau) * sum_{|l|>=2} ghat_l [ Phi_l - Phi_eq_l ] e^{i l phi}

with local rate field

    r(x,tau) = g * (e(x,tau)/e_ref)^alpha * (tau0/tau  if bjorken else 1)

and ghat_l the mode-resolved relaxation spectrum, normalized to ghat_2 = 1
so that g == the l=2 rate (all spectra share the same shear viscosity).

Conservation of energy and momentum is EXACT by construction: the collision
term has no l = 0, +-1 components (gamma_0 = gamma_1 = 0), and the spectral
advection preserves every spatially-integrated phi-moment exactly.

Equilibrium: Landau-matched boosted thermal distribution of 2D massless
partons, which has the closed form

    Phi_eq(phi) = e_LRF / ( 2 pi (u^tau - u_x cos phi - u_y sin phi)^3 ),

normalized such that its lab-frame T^{mu nu} is the ideal conformal tensor
(P = e/2). Landau matching (timelike eigenvector of T^mu_nu) is solved per
grid point by power iteration on M^2 in a Numba kernel.

Numerics:
  * phi     : uniform grid, rfft harmonics (kernel is diagonal in l)
  * x       : uniform Cartesian grid; advection is an exact spectral
              translation per phi-slice (v is constant on a slice), so free
              streaming is exact to spectral accuracy
  * time    : Strang splitting  A(dt/2) C(dt) A(dt/2); the collision substep
              uses the EXACT exponential relaxation factor per harmonic,
              robust at arbitrary stiffness.

One-hit mode: evolves by pure free streaming and accumulates the master
formula   dV_n/dtau = -(2 pi / E_tot) int d^2x  gamma_n [Phi_{-n}-Phi^eq_{-n}]
which is the leading-opacity prediction the full solver must reproduce as
g -> 0 (validation gate V4).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from numba import njit, prange


# ----------------------------------------------------------------------------
## @brief parameters
# ----------------------------------------------------------------------------

@dataclass
class Params:
    ## @brief geometry / initial state (units: R = 1, Etot = 1)
    R: float = 1.0
    Etot: float = 1.0
    n_ecc: int = 2          #< harmonic of the eccentric perturbation
    beta: float = 0.0       #< perturbation amplitude beta_n  (Eq. eccentric Gaussian)

    ## @brief grid
    L: float = 9.0          #< half box size (units of R)
    Nx: int = 128
    Nphi: int = 64

    ## @brief time
    tau0: float = 0.1       #< initialization time (units of R); also Bjorken regulator scale
    tau_max: float = 6.0
    dt: float = 0.02

    ## @brief collision kernel
    g: float = 0.5          #< ghat_2 rate scale  (opacity knob)
    alpha: float = 1.0      #< rate  ~ (e/e_ref)^alpha
    bjorken: bool = True    #< tau0/tau dilution of the rate (regulator, Sec. 8 of the note)
    spectrum: str = "flat"  #< 'flat' | 'diff' | 'mcdiff' | 'mixed'
    b_over_a: float = 1.0   #< only for spectrum == 'mixed'

    ## @brief numerics
    e_thresh: float = 1e-7  #< skip collisions where e < e_thresh * e_ref
    u_tau_max: float = 5.0  #< reject cells whose Landau boost exceeds this
                            ## @brief (near-vacuum cells with slightly negative Phi can
                            # yield spurious large boosts whose equilibrium
                            # field spikes ~ gamma_L^3 in high harmonics; such
                            # cells carry no energy and are treated as vacuum)
    n_harm_obs: int = 6     #< record V_n for n = 1..n_harm_obs


def gamma_hat(l: np.ndarray, spectrum: str, b_over_a: float = 1.0) -> np.ndarray:
    """!
    @brief Relaxation spectrum ghat_l, normalized so ghat_2 = 1; ghat_{0,1} = 0.
    @param l See the function description.
    @param spectrum See the function description.
    @param b_over_a See the function description.
    @return The value described above.
    """
    l = np.asarray(l, dtype=np.float64)
    if spectrum == "flat":
        gh = np.ones_like(l)
    elif spectrum == "diff":            #< gamma_l ~ l^2
        gh = l**2 / 4.0
    elif spectrum == "mcdiff":          #< gamma_l ~ l^2 - 1  (momentum-conserving diffusion)
        gh = (l**2 - 1.0) / 3.0
    elif spectrum == "mixed":           #< gamma_l ~ a + b l^2
        b = b_over_a
        gh = (1.0 + b * l**2) / (1.0 + 4.0 * b)
    else:
        raise ValueError(f"unknown spectrum '{spectrum}'")
    gh = np.where(l < 2, 0.0, gh)       #< exact conservation constraints gamma_0 = gamma_1 = 0
    return gh


# ----------------------------------------------------------------------------
## @brief Numba kernel: Landau matching + closed-form equilibrium per grid point
# ----------------------------------------------------------------------------

@njit(parallel=True, fastmath=False, cache=True)
def _equilibrium_field(e, tx, ty, txx, txy, tyy, cphi, sphi, valid, out,
                       ut_max):
    """!
    @brief For each (flattened) grid point with valid[i]: solve the Landau

    @details
    eigenproblem T^mu_nu u^nu = e_LRF u^mu by power iteration on M^2,
        then fill out[i, :] with the closed-form boosted equilibrium
        Phi_eq(phi_k) = e_LRF / (2 pi (u^t - ux c_k - uy s_k)^3).
        Points with valid[i] == False are left untouched (treated as vacuum).
    @param e See the function description.
    @param tx See the function description.
    @param ty See the function description.
    @param txx See the function description.
    @param txy See the function description.
    @param tyy See the function description.
    @param cphi See the function description.
    @param sphi See the function description.
    @param valid See the function description.
    @param out See the function description.
    @param ut_max See the function description.
    """
    npt = e.shape[0]
    nphi = cphi.shape[0]
    twopi = 2.0 * np.pi
    for i in prange(npt):
        if not valid[i]:
            continue
        T00 = e[i]
        T0x = tx[i]
        T0y = ty[i]
        Txx = txx[i]
        Txy = txy[i]
        Tyy = tyy[i]

        ## @brief A = M @ M  with  M = T^mu_nu  (metric +,-,-), written out by hand
        a00 = T00 * T00 - T0x * T0x - T0y * T0y
        a01 = -T00 * T0x + T0x * Txx + T0y * Txy
        a02 = -T00 * T0y + T0x * Txy + T0y * Tyy
        a10 = T0x * T00 - Txx * T0x - Txy * T0y
        a11 = -T0x * T0x + Txx * Txx + Txy * Txy
        a12 = -T0x * T0y + Txx * Txy + Txy * Tyy
        a20 = T0y * T00 - Txy * T0x - Tyy * T0y
        a21 = -T0y * T0x + Txy * Txx + Tyy * Txy
        a22 = -T0y * T0y + Txy * Txy + Tyy * Tyy

        ## @brief power iteration on A (eigenvalues e^2, p1^2, p2^2; e is dominant)
        w0, w1, w2 = 1.0, 0.0, 0.0
        for _ in range(200):
            v0 = a00 * w0 + a01 * w1 + a02 * w2
            v1 = a10 * w0 + a11 * w1 + a12 * w2
            v2 = a20 * w0 + a21 * w1 + a22 * w2
            nrm = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
            if nrm == 0.0:
                v0, v1, v2 = 1.0, 0.0, 0.0
                break
            v0 /= nrm
            v1 /= nrm
            v2 /= nrm
            d0 = v0 - w0
            d1 = v1 - w1
            d2 = v2 - w2
            w0, w1, w2 = v0, v1, v2
            if d0 * d0 + d1 * d1 + d2 * d2 < 1e-28:
                break

        if w0 < 0.0:
            w0, w1, w2 = -w0, -w1, -w2

        s = w0 * w0 - w1 * w1 - w2 * w2
        if s <= 1e-14:
            ## @brief no timelike eigenvector (cell contaminated by negative Phi in
            # the far tail): dynamically irrelevant -- treat as vacuum
            valid[i] = False
            continue
        inv = 1.0 / np.sqrt(s)
        ut, ux, uy = w0 * inv, w1 * inv, w2 * inv
        if ut > ut_max:
            ## @brief spurious near-lightlike boost in a near-vacuum cell; the
            # equilibrium field would spike ~ ut^3 in high harmonics and,
            # weighted by ghat_l ~ l^2, contaminate the response.  Vacuum.
            valid[i] = False
            continue

        ## @brief e_LRF = u_mu T^{mu nu} u_nu   (quadratic form; robust)
        elrf = (T00 * ut * ut
                - 2.0 * ut * (T0x * ux + T0y * uy)
                + Txx * ux * ux + 2.0 * Txy * ux * uy + Tyy * uy * uy)
        if elrf < 0.0:
            elrf = 0.0

        pref = elrf / twopi
        for k in range(nphi):
            den = ut - ux * cphi[k] - uy * sphi[k]
            out[i, k] = pref / (den * den * den)


# ----------------------------------------------------------------------------
## @brief solver
# ----------------------------------------------------------------------------

class Solver:
    def __init__(self, par: Params):
        self.par = par
        p = par

        ## @brief spatial grid
        self.x = np.linspace(-p.L, p.L, p.Nx, endpoint=False) + p.L / p.Nx
        self.dx = self.x[1] - self.x[0]
        self.dA = self.dx * self.dx
        X, Y = np.meshgrid(self.x, self.x, indexing="ij")
        self.X, self.Y = X, Y
        self.rr = np.sqrt(X**2 + Y**2)
        self.th = np.arctan2(Y, X)

        ## @brief momentum-angle grid
        self.phi = 2.0 * np.pi * np.arange(p.Nphi) / p.Nphi
        self.dphi = 2.0 * np.pi / p.Nphi
        self.cphi = np.cos(self.phi)
        self.sphi = np.sin(self.phi)

        ## @brief rfft harmonic index and relaxation spectrum
        self.lmax = p.Nphi // 2
        self.ls = np.arange(self.lmax + 1)
        self.ghat = gamma_hat(self.ls, p.spectrum, p.b_over_a)

        ## @brief spectral advection phases for a half step dt/2 (per phi slice)
        kx = 2.0 * np.pi * np.fft.fftfreq(p.Nx, d=self.dx)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        arg = np.einsum("xy,k->xyk", KX, self.cphi) + np.einsum("xy,k->xyk", KY, self.sphi)
        self.phase_half = np.exp(-1j * arg * (p.dt / 2.0))

        ## @brief initial condition: eccentric Gaussian, isotropic in momentum
        prof = (p.Etot / (2.0 * np.pi * p.R**2)) * np.exp(-self.rr**2 / (2.0 * p.R**2))
        if p.beta != 0.0:
            prof = prof * (1.0 + p.beta * (self.rr / p.R)**p.n_ecc
                           * np.cos(p.n_ecc * self.th))
        self.E0 = prof
        self.e_ref = p.Etot / (2.0 * np.pi * p.R**2)   #< background central density
        self.Phi = np.repeat(prof[:, :, None], p.Nphi, axis=2) / (2.0 * np.pi)

        ## @brief scratch
        self._eq_flat = np.zeros((p.Nx * p.Nx, p.Nphi))

    ## @brief -- elementary substeps -------------------------------------------------

    def advect_half(self):
        """!
        @brief Exact spectral translation by v(phi) * dt/2 per slice.
        """
        F = np.fft.fft2(self.Phi, axes=(0, 1))
        F *= self.phase_half
        self.Phi = np.real(np.fft.ifft2(F, axes=(0, 1)))

    def _moments_and_eq(self):
        """!
        @brief rfft over phi; lab-frame T^{mu nu}; Landau + equilibrium field.
        @return The value described above.
        """
        p = self.par
        F = np.fft.rfft(self.Phi, axis=2)          #< F_l = Nphi * Phi_l
        dphi = self.dphi

        e = dphi * F[:, :, 0].real
        tx = dphi * F[:, :, 1].real
        ty = -dphi * F[:, :, 1].imag
        txx = 0.5 * dphi * (F[:, :, 0].real + F[:, :, 2].real)
        tyy = 0.5 * dphi * (F[:, :, 0].real - F[:, :, 2].real)
        txy = -0.5 * dphi * F[:, :, 2].imag

        valid = (e > p.e_thresh * self.e_ref)

        eq = self._eq_flat
        valid = valid.ravel()
        _equilibrium_field(e.ravel(), tx.ravel(), ty.ravel(),
                           txx.ravel(), txy.ravel(), tyy.ravel(),
                           self.cphi, self.sphi, valid, eq,
                           p.u_tau_max)
        valid = valid.reshape(e.shape)
        Feq = np.fft.rfft(eq.reshape(self.Phi.shape), axis=2)
        return F, Feq, e, valid

    def _rate_field(self, e, tau_mid):
        p = self.par
        r = p.g * np.clip(e / self.e_ref, 0.0, None) ** p.alpha
        if p.bjorken:
            r *= p.tau0 / tau_mid
        return r

    def collide(self, tau_mid, one_hit_acc=None):
        """!
        @brief Exact exponential relaxation of harmonics l >= 2 toward Landau

        @details
        equilibrium.  If one_hit_acc is given, skip relaxation and instead
                accumulate the one-hit master formula into it.
        @param tau_mid See the function description.
        @param one_hit_acc See the function description.
        """
        p = self.par
        F, Feq, e, valid = self._moments_and_eq()
        rate = self._rate_field(e, tau_mid)

        if one_hit_acc is not None:
            ## @brief dV_n = -(dt / S0) sum_x rate * ghat_n * conj(F_n - Feq_n)
            S0 = F[:, :, 0].real.sum()
            for n in range(2, len(one_hit_acc)):
                if n > self.lmax:
                    break
                dF = np.where(valid, (F[:, :, n] - Feq[:, :, n]), 0.0)
                one_hit_acc[n] += (-p.dt / S0) * self.ghat[n] \
                    * np.conj((rate * dF).sum())
            return

        decay = np.exp(-np.einsum("l,xy->xyl", self.ghat, rate * p.dt))
        Fnew = Feq + (F - Feq) * decay
        Fnew[:, :, 0] = F[:, :, 0]                 #< gamma_0 = 0 (energy)
        Fnew[:, :, 1] = F[:, :, 1]                 #< gamma_1 = 0 (momentum)
        F = np.where(valid[:, :, None], Fnew, F)
        self.Phi = np.fft.irfft(F, n=p.Nphi, axis=2)

    ## @brief -- observables ---------------------------------------------------------

    def observables(self):
        """!
        @brief V_n = conj(sum_x F_n) / sum_x F_0, energy, momentum (per unit dA).
        @return The value described above.
        """
        F = np.fft.rfft(self.Phi, axis=2)
        S = F.sum(axis=(0, 1))
        S0 = S[0].real
        nmax = min(self.par.n_harm_obs, self.lmax)
        Vn = np.conj(S[1:nmax + 1]) / S0
        E = self.dphi * S0 * self.dA
        Px = self.dphi * S[1].real * self.dA
        Py = -self.dphi * S[1].imag * self.dA
        return Vn, E, Px, Py

    ## @brief -- driver --------------------------------------------------------------

    def run(self, one_hit=False, store_every=10, verbose=False):
        p = self.par
        nsteps = int(round((p.tau_max - p.tau0) / p.dt))
        acc = np.zeros(self.par.n_harm_obs + 1, dtype=np.complex128) if one_hit else None

        taus, Vns, Es, Pxs, Pys = [], [], [], [], []

        def record(tau):
            Vn, E, Px, Py = self.observables()
            taus.append(tau)
            Vns.append(Vn.copy())
            Es.append(E)
            Pxs.append(Px)
            Pys.append(Py)

        tau = p.tau0
        record(tau)
        for it in range(nsteps):
            tau_mid = tau + 0.5 * p.dt
            self.advect_half()
            self.collide(tau_mid, one_hit_acc=acc)
            self.advect_half()
            tau += p.dt
            if (it + 1) % store_every == 0 or it == nsteps - 1:
                record(tau)
                if verbose:
                    print(f"  tau = {tau:6.3f}   Re V = "
                          + " ".join(f"{v.real:+.3e}" for v in Vns[-1]))

        out = {
            "params": asdict(p),
            "tau": np.array(taus),
            "Vn": np.array(Vns),               #< shape [ntimes, n_harm_obs]
            "E": np.array(Es),
            "Px": np.array(Pxs),
            "Py": np.array(Pys),
        }
        if one_hit:
            out["Vn_onehit"] = acc[1:self.par.n_harm_obs + 1].copy()
        return out


# ----------------------------------------------------------------------------
## @brief response-coefficient extraction
# ----------------------------------------------------------------------------

def eccentricity(E0, X, Y, n, dA):
    """!
    @brief Numerical eps_n of a profile (standard sign convention).
    @param E0 See the function description.
    @param X See the function description.
    @param Y See the function description.
    @param n See the function description.
    @param dA See the function description.
    @return The value described above.
    """
    r = np.sqrt(X**2 + Y**2)
    th = np.arctan2(Y, X)
    num = (r**n * np.exp(1j * n * th) * E0).sum() * dA
    den = (r**n * E0).sum() * dA
    return -num / den


def kappa_n(par: Params, n: int, delta: float, one_hit=False, **run_kw):
    """!
    @brief Centered-difference response  kappa_n = dv_n/deps_n at eps = 0.

    @details
    Runs the solver at beta = +-delta for harmonic n and returns
        (kappa_n, eps_n(delta), details).
    @param par See the function description.
    @param n See the function description.
    @param delta See the function description.
    @param one_hit See the function description.
    @return The value described above.
    """
    res = {}
    for s in (+1, -1):
        p = Params(**{**asdict(par), "beta": s * delta, "n_ecc": n})
        sol = Solver(p)
        if s == +1:
            eps = eccentricity(sol.E0, sol.X, sol.Y, n, sol.dA).real
        r = sol.run(one_hit=one_hit, **run_kw)
        res[s] = r["Vn_onehit"][n - 1].real if one_hit else r["Vn"][-1, n - 1].real
    kap = (res[+1] - res[-1]) / (2.0 * eps)
    return kap, eps, res
