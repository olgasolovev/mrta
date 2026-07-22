"""!
@file solver_p.py
@brief mrta.solver_p -- momentum-resolved 2D solver with the linearized
leading-log QCD (Fokker-Planck) kernel: the non-separable benchmark.

@details
Model (energy-weighted, momentum-RESOLVED distribution F(x, phi, p, tau)
with F = p^2 f, so that  Phi(x,phi) = int dp F  reproduces the p-integrated
theory of mrta.solver):

    d_tau F + v(phi) . grad_x F
        = - r(x,tau) * sum_{|l|>=2} Lambda~_l [ F_l - F_l^eq ] e^{i l phi}

with the same local rate field as the MRTA solver,

    r(x,tau) = g * (e(x,tau)/e_ref)^alpha * (tau0/tau  if bjorken else 1),

but the SCALAR rates ghat_l replaced by the RADIAL OPERATORS Lambda_l of the
linearized Fokker-Planck (leading-log AMY) kernel -- see mrta.fp_kernel.
The kernel's angular relaxation rate is momentum-dependent,
gamma_l(p) = kappa l^2/p^2: the model is deliberately OUTSIDE the MRTA
separability ansatz gamma_l = ghat_l W(x,tau).

Normalization: kappa = 1/2, T0 = 1, so the one-hit thermal-shape effective
l = 2 rate equals g, exactly matching the ghat_2 = 1 convention of the
p-integrated spectra ('flat', 'diff', ...); double ratios between this
solver and the flat baseline are therefore taken at matched opacity.

Equilibrium: fixed-T0 fugacity matching -- the exact momentum-resolved lift
of the closed-form equilibrium of mrta.solver:

    F_eq(phi, p) = [e_LRF / (4 pi T0^3)] p^2 exp(-p sigma(phi) / T0),
    sigma = u^tau - u_x cos(phi) - u_y sin(phi),

whose p-integral is e_LRF/(2 pi sigma^3), identical to the p-integrated
solver's Phi_eq, and whose lab-frame T^{mu nu} is the same ideal conformal
tensor.  (Letting the bath scale run with a conformal local temperature is
a straightforward refinement that affects the background attenuation, not
the (l,p) entanglement under study.)

Conservation is EXACT by construction: the l = 0, +-1 channels are
untouched (gamma_0 = gamma_1 = 0), and spectral advection preserves every
spatially integrated phi-moment.

Numerics: same scheme as mrta.solver -- Strang splitting
A(dt/2) C(dt) A(dt/2) with consecutive half steps fused into full steps;
exact spectral translation for advection (rfft2; the translation phase is
p-independent for massless partons); the phi <-> harmonic transforms are
BLAS matmuls (fast for many short transforms); the collision substep
applies the exact operator exponential per harmonic via the precomputed
eigen-factorization  exp(-r dt Lambda~_l) = R_l exp(-r dt Gam_l) L_l,
unconditionally stable at any opacity.

One-hit mode: accumulates the momentum-resolved master formula
    dV_n/dtau = -(1/S0) sum_x r(x)  1^T Lambda~_n [F_n - F_n^eq](x, .)
with the direct (F) and equilibrium-subtraction (F_eq) pieces stored
separately, so the dilute anchors can be attributed term by term.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from numba import njit, prange

from .fp_kernel import factorize, KAPPA, TBATH


# ----------------------------------------------------------------------------
## @brief parameters
# ----------------------------------------------------------------------------

@dataclass
class ParamsP:
    ## @brief geometry / initial state (units: R = 1, Etot = 1, T0 = 1)
    R: float = 1.0
    Etot: float = 1.0
    n_ecc: int = 2
    beta: float = 0.0

    ## @brief grid
    L: float = 9.0
    Nx: int = 80
    Nphi: int = 32
    Np: int = 20            #< momentum-modulus cells
    pmax: float = 12.0      #< truncation of the p grid (units of T0)
    pmin: float = 0.0       #< explicit infrared cutoff (0 = none; exact
                            #  no-flux at p = 0 since the flux P(0) = 0)
    bc_inner: str = "noflux"  #< inner BC at pmin > 0: 'noflux'|'dirichlet'
    local_T: bool = False   #< run the bath scale of the operator and the
                            #  equilibrium with the conformal local
                            #  temperature T(x,tau) = T0 (e_LRF/e_ref)^{1/3}.
                            #  Exact self-similarity Lambda^{(T)} =
                            #  (T/T0) S^-1 Lambda^{(T0)} S (S: p -> p T0/T)
                            #  is used; the overall rate normalization is
                            #  held to the same r(x,tau) as the frozen-bath
                            #  model, isolating the SHAPE effect (the
                            #  conformal magnitude factor T/T0 is exactly an
                            #  alpha -> alpha + 1/3 shift, scanned
                            #  separately).  T is clipped to
                            #  [local_T_floor, 3] T0; the floor is a
                            #  resolution regularization active only in the
                            #  e-suppressed halo (contribution ~ e there).
    local_T_floor: float = 0.6
    gain_proj: bool = False   #< conservation-projected gain sector: relax
                              #  the l = 0, +-1 radial deviations under
                              #  (1-P) Lambda_l (1-P); identical to the
                              #  default (frozen l = 0, 1) in the whole
                              #  l >= 2 sector, E and P conserved exactly
                              #  in both.

    ## @brief time
    tau0: float = 0.1
    tau_max: float = 6.0
    dt: float = 0.03

    ## @brief collision kernel; opacity knob g as in Params
    kernel: str = "fp"      #< 'fp' (operator-valued, non-separable) or any
                            #  scalar spectrum of mrta.solver.gamma_hat
                            #  ('flat' | 'diff' | 'mcdiff' | 'mixed'), which
                            #  reproduces the p-integrated dynamics exactly
    b_over_a: float = 1.0   #< only for kernel == 'mixed'
    g: float = 0.5
    alpha: float = 1.0
    bjorken: bool = True

    ## @brief numerics
    e_thresh: float = 1e-7
    u_tau_max: float = 5.0
    n_harm_obs: int = 6


# ----------------------------------------------------------------------------
## @brief Numba kernel: Landau matching (returns the frame, not the field)
# ----------------------------------------------------------------------------

@njit(parallel=True, fastmath=False, cache=True)
def _match_field(e, tx, ty, txx, txy, tyy, valid, ut_a, ux_a, uy_a, el_a,
                 ut_max):
    """!
    @brief Per-point Landau matching by power iteration on M^2 (identical
    logic to solver._equilibrium_field), writing the Landau frame
    (u^tau, u_x, u_y, e_LRF) instead of the equilibrium field.
    """
    npt = e.shape[0]
    for i in prange(npt):
        if not valid[i]:
            continue
        T00 = e[i]; T0x = tx[i]; T0y = ty[i]
        Txx = txx[i]; Txy = txy[i]; Tyy = tyy[i]

        a00 = T00 * T00 - T0x * T0x - T0y * T0y
        a01 = -T00 * T0x + T0x * Txx + T0y * Txy
        a02 = -T00 * T0y + T0x * Txy + T0y * Tyy
        a10 = T0x * T00 - Txx * T0x - Txy * T0y
        a11 = -T0x * T0x + Txx * Txx + Txy * Txy
        a12 = -T0x * T0y + Txx * Txy + Txy * Tyy
        a20 = T0y * T00 - Txy * T0x - Tyy * T0y
        a21 = -T0y * T0x + Txy * Txx + Tyy * Txy
        a22 = -T0y * T0y + Txy * Txy + Tyy * Tyy

        w0, w1, w2 = 1.0, 0.0, 0.0
        for _ in range(200):
            v0 = a00 * w0 + a01 * w1 + a02 * w2
            v1 = a10 * w0 + a11 * w1 + a12 * w2
            v2 = a20 * w0 + a21 * w1 + a22 * w2
            nrm = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
            if nrm == 0.0:
                v0, v1, v2 = 1.0, 0.0, 0.0
                break
            v0 /= nrm; v1 /= nrm; v2 /= nrm
            d0 = v0 - w0; d1 = v1 - w1; d2 = v2 - w2
            w0, w1, w2 = v0, v1, v2
            if d0 * d0 + d1 * d1 + d2 * d2 < 1e-28:
                break

        if w0 < 0.0:
            w0, w1, w2 = -w0, -w1, -w2
        s = w0 * w0 - w1 * w1 - w2 * w2
        if s <= 1e-14:
            valid[i] = False
            continue
        inv = 1.0 / np.sqrt(s)
        ut, ux, uy = w0 * inv, w1 * inv, w2 * inv
        if ut > ut_max:
            valid[i] = False
            continue
        elrf = (T00 * ut * ut
                - 2.0 * ut * (T0x * ux + T0y * uy)
                + Txx * ux * ux + 2.0 * Txy * ux * uy + Tyy * uy * uy)
        if elrf < 0.0:
            elrf = 0.0
        ut_a[i] = ut; ux_a[i] = ux; uy_a[i] = uy; el_a[i] = elrf


# ----------------------------------------------------------------------------
## @brief solver
# ----------------------------------------------------------------------------

class SolverP:
    """!
    @brief Momentum-resolved deterministic solver for the conservation-
    projected leading-log Fokker-Planck kernel (and scalar spectra).
    @details State layout: F(x, y, p, phi) real, phi LAST (contiguous
    transforms).  Strang splitting A(dt/2) C(dt) A(dt/2) with fused half
    steps; exact spectral advection; exact per-harmonic operator
    exponentials via the precomputed eigenfactorization.
    """

    def __init__(self, par: ParamsP):
        self.par = par
        p = par

        ## spatial grid (identical to Solver)
        self.x = np.linspace(-p.L, p.L, p.Nx, endpoint=False) + p.L / p.Nx
        self.dx = self.x[1] - self.x[0]
        self.dA = self.dx * self.dx
        X, Y = np.meshgrid(self.x, self.x, indexing="ij")
        self.X, self.Y = X, Y
        self.rr = np.sqrt(X**2 + Y**2)
        self.th = np.arctan2(Y, X)

        ## momentum-angle grid
        self.phi = 2.0 * np.pi * np.arange(p.Nphi) / p.Nphi
        self.dphi = 2.0 * np.pi / p.Nphi
        self.cphi = np.cos(self.phi)
        self.sphi = np.sin(self.phi)
        self.lmax = p.Nphi // 2
        Nl = self.lmax + 1

        ## phi <-> harmonic DFT matrices (rfft convention: F_l = sum_k F_k e^{-il phi_k})
        ls = np.arange(Nl)
        LP = np.outer(self.phi, ls)                 # (Nphi, Nl)
        self.Ec = np.ascontiguousarray(np.cos(LP))
        self.Es = np.ascontiguousarray(-np.sin(LP))
        cl = np.full(Nl, 2.0); cl[0] = 1.0; cl[-1] = 1.0
        cl /= p.Nphi
        self.Wc = np.ascontiguousarray((cl[:, None] * np.cos(LP.T)))   # (Nl, Nphi)
        self.Ws = np.ascontiguousarray((cl[:, None] * np.sin(LP.T)))

        ## momentum-modulus grid + operator eigen-factorization
        self.K = factorize(self.lmax, p.Np, p.pmax, KAPPA, TBATH,
                           p.pmin, p.bc_inner, gain_proj=p.gain_proj)
        self.pgrid = self.K.p
        self.dp = self.K.dp
        if p.kernel == "fp":
            self.ghat = None
        else:                       #< scalar spectrum: Lambda~_l = ghat_l * Id
            from .solver import gamma_hat
            self.ghat = gamma_hat(np.arange(self.lmax + 1), p.kernel, p.b_over_a)

        ## spectral advection phases (p-independent); axis 1 is rfft axis
        kx = 2.0 * np.pi * np.fft.fftfreq(p.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.rfftfreq(p.Nx, d=self.dx)
        arg = (kx[:, None, None] * self.cphi[None, None, :]
               + ky[None, :, None] * self.sphi[None, None, :])
        self.phase_half = np.exp(-1j * arg * (p.dt / 2.0))   # (Nx, Nx/2+1, Nphi)
        self.phase_full = self.phase_half * self.phase_half

        ## initial condition: eccentric Gaussian x thermal p-shape (T0 = 1)
        prof = (p.Etot / (2.0 * np.pi * p.R**2)) * np.exp(-self.rr**2 / (2.0 * p.R**2))
        if p.beta != 0.0:
            prof = prof * (1.0 + p.beta * (self.rr / p.R)**p.n_ecc
                           * np.cos(p.n_ecc * self.th))
        self.E0 = prof
        self.e_ref = p.Etot / (2.0 * np.pi * p.R**2)
        h0 = self.pgrid**2 * np.exp(-self.pgrid / TBATH)
        h0 /= h0.sum() * self.dp                       #< int dp h0 = 1
        self.h0 = h0
        F = (prof[:, :, None, None] / (2.0 * np.pi)) * h0[None, None, :, None]
        self.F = np.ascontiguousarray(
            np.broadcast_to(F, (p.Nx, p.Nx, p.Np, p.Nphi)).copy())

    ## -- elementary substeps -------------------------------------------------

    def _advect(self, phase):
        """!
        @brief Exact spectral translation by v(phi) per (p, phi) slice.
        @param phase precomputed translation phases (Nx, Nx/2+1, Nphi)
        """
        Fh = np.fft.rfft2(self.F, axes=(0, 1))
        Fh *= phase[:, :, None, :]
        self.F = np.fft.irfft2(Fh, s=(self.par.Nx, self.par.Nx), axes=(0, 1))

    def advect_half(self):
        self._advect(self.phase_half)

    def advect_full(self):
        self._advect(self.phase_full)

    def _harmonics(self):
        """!
        @brief phi -> circular-harmonic transform via BLAS matmuls.
        @return (Fr, Fi): real/imag harmonic parts, each (npt, Np, Nl)
        """
        p = self.par
        A = self.F.reshape(-1, p.Nphi)
        Fr = (A @ self.Ec).reshape(-1, p.Np, self.lmax + 1)
        Fi = (A @ self.Es).reshape(-1, p.Np, self.lmax + 1)
        return Fr, Fi

    def _from_harmonics(self, Fr, Fi):
        p = self.par
        A = (Fr.reshape(-1, self.lmax + 1) @ self.Wc
             - Fi.reshape(-1, self.lmax + 1) @ self.Ws)
        self.F = np.ascontiguousarray(A.reshape(p.Nx, p.Nx, p.Np, p.Nphi))

    def _moments_and_frame(self, Fr, Fi):
        """!
        @brief p-summed lab-frame T^{mu nu} and per-point Landau matching.
        @return (e, valid, u^tau, u_x, u_y, e_LRF), flattened over space
        """
        p = self.par
        c = self.dphi * self.dp
        S0 = Fr[:, :, 0].sum(axis=1)
        S1r = Fr[:, :, 1].sum(axis=1); S1i = Fi[:, :, 1].sum(axis=1)
        S2r = Fr[:, :, 2].sum(axis=1); S2i = Fi[:, :, 2].sum(axis=1)
        e = c * S0
        tx = c * S1r
        ty = -c * S1i
        txx = 0.5 * c * (S0 + S2r)
        tyy = 0.5 * c * (S0 - S2r)
        txy = -0.5 * c * S2i

        valid = (e > p.e_thresh * self.e_ref)
        n = e.size
        ut = np.ones(n); ux = np.zeros(n); uy = np.zeros(n); el = np.zeros(n)
        _match_field(e, tx, ty, txx, txy, tyy,
                     valid, ut, ux, uy, el, p.u_tau_max)
        return e, valid, ut, ux, uy, el

    def _equilibrium_harmonics(self, idx, ut, ux, uy, el, Tloc=None):
        """!
        @brief F_eq harmonics on the valid points idx: rfft over phi of
        [e_LRF/(4 pi T^3)] p^2 exp(-p sigma(phi)/T), with T = T0 (frozen)
        or the conformal local temperature (local_T)."""
        p = self.par
        T = TBATH if Tloc is None else Tloc[:, None]             # (nv,1)|scalar
        sigma = (ut[idx, None] - ux[idx, None] * self.cphi[None, :]
                 - uy[idx, None] * self.sphi[None, :]) / T       # (nv, Nphi)
        Feq = np.exp(-self.pgrid[None, :, None]
                     * sigma[:, None, :])                        # (nv, Np, Nphi)
        T3 = TBATH**3 if Tloc is None else (Tloc**3)[:, None, None]
        Feq *= (el[idx, None, None] / (4.0 * np.pi * T3)) \
            * (self.pgrid**2)[None, :, None]
        A = Feq.reshape(-1, p.Nphi)
        Er = (A @ self.Ec).reshape(-1, p.Np, self.lmax + 1)
        Ei = (A @ self.Es).reshape(-1, p.Np, self.lmax + 1)
        return Er, Ei

    @staticmethod
    def _resample(G, pgrid, dp, pmin, scale):
        """!
        @brief Sample G (nv, Np) at positions p_j * scale[i] (linear interp,
        zero-padded at p = pmin-dp/2 and beyond pmax).  scale = 1 is exact
        identity."""
        import numpy as np
        nv, Np = G.shape
        s = pgrid[None, :] * scale[:, None]                 # (nv, Np)
        t = (s - (pmin + 0.5 * dp)) / dp                    # cell index space
        i = np.floor(t).astype(np.int64)
        w = t - i
        lo = np.clip(i, -1, Np - 1)
        hi = np.clip(i + 1, 0, Np)
        pad = np.concatenate([G, np.zeros((nv, 1), G.dtype)], axis=1)
        # index -1 -> value 0 (left pad conceptually at p ~ pmin)
        left = np.where(lo[:, :] >= 0,
                        np.take_along_axis(pad, np.clip(lo, 0, Np), axis=1),
                        0.0)
        right = np.take_along_axis(pad, hi, axis=1)
        return (1.0 - w) * left + w * right

    def _rate_field(self, e, tau_mid):
        p = self.par
        r = p.g * np.clip(e / self.e_ref, 0.0, None) ** p.alpha
        if p.bjorken:
            r *= p.tau0 / tau_mid
        return r

    def collide(self, tau_mid, one_hit_acc=None):
        """!
        @brief Collision substep: exact operator-exponential relaxation of
        harmonics toward the Landau-matched equilibrium (valid points only).
        @details l >= 2 always; with gain_proj also the conservation-
        projected l = 0, 1 radial dynamics; with local_T the change-only
        resampled self-similar operator.  If one_hit_acc = (direct, sub) is
        given, the one-hit master formula is accumulated instead (direct
        and equilibrium-subtraction pieces stored separately).
        @param tau_mid  midpoint time of the step (rate field evaluation)
        @param one_hit_acc  optional pair of complex accumulators
        """
        p = self.par
        Fr, Fi = self._harmonics()
        e, valid, ut, ux, uy, el = self._moments_and_frame(Fr, Fi)
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            return
        rate = self._rate_field(e, tau_mid)[idx]
        if p.local_T:
            Tloc = np.clip((el[idx] / self.e_ref) ** (1.0 / 3.0),
                           p.local_T_floor, 3.0)
        else:
            Tloc = None
        Er, Ei = self._equilibrium_harmonics(idx, ut, ux, uy, el, Tloc)
        K = self.K

        if one_hit_acc is not None:
            acc_dir, acc_sub = one_hit_acc
            S0 = Fr[:, :, 0].sum()
            for n in range(2, len(acc_dir)):
                if n > self.lmax:
                    break
                m = K.m[n] if self.ghat is None \
                    else self.ghat[n] * np.ones(p.Np)
                dir_r = (rate * (Fr[idx, :, n] @ m)).sum()
                dir_i = (rate * (Fi[idx, :, n] @ m)).sum()
                sub_r = (rate * (Er[:, :, n] @ m)).sum()
                sub_i = (rate * (Ei[:, :, n] @ m)).sum()
                # V_n uses conj(F_n): accumulate conj increments
                acc_dir[n] += (-p.dt / S0) * (dir_r - 1j * dir_i)
                acc_sub[n] += (+p.dt / S0) * (sub_r - 1j * sub_i)
            return

        rdt = rate * p.dt
        l_start = 0 if (p.gain_proj and self.ghat is None
                        and not p.local_T) else 2
        for l in range(l_start, self.lmax + 1):
            Gr = np.ascontiguousarray(Fr[idx, :, l]) - Er[:, :, l]
            Gi = np.ascontiguousarray(Fi[idx, :, l]) - Ei[:, :, l]
            if self.ghat is None:
                if p.local_T:
                    # apply only the CHANGE through the rescaled operator:
                    # e^{-r dt Lam_loc} G = G + S^{-1} (e^{-r dt Gam} - 1) S G,
                    # exact at zero rate (no spurious S^{-1}S != 1 error).
                    Hr = self._resample(Gr, self.pgrid, self.dp, p.pmin, Tloc)
                    Hi = self._resample(Gi, self.pgrid, self.dp, p.pmin, Tloc)
                    dec1 = np.exp(-np.outer(rdt, K.Gam[l])) - 1.0
                    dr = ((Hr @ K.L[l].T) * dec1) @ K.R[l].T
                    di = ((Hi @ K.L[l].T) * dec1) @ K.R[l].T
                    inv = 1.0 / Tloc
                    Gr2 = Gr + self._resample(dr, self.pgrid, self.dp,
                                              p.pmin, inv)
                    Gi2 = Gi + self._resample(di, self.pgrid, self.dp,
                                              p.pmin, inv)
                else:
                    dec = np.exp(-np.outer(rdt, K.Gam[l]))
                    Gr2 = ((Gr @ K.L[l].T) * dec) @ K.R[l].T
                    Gi2 = ((Gi @ K.L[l].T) * dec) @ K.R[l].T
                Fr[idx, :, l] = Er[:, :, l] + Gr2
                Fi[idx, :, l] = Ei[:, :, l] + Gi2
            else:
                dec = np.exp(-rdt * self.ghat[l])[:, None]
                Fr[idx, :, l] = Er[:, :, l] + Gr * dec
                Fi[idx, :, l] = Ei[:, :, l] + Gi * dec
        self._from_harmonics(Fr, Fi)

    ## -- observables ---------------------------------------------------------

    def observables(self):
        Fr, Fi = self._harmonics()
        Sr = Fr.sum(axis=(0, 1)); Si = Fi.sum(axis=(0, 1))
        S0 = Sr[0]
        nmax = min(self.par.n_harm_obs, self.lmax)
        Vn = (Sr[1:nmax + 1] - 1j * Si[1:nmax + 1]) / S0
        E = self.dphi * S0 * self.dA * self.dp
        Px = self.dphi * Sr[1] * self.dA * self.dp
        Py = -self.dphi * Si[1] * self.dA * self.dp
        return Vn, E, Px, Py

    def Vn_binned(self, edges, nmax=4):
        """!
        @brief Momentum-differential flow: V_n restricted to p-bins
        [edges[i], edges[i+1]), energy-weighted within the bin,
        V_n^bin = sum_{x, p in bin} F_n / sum_{x, p in bin} F_0.
        Returns array (nbins, nmax) complex."""
        Fr, Fi = self._harmonics()
        Sr = Fr.sum(axis=0)                      # (Np, Nl)
        Si = Fi.sum(axis=0)
        out = np.zeros((len(edges) - 1, nmax), dtype=np.complex128)
        for b in range(len(edges) - 1):
            m = (self.pgrid >= edges[b]) & (self.pgrid < edges[b + 1])
            S0 = Sr[m, 0].sum()
            for n in range(1, nmax + 1):
                out[b, n - 1] = (Sr[m, n].sum() - 1j * Si[m, n].sum()) / S0
        return out

    def mean_p_perturb(self, n):
        """!
        @brief <p> of the momentum profile carrying V_n: modulus of the
        spatially integrated complex harmonic, |int d^2x F_n(x,p)|, i.e.
        the p-differential flow profile.  The source (free-streaming
        eccentric deviation) has the thermal value <p> = 3 T_0; growth
        above it is the momentum-space escape of the flow-carrying
        deviation.  (Meaningful once V_n != 0; under pure free streaming
        the integrated harmonic vanishes identically by Prop. 1.)"""
        Fr, Fi = self._harmonics()
        cr = Fr[:, :, n].sum(axis=0)
        ci = Fi[:, :, n].sum(axis=0)
        c = np.sqrt(cr**2 + ci**2)                                 # (Np,)
        den = c.sum()
        return (c * self.pgrid).sum() / den if den > 0 else np.nan

    ## -- driver --------------------------------------------------------------

    def run(self, one_hit=False, store_every=10, verbose=False, track_p=None):
        p = self.par
        nsteps = int(round((p.tau_max - p.tau0) / p.dt))
        acc = None
        if one_hit:
            acc = (np.zeros(p.n_harm_obs + 1, dtype=np.complex128),
                   np.zeros(p.n_harm_obs + 1, dtype=np.complex128))

        taus, Vns, Es, Pxs, Pys, mps = [], [], [], [], [], []

        def record(tau):
            Vn, E, Px, Py = self.observables()
            taus.append(tau); Vns.append(Vn.copy())
            Es.append(E); Pxs.append(Px); Pys.append(Py)
            if track_p is not None:
                mps.append(self.mean_p_perturb(track_p))

        tau = p.tau0
        record(tau)
        pending_half = False        # True when state is mid-cell (after A_half)
        self.advect_half()
        pending_half = True
        for it in range(nsteps):
            tau_mid = tau + 0.5 * p.dt
            self.collide(tau_mid, one_hit_acc=acc)
            tau += p.dt
            do_rec = ((it + 1) % store_every == 0) or it == nsteps - 1
            if do_rec:
                self.advect_half()
                pending_half = False
                record(tau)
                if verbose:
                    print(f"  tau = {tau:6.3f}   Re V = "
                          + " ".join(f"{v.real:+.3e}" for v in Vns[-1]))
                if it != nsteps - 1:
                    self.advect_half()
                    pending_half = True
            else:
                self.advect_full()

        out = {
            "params": asdict(p),
            "tau": np.array(taus),
            "Vn": np.array(Vns),
            "E": np.array(Es),
            "Px": np.array(Pxs),
            "Py": np.array(Pys),
        }
        if track_p is not None:
            out["mean_p"] = np.array(mps)
        if one_hit:
            out["Vn_onehit"] = (acc[0] + acc[1])[1:p.n_harm_obs + 1].copy()
            out["Vn_onehit_direct"] = acc[0][1:p.n_harm_obs + 1].copy()
            out["Vn_onehit_sub"] = acc[1][1:p.n_harm_obs + 1].copy()
        return out


# ----------------------------------------------------------------------------
## @brief response-coefficient extraction
# ----------------------------------------------------------------------------

def kappa_n_p(par: ParamsP, n: int, delta: float, one_hit=False,
              two_sided=False, **run_kw):
    """!
    @brief Response kappa_n = dv_n/deps_n at eps = 0 for the FP kernel.

    @details
    By rotation covariance, beta -> -beta is a rotation by pi/n of the
    initial profile, so V_n(-beta) = -V_n(+beta) EXACTLY and a single-sided
    run suffices (the O(beta^2) profile terms feed only harmonics 0 and 2n).
    Set two_sided=True to run the centered difference as a cross-check.
    """
    from .solver import eccentricity

    def one(sign):
        pp = ParamsP(**{**asdict(par), "beta": sign * delta, "n_ecc": n})
        sol = SolverP(pp)
        eps = eccentricity(sol.E0, sol.X, sol.Y, n, sol.dA).real
        r = sol.run(one_hit=one_hit, **run_kw)
        v = r["Vn_onehit"][n - 1].real if one_hit else r["Vn"][-1, n - 1].real
        return v, eps, r

    vp, eps, rp = one(+1)
    if two_sided:
        vm, _, _ = one(-1)
        kap = (vp - vm) / (2.0 * eps)
    else:
        kap = vp / eps
    return kap, eps, rp
