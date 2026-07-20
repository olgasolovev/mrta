# mrta — 2D mode-resolved transport solver

Numerical solver for the mode-resolved relaxation-time approximation (RTA) in
2D massless kinetic theory, built to test the **ratio theorem**

    (kappa_3/kappa_2)|_spectrum / (kappa_3/kappa_2)|_flat  ->  gamma_3/gamma_2

(leading order in opacity) and to compute the full opacity dependence of the
eccentricity-to-flow response coefficients kappa_n = v_n / eps_n for
different relaxation spectra gamma_l — the "money plot" of the PLB letter.

## Model

Energy-weighted, |p|-integrated distribution Phi(x, phi, tau):

    d_tau Phi + v(phi).grad Phi
        = - g (e/e_ref)^alpha (tau0/tau) sum_{|l|>=2} ghat_l [Phi_l - Phi_eq_l] e^{il phi}

* `ghat_l` spectra (all normalized to ghat_2 = 1, so every spectrum shares the
  same shear viscosity and any kappa_n difference at matched opacity is a pure
  spectrum effect):
  - `flat`    : ghat_l = 1                (standard RTA)
  - `diff`    : ghat_l = l^2/4            (angular diffusion)
  - `mcdiff`  : ghat_l = (l^2-1)/3        (momentum-conserving diffusion)
  - `mixed`   : ghat_l = (1 + b l^2)/(1 + 4b),  b = b_over_a (elastic/inelastic mix)
* gamma_0 = gamma_1 = 0 exactly (energy/momentum conservation constraints).
* Equilibrium: Landau-matched boosted thermal state of 2D massless partons,
  closed form Phi_eq = e_LRF / (2 pi (u.v)^3), matched per grid point.
* `tau0/tau` factor: the Bjorken regulator of the geometry integral (Sec. 8 of
  the derivation note); switchable.

## Numerics

* phi: uniform grid + rfft (the kernel is diagonal in l).
* x: uniform Cartesian grid; advection is an **exact spectral translation**
  per phi slice (v constant on a slice) — free streaming is exact to spectral
  accuracy, and every spatially integrated phi-moment (hence every V_n) is
  preserved by advection to machine precision.
* time: Strang splitting A(dt/2) C(dt) A(dt/2); the collision substep uses the
  exact per-harmonic exponential relaxation factor -> unconditionally stable
  at any opacity (stiff hydro limit included).
* Landau matching: power iteration on M^2 (M = T^mu_nu) in a Numba kernel.
* Conservation is exact by construction (verified to 1e-14 over full runs).

## Layout

    mrta/solver.py       core solver, Params, kappa_n extraction
    mrta/analytic.py     validation oracle (exact free streaming, closed-form
                         eccentricities, dilute-limit ratio theorem)
    tests_validation.py  gates V1-V4 (run: python tests_validation.py [--fast])
    run_scan.py          production driver (one JSON per (spectrum, g) point;
                         Slurm-job-array friendly)

## Validation status (all gates PASS)

| gate | statement | result |
|------|-----------|--------|
| V1 | advection vs exact free streaming | rel. err 4e-11 |
| V2 | homogeneous harmonic decay e^{-ghat_l g t}, all spectra | 1e-10 |
| V3 | exact E, P conservation in coupled run | 1e-14 |
| V4a | dilute limit vs one-hit master formula, n=2,3 | <0.1% |
| V4b | **ratio theorem: full-solver double ratio = 2.2502 vs 9/4** | 0.01% |

Remaining plan gates: V5 (flat-RTA kappa_2(opacity) vs the published
Ambrus-Schlichting-Werthmann curve — requires mapping g to their opacity
convention) and V6 (grid-refinement convergence at the production resolution).

## Usage

Validation:

    python tests_validation.py          # full
    python tests_validation.py --fast   # reduced grids, ~90 s

One scan point (response coefficients for n = 2, 3, 4):

    python run_scan.py --spectrum diff --g 1.0 --out results/ --linearity-check

Opacity scan as a Slurm array: see the docstring of run_scan.py.
One-hit reference values: add `--one-hit`.

## Numerical lessons learned (important for reproduction)

1. **Landau guards are physics-critical.** Near-vacuum tail cells that carry
   slightly negative Phi (polynomial eccentric perturbations make E0 < 0 at
   large r) can yield spurious near-lightlike Landau boosts; the equilibrium
   field then spikes ~ u_tau^3 in *high* harmonics, and spectra with growing
   ghat_l amplify exactly those deposits. Guards: cells with no timelike
   eigenvector or u_tau > u_tau_max (default 5) are treated as vacuum, and
   e_thresh = 1e-7 (relative). With the guards, the beta -> -beta rotation
   antisymmetry of V_3 holds to 1e-16; without them it was violated at 1e-5
   for the `diff` spectrum.
2. **The dilute gate must sit deep in the linear regime.** The odd-n response
   at alpha = 1 is selection-rule suppressed at leading order (Sec. 7 of the
   note), so its relative finite-opacity corrections are parametrically
   enhanced (~50 g empirically, vs ~0.05 g for n = 2). Gate V4 runs at
   g = 2e-4.
3. **Grid symmetry diagnostic.** beta -> -beta equals a rotation by pi/n; for
   n = 2 this is a lattice symmetry of the square grid, for n = 3 it is not —
   the residual of V_n(+beta) + V_n(-beta) is therefore a sensitive
   free diagnostic of grid contamination for odd n. Use it.

## Physics conventions

* Units: R = 1, Etot = 1. Response kappa_n = dv_n/deps_n at eps -> 0 via
  centered difference over beta = +-delta (default delta = 0.04; use
  --linearity-check to verify).
* eps_n computed numerically from the discretized profile (closed form in
  mrta/analytic.py as a cross-check).
* Opacity knob is g (the l = 2 rate scale). The mapping to the ASW opacity
  parameter gamma-hat is a convention choice deferred to gate V5.

## Requirements

numpy, numba (tested: numpy 2.4, numba 0.66). Single scan point at production
resolution (Nx = 192, Nphi = 128, dt = 0.02, tau_max = 8): a few minutes on
one node; the full money-plot scan (4 spectra x 12 opacities x 3 harmonics)
is a parallel job array.
