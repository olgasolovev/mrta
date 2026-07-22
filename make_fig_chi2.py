"""!
@file make_fig_chi2.py
@brief Figure: the triangular zero crossing in the matched opacity variable.
Left: kappa_3 versus chi_2 -- the IR-cutoff configurations collapse onto the
baseline curve and cross zero in a common window.  Right: the same points
versus the raw code knob g -- the apparent regulator dependence that chi_2
removes."""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mrta.fp_kernel import onehit_rate_thermal_cut as G

OUT = "results_fp"


def jl(t):
    return json.load(open(os.path.join(OUT, t + ".json")))


def main():
    # baseline kappa_3(g) (pmin = 0 -> chi2 = g)
    gs = [0.5, 1.0, 2.0, 4.0, 8.0]
    k3b = [jl(f"fp_g{g:g}")["kappa"]["3"] for g in gs]

    cfgs = [("a015", 0.15, "C1", "o"), ("a030", 0.30, "C2", "s"),
            ("a060", 0.60, "C4", "D")]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)

    for ax, matched in ((axes[0], True), (axes[1], False)):
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.axvspan(2.0, 4.0, color="C3", alpha=0.10, lw=0)
        ax.plot(gs, np.array(k3b) * 1e3, "s-", color="C3",
                label=r"$p_{\min}=0$ (baseline; $\chi_2 \equiv g$)")
        for tag, a, c, m in cfgs:
            r = jl(f"chi2_{tag}")
            u = r["u"]
            chis = np.array([2.0, 4.0])
            k3 = np.array([r["k3_chi2"], r["k3_chi4"]]) * 1e3
            x = chis if matched else chis / u
            ax.plot(x, k3, m, color=c, ms=7,
                    label=rf"$p_{{\min}}={a}\,T_0$"
                          + (rf"  ($u={u:.2f}$)" if not matched else ""))
        ax.set_xlabel(r"matched opacity $\chi_2$" if matched
                      else r"raw code opacity $g$")
        ax.set_xlim(0.3, 9.0)
        ax.set_xscale("log")
        ax.set_title("(a) crossing in the physical variable" if matched
                     else "(b) the same runs versus the raw knob",
                     fontsize=10)
        if matched:
            ax.set_ylabel(r"$\kappa_3 \times 10^{3}$")
            ax.legend(fontsize=8.5, loc="lower left")
        else:
            ax.legend(fontsize=8.5, loc="lower left")
    fig.suptitle(r"Triangular zero crossing in matched opacity: "
                 r"$\chi_2 = g\,\gamma_2^{\rm eff}(p_{\min})/"
                 r"\gamma_2^{\rm eff}(0)$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_chi2.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_chi2.pdf"))
    print("wrote fig_chi2")
    # numbers for the text
    for tag, a, _, _ in cfgs:
        r = jl(f"chi2_{tag}")
        rho = (G(3, a) / G(3, 0)) / (G(2, a) / G(2, 0))
        print(f"{tag}: k3(chi2=2)={r['k3_chi2']:+.3e}  "
              f"k3(chi2=4)={r['k3_chi4']:+.3e}  rho={rho:.4f}")


if __name__ == "__main__":
    main()
