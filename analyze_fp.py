"""!
@file analyze_fp.py
@brief Collect results_fp/*.json, print the benchmark tables, and produce the
two-panel figure: (a) D32(g) for the separable 'diff' spectrum vs the
non-separable FP kernel, with dilute anchors and the late-time Rydberg
degeneracy; (b) perturbation hardening <p>(tau) and the operator spectrum."""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mrta import fp_kernel as fp

OUT = "results_fp"


def load():
    recs = {}
    for f in glob.glob(os.path.join(OUT, "*.json")):
        with open(f) as fh:
            recs[os.path.basename(f)[:-5]] = json.load(fh)
    return recs


def d32(recs, kern, g, base="flat"):
    a = recs[f"{kern}_g{g:g}"]["kappa"]
    b = recs[f"{base}_g{g:g}"]["kappa"]
    return (a["3"] / a["2"]) / (b["3"] / b["2"])


def main():
    recs = load()
    GS = [0.5, 1.0, 2.0, 4.0, 8.0]

    print("=== dilute (one-hit) anchors, alpha = 1 ===")
    f1, p1 = recs["flat_g1_1hit"]["kappa"], recs["fp_g1_1hit"]["kappa"]
    pd, ps = recs["fp_g1_1hit"]["kappa_direct"], recs["fp_g1_1hit"]["kappa_sub"]
    D32_dir_flat_needed = None
    for n in ("2", "3", "4"):
        print(f" n={n}: flat {f1[n]:+.4e}   fp {p1[n]:+.4e}"
              f"   (fp direct {pd[n]:+.4e}, sub {ps[n]:+.4e})")
    D32_tot = (p1["3"] / p1["2"]) / (f1["3"] / f1["2"])
    D42_tot = (p1["4"] / p1["2"]) / (f1["4"] / f1["2"])
    print(f" one-hit D32(total) = {D32_tot:.4f}   D42(total) = {D42_tot:.4f}")
    d1 = recs["diff_g1_1hit"]["kappa"]
    print(f" scalar-diff check D32 = "
          f"{(d1['3']/d1['2'])/(f1['3']/f1['2']):.6f} (thm: 2.25), "
          f"D42 = {(d1['4']/d1['2'])/(f1['4']/f1['2']):.6f} (thm: 4)")

    if "fp_g1_1hit_a0.333" in recs:
        fa = recs["flat_g1_1hit_a0.333"]["kappa"]
        pa = recs["fp_g1_1hit_a0.333"]["kappa"]
        pda = recs["fp_g1_1hit_a0.333"]["kappa_direct"]
        psa = recs["fp_g1_1hit_a0.333"]["kappa_sub"]
        D32a = (pa["3"] / pa["2"]) / (fa["3"] / fa["2"])
        print(f"=== alpha = 1/3: one-hit D32(total) = {D32a:.4f}"
              f"  (fp n=3 direct {pda['3']:+.3e} vs sub {psa['3']:+.3e})")

    print("=== opacity scan D32(g) ===")
    tab = {}
    for kern in ("diff", "fp"):
        tab[kern] = [d32(recs, kern, g) for g in GS]
        print(f" {kern:5s}: " + "  ".join(f"{v:.4f}" for v in tab[kern]))

    if "fp_g2_hiNp" in recs:
        for chk in ("fp_g2_hiNp", "fp_g2_hiNx"):
            a = recs[chk]["kappa"]
            b = recs["flat_g2"]["kappa"]
            print(f" resolution {chk}: D32 = {(a['3']/a['2'])/(b['3']/b['2']):.4f}"
                  f"  (base {d32(recs,'fp',2.0):.4f})")


    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    ax.axhline(2.25, color="C0", ls=":", lw=1)
    ax.axhline(1.0, color="gray", ls="-", lw=0.6)
    ax.plot(GS, tab["diff"], "o-", color="C0",
            label=r"separable $\gamma_l\propto l^2$ (MRTA)")
    below = [g for g in GS if g <= 2.0]
    ax.plot(below, tab["fp"][:len(below)], "s-", color="C3",
            label=r"leading-log QCD (FP), non-separable")
    ax.axvspan(2.0, 4.0, color="C3", alpha=0.12, lw=0)
    ax.annotate(r"$\kappa_3^{\rm FP}$ zero crossing" "\n"
                r"$g^{*}_{3}\in(2,4)$", (2.15, 0.25), fontsize=9, color="C3")
    ax.plot([0.15], [2.25], marker="*", ms=13, color="C0", clip_on=False)
    ax.plot([0.15], [D32_tot], marker="*", ms=13, color="C3", clip_on=False)
    ax.annotate(r"$\gamma_3/\gamma_2 = 9/4$", (0.16, 2.31), fontsize=9,
                color="C0")
    ax.annotate("one-hit", (0.16, D32_tot - 0.17), fontsize=8, color="C3")
    ax.set_xscale("log")
    ax.set_xlim(0.12, 9.5)
    ax.set_ylim(0.0, 2.55)
    ax.set_xlabel(r"opacity $g$")
    ax.set_ylabel(r"$D_{32} \equiv (\kappa_3/\kappa_2)\,/\,"
                  r"(\kappa_3/\kappa_2)_{\rm flat}$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("(a) double ratio: separable vs microscopic kernel",
                 fontsize=10)

    ax = axes[1]
    ax.axhline(3.0, color="gray", ls=":", lw=1)
    ax.annotate(r"thermal source $\langle p\rangle = 3\,T_0$",
                (0.35, 3.1), fontsize=9, color="gray")
    for g, c in ((0.5, "C1"), (2.0, "C3")):
        d = np.load(os.path.join(OUT, f"hardening_g{g:g}_n2.npz"))
        ax.plot(d["tau"], d["mean_p"], "-", color=c, label=rf"$g={g:g}$")
    ax.set_xlabel(r"$\tau/R$")
    ax.set_ylabel(r"$\langle p\rangle$ of the $V_2$-carrying profile  $[T_0]$")
    ax.set_ylim(2.6, 6.4)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("(b) the flow is carried by hard momenta", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fp_benchmark.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_fp_benchmark.pdf"))
    print("wrote figure")


if __name__ == "__main__":
    main()
