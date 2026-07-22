"""!
@file make_fig_pbins.py
@brief Figure: momentum-space spectral tomography."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_fp"
d = np.load(os.path.join(OUT, "rate_tomography.npz"))
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

ax = axes[0]
ax.axhline(2.25, color="gray", ls=":", lw=1)
ax.axhline(1.0204, color="gray", ls="--", lw=1)
ax.annotate(r"$\gamma_3/\gamma_2 = 9/4$ (one-hit, all bins)", (0.55, 2.28),
            fontsize=8.5, color="gray")
ax.annotate(r"$\gamma_{3,0}/\gamma_{2,0} = 1.02$", (0.55, 1.05),
            fontsize=8.5, color="gray")
for key, c, lab in (("soft", "C0", r"soft $[0,2)\,T_0$"),
                    ("mid", "C2", r"mid $[2,4)\,T_0$"),
                    ("hard", "C3", r"hard $[4,8)\,T_0$")):
    ax.plot(d["ts"][1:], d[key][1:], "o-", color=c, label=lab)
ax.set_xscale("log")
ax.set_xlabel(r"accumulated relaxation weight $t = g\int r\,d\tau$")
ax.set_ylabel(r"$\gamma_3^{\rm eff,bin}/\gamma_2^{\rm eff,bin}$")
ax.set_ylim(0.95, 2.45)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("(a) rate tomography (exact): the fan inverts naive ordering",
             fontsize=10)

ax = axes[1]
gs = (0.5, 2.0, 8.0)
x = np.arange(3)
wdt = 0.25
for i, g in enumerate(gs):
    k2 = json.load(open(os.path.join(OUT, f"pb_fp_n2_g{g:g}.json")))["kappa_bins"]
    ax.bar(x + (i - 1) * wdt, k2, wdt, label=rf"$g={g:g}$",
           color=["C0", "C2", "C3"][i], alpha=0.85)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([r"soft $[0,2)$", r"mid $[2,4)$", r"hard $[4,8)$"])
ax.set_ylabel(r"$\kappa_2^{\rm bin}$")
ax.legend(fontsize=9)
ax.set_title("(b) response tomography: soft anti-flow, hard flow", fontsize=10)
fig.suptitle("Momentum-space spectral tomography (FP kernel)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_pbins.png"), dpi=200)
fig.savefig(os.path.join(OUT, "fig_pbins.pdf"))
print("wrote fig_pbins")
