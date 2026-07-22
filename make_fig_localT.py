"""!
@file make_fig_localT.py
@brief Figure: what survives the local temperature."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_fp"
def k(f, n): return json.load(open(os.path.join(OUT, f + ".json")))["kappa"][str(n)]

gs = [0.5, 1.0, 2.0, 4.0, 8.0]
flat = {g: json.load(open(os.path.join(OUT, f"flat_g{g:g}.json")))["kappa"] for g in gs}
diff = {g: json.load(open(os.path.join(OUT, f"diff_g{g:g}.json")))["kappa"] for g in gs}
d32_diff = [(diff[g]["3"]/diff[g]["2"])/(flat[g]["3"]/flat[g]["2"]) for g in gs]
gfr = [0.5, 1.0, 2.0]
d32_fr = [(k(f"fp_g{g:g}",3)/k(f"fp_g{g:g}",2))/(flat[g]["3"]/flat[g]["2"]) for g in gfr]
glt = [1.0, 2.0, 4.0]
d32_lt = [(k(f"lT_fp_n3_g{g:g}",3)/k(f"lT_fp_n2_g{g:g}",2))/(flat[g]["3"]/flat[g]["2"]) for g in glt]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
ax.axhline(1.0, color="gray", lw=0.8)
ax.axhline(2.25, color="C0", ls=":", lw=1)
ax.plot(gs, d32_diff, "o-", color="C0", label=r"separable $\gamma_l\propto l^2$")
ax.plot(gfr, d32_fr, "s-", color="C3",
        label=r"FP, frozen bath $T_0$ ($\kappa_3$ crosses at $g\in(2,3)$)")
ax.axvspan(2.0, 3.0, color="C3", alpha=0.10, lw=0)
ax.plot(glt, d32_lt, "D-", color="C2",
        label=r"FP, local $T(x,\tau)$: plateau $\simeq 1.5$ ($b/a\simeq 1/6$)")
ax.axhline(1.5, color="C2", ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel(r"opacity $g$")
ax.set_ylabel(r"$D_{32}$"); ax.set_ylim(0.0, 2.5)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("(a) FP-scalar separation survives local $T$", fontsize=10)

ax = axes[1]
ax.axhline(3.0, color="gray", ls=":", lw=1)
df = np.load(os.path.join(OUT, "hardening_g2_n2.npz"))
dl = np.load(os.path.join(OUT, "hardening_lT_g2_n2.npz"))
ax.plot(df["tau"], df["mean_p"], "-", color="C3", label=r"frozen bath $T_0$")
ax.plot(dl["tau"], dl["mean_p"], "-", color="C2", label=r"local $T(x,\tau)$")
ax.annotate(r"thermal source", (0.4, 3.15), fontsize=9, color="gray")
ax.set_xlabel(r"$\tau/R$")
ax.set_ylabel(r"$\langle p\rangle$ of the $V_2$-carrying profile $[T_0]$")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("(b) momentum-space escape intensifies", fontsize=10)
fig.suptitle(r"Local-temperature check ($g=2$ in panel b)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_localT.png"), dpi=200)
fig.savefig(os.path.join(OUT, "fig_localT.pdf"))
print("wrote fig_localT")
