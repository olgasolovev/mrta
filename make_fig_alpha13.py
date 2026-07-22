"""!
@file make_fig_alpha13.py
@brief Figure: signed responses at the conformal rate exponent alpha = 1/3."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_fp"
def jl(t): return json.load(open(os.path.join(OUT, t + ".json")))

g3 = [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
k3 = [jl(f"a13_fp_n3_g{g:g}")["kappa"]["3"] for g in g3]
ok = np.array(g3) <= 2.5
g1a = [0.5, 1.0, 2.0, 4.0, 8.0]
k3a1 = [jl(f"fp_g{g:g}")["kappa"]["3"] for g in g1a]
gf = [0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0]
k3f = [jl(f"a13_flat_g{g:g}")["kappa"]["3"] for g in gf]
k3d = [jl(f"a13_diff_g{g:g}")["kappa"]["3"] for g in gf]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
ax.axhline(0, color="gray", lw=0.8)
ax.axvspan(0.1, 0.2, color="C3", alpha=0.15, lw=0)
ax.axvspan(2.0, 4.0, color="C0", alpha=0.10, lw=0)
ax.plot(gf, np.array(k3f)*1e3, "^-", color="0.55", label=r"flat, $\alpha=1/3$")
ax.plot(gf, np.array(k3d)*1e3, "v-", color="0.75", label=r"diff, $\alpha=1/3$")
ax.plot(g1a, np.array(k3a1)*1e3, "s--", color="C0",
        label=r"FP, $\alpha=1$  ($g_3^*\in(2,4)$)")
g3arr, k3arr = np.array(g3), np.array(k3)*1e3
ax.plot(g3arr[ok], k3arr[ok], "o-", color="C3",
        label=r"FP, $\alpha=1/3$  ($g_3^*\in(0.1,0.2)$)")
ax.plot(g3arr[~ok], k3arr[~ok], "o", mfc="none", color="C3",
        label=r"FP, $\alpha=1/3$ (mask-limited)")
ax.set_xscale("log"); ax.set_xlabel(r"opacity $g$")
ax.set_ylabel(r"$\kappa_3 \times 10^3$")
ax.set_ylim(-25, 40); ax.legend(fontsize=8, loc="upper left")
ax.set_title(r"(a) signed triangular response", fontsize=10)

ax = axes[1]
ax.axhline(0, color="gray", lw=0.8)
g2 = [0.5, 2.0]
k2 = [jl(f"a13_fp_n2_g{g:g}")["kappa"]["2"] for g in g2]
k2f = [jl(f"a13_flat_g{g:g}")["kappa"]["2"] for g in gf]
ax.plot(gf, np.array(k2f)*1e3, "^-", color="0.55", label=r"flat, $\alpha=1/3$")
gg = np.linspace(0.02, 0.3, 50)
ax.plot(gg, 7.05*gg, ":", color="C3", label=r"FP one-hit slope ($+$)")
ax.plot(g2, np.array(k2)*1e3, "o", color="C3", ms=8,
        label=r"FP, $\alpha=1/3$ (full)")
ax.set_xscale("log"); ax.set_xlabel(r"opacity $g$")
ax.set_ylabel(r"$\kappa_2 \times 10^3$")
ax.set_ylim(-3, 16); ax.legend(fontsize=8, loc="upper left")
ax.set_title(r"(b) even the elliptic response changes sign", fontsize=10)
fig.suptitle(r"Conformal rate $r \propto e^{1/3}$: signed responses of the "
             r"momentum-entangled kernel", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_alpha13.png"), dpi=200)
fig.savefig(os.path.join(OUT, "fig_alpha13.pdf"))
print("wrote fig_alpha13")
