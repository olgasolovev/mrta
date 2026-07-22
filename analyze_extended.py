"""!
@file analyze_extended.py
@brief Reproduce every number quoted in para_extended_opacity.tex from
results_fp/d32_extended.csv (plateau values, g=8 signal fractions, collapse
thresholds, asymptotic exponent, residual merging)."""

import csv
import os

import numpy as np

CSV = os.path.join("results_fp", "d32_extended.csv")


def load():
    D = {"diff": [], "mcdiff": []}
    with open(CSV) as f:
        for g, s, d in list(csv.reader(f))[1:]:
            D[s].append((float(g), float(d)))
    return {s: np.array(sorted(v)) for s, v in D.items()}


def main():
    D = load()
    anchors = {"diff": 9.0 / 4.0, "mcdiff": 8.0 / 3.0}
    for s, arr in D.items():
        g, d = arr[:, 0], arr[:, 1]
        r = d - 1.0
        plateau = d[g <= 0.1].mean()
        d8 = np.interp(8.0, g, d)
        print(f"--- {s} (anchor {anchors[s]:.4f})")
        print(f"  plateau (g<=0.1): {plateau:.4f}  "
              f"({plateau/anchors[s]-1:+.2%} from anchor)")
        print(f"  D32(g=8) = {d8:.3f}; surviving signal fraction "
              f"(D32-1)/(anchor-1) = {(d8-1)/(anchors[s]-1):.1%}")
        for thr in (0.5, 0.1, 0.05):
            i = np.searchsorted(-r, -thr)
            gg = np.exp(np.interp(np.log(thr), np.log(r[[i, i - 1]]),
                                  np.log(g[[i, i - 1]])))
            print(f"  |D32-1| < {thr}: g > {gg:.3g}")
        m = g >= g.max() / 12.0
        sl = np.polyfit(np.log(g[m]), np.log(r[m]), 1)[0]
        print(f"  local asymptotic exponent (final decade): {sl:.3f}")
    g1, d1 = D["diff"][:, 0], D["diff"][:, 1]
    g2, d2 = D["mcdiff"][:, 0], D["mcdiff"][:, 1]
    for gq in (30.0, 512.0):
        ratio = (np.interp(gq, g2, d2) - 1) / (np.interp(gq, g1, d1) - 1)
        print(f"residual ratio mcdiff/diff at g={gq:g}: {ratio:.3f}")


if __name__ == "__main__":
    main()
