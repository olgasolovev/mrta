#!/usr/bin/env python3
"""!
@file plot_merged_results.py
@brief Merge harmonic-specific MRTA scan records and plot all response results.

@details
The script searches recursively for JSON output from run_scan.py. Records with
the same spectrum, opacity, and evolution mode are merged, which permits each
harmonic to have been calculated with its own perturbation amplitude.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


RecordKey = tuple[str, float, bool, float]


def record_key(record: dict[str, Any]) -> RecordKey:
    """!
    @brief Return the physical key used to merge scan records.
    @param record Parsed MRTA JSON record.
    @return Spectrum, opacity, one-hit flag, and mixed-spectrum ratio.
    """
    return (
        str(record["spectrum"]),
        float(record["g"]),
        bool(record.get("one_hit", False)),
        float(record.get("b_over_a", 1.0)),
    )


def load_and_merge(root: Path) -> dict[RecordKey, dict[str, Any]]:
    """!
    @brief Recursively load and merge harmonic-specific scan records.
    @param root Root directory containing JSON scan outputs.
    @return Records indexed by their physical scan key.
    @throws ValueError If no valid records are found or conflicting values occur.
    """
    merged: dict[RecordKey, dict[str, Any]] = {}
    sources: dict[tuple[RecordKey, str], Path] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {path}: {error}")
            continue
        if not {"spectrum", "g", "kappa"} <= record.keys():
            continue

        key = record_key(record)
        destination = merged.setdefault(key, {
            "spectrum": key[0], "g": key[1], "one_hit": key[2],
            "b_over_a": key[3], "kappa": {}, "eps": {},
            "linearity_dev": {},
        })
        for field in ("kappa", "eps", "linearity_dev"):
            for harmonic, value in record.get(field, {}).items():
                item_key = (key, harmonic)
                if harmonic in destination[field] and destination[field][harmonic] != value:
                    raise ValueError(
                        f"Conflicting {field}[{harmonic}] in {sources[item_key]} and {path}"
                    )
                destination[field][harmonic] = value
                sources[item_key] = path

    if not merged:
        raise ValueError(f"No MRTA JSON records found below {root}")
    return merged


def label_for(key: RecordKey) -> str:
    """!
    @brief Construct a human-readable curve label.
    @param key Physical scan key.
    @return Legend label for the spectrum and evolution mode.
    """
    spectrum, _, one_hit, b_over_a = key
    label = f"mixed (b/a={b_over_a:g})" if spectrum == "mixed" else spectrum
    return label + (" one-hit" if one_hit else " full")


def grouped_records(
    merged: dict[RecordKey, dict[str, Any]],
) -> dict[tuple[str, bool, float], list[dict[str, Any]]]:
    """!
    @brief Group merged records into opacity-ordered plotting curves.
    @param merged Merged MRTA scan records.
    @return Records grouped by spectrum, mode, and mixed-spectrum ratio.
    """
    groups: dict[tuple[str, bool, float], list[dict[str, Any]]] = defaultdict(list)
    for key, record in merged.items():
        groups[(key[0], key[2], key[3])].append(record)
    for records in groups.values():
        records.sort(key=lambda record: float(record["g"]))
    return dict(groups)


def plot_kappas(groups: dict, harmonics: list[int], output: Path) -> None:
    """!
    @brief Plot response coefficients for every requested harmonic.
    @param groups Opacity-ordered scan curves.
    @param harmonics Harmonic numbers to plot.
    @param output Output figure path.
    """
    figure, axes = plt.subplots(len(harmonics), 1, figsize=(7.4, 3.1 * len(harmonics)),
                                sharex=True, squeeze=False)
    for axis, harmonic in zip(axes[:, 0], harmonics):
        hkey = str(harmonic)
        for group_key, records in groups.items():
            points = [(r["g"], r["kappa"][hkey]) for r in records if hkey in r["kappa"]]
            if points:
                axis.plot(*zip(*points), marker="o",
                          label=label_for((group_key[0], 0.0, group_key[1], group_key[2])))
        axis.set_xscale("log")
        axis.set_ylabel(rf"$\kappa_{{{harmonic}}}$")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    axes[-1, 0].set_xlabel(r"opacity parameter $g$")
    figure.suptitle("MRTA harmonic response")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_raw_ratios(groups: dict, harmonics: list[int], denominator: int,
                    output: Path) -> None:
    """!
    @brief Plot raw harmonic-response ratios.
    @param groups Opacity-ordered scan curves.
    @param harmonics Numerator harmonics.
    @param denominator Denominator harmonic.
    @param output Output figure path.
    """
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    dkey = str(denominator)
    for numerator in harmonics:
        nkey = str(numerator)
        for group_key, records in groups.items():
            points = []
            for record in records:
                kappa = record["kappa"]
                if nkey in kappa and dkey in kappa and kappa[dkey] != 0.0:
                    points.append((record["g"], kappa[nkey] / kappa[dkey]))
            if points:
                name = label_for((group_key[0], 0.0, group_key[1], group_key[2]))
                axis.plot(*zip(*points), marker="o",
                          label=rf"{name}: $\kappa_{{{numerator}}}/\kappa_{{{denominator}}}$")
    axis.set_xscale("log")
    axis.set_xlabel(r"opacity parameter $g$")
    axis.set_ylabel("response ratio")
    axis.set_title("MRTA harmonic-response ratios")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_double_ratios(groups: dict, harmonics: list[int], denominator: int,
                       output: Path) -> None:
    """!
    @brief Plot response double ratios relative to the flat spectrum.
    @param groups Opacity-ordered scan curves.
    @param harmonics Numerator harmonics.
    @param denominator Denominator harmonic.
    @param output Output figure path.
    """
    figure, axes = plt.subplots(len(harmonics), 1, figsize=(7.6, 3.5 * len(harmonics)),
                                sharex=True, squeeze=False)
    dkey = str(denominator)
    targets = {("diff", 3): 9 / 4, ("mcdiff", 3): 8 / 3,
               ("diff", 4): 4.0, ("mcdiff", 4): 5.0}
    modes = sorted({key[1] for key in groups})

    for axis, numerator in zip(axes[:, 0], harmonics):
        nkey = str(numerator)
        for one_hit in modes:
            flat_records = groups.get(("flat", one_hit, 1.0), [])
            flat = {float(r["g"]): r for r in flat_records}
            for (spectrum, mode, ratio), records in groups.items():
                if mode != one_hit or spectrum == "flat":
                    continue
                points = []
                for record in records:
                    reference = flat.get(float(record["g"]))
                    if reference is None:
                        continue
                    kappa, base = record["kappa"], reference["kappa"]
                    if not {nkey, dkey} <= kappa.keys() or not {nkey, dkey} <= base.keys():
                        continue
                    if kappa[dkey] == 0.0 or base[dkey] == 0.0 or base[nkey] == 0.0:
                        continue
                    value = (kappa[nkey] / kappa[dkey]) / (base[nkey] / base[dkey])
                    points.append((record["g"], value))
                if points:
                    style = "--" if one_hit else "-"
                    axis.plot(*zip(*points), marker="o", linestyle=style,
                              label=f"{spectrum} {'one-hit' if one_hit else 'full'}")
                target = targets.get((spectrum, numerator))
                if target is not None:
                    axis.axhline(target, color="gray", linewidth=0.8, alpha=0.35)
        axis.set_ylabel(rf"$D_{{{numerator}{denominator}}}$")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    axes[-1, 0].set_xscale("log")
    axes[-1, 0].set_xlabel(r"opacity parameter $g$")
    figure.suptitle("Double ratios relative to the flat spectrum")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """! @brief Parse arguments, merge scan records, and create all figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    parser.add_argument("--harmonics", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ratio-denominator", type=int, default=2)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    args = parser.parse_args()

    merged = load_and_merge(args.results)
    groups = grouped_records(merged)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = args.format
    plot_kappas(groups, args.harmonics, args.out / f"kappa_vs_g.{suffix}")
    numerators = [n for n in args.harmonics if n != args.ratio_denominator]
    plot_raw_ratios(groups, numerators, args.ratio_denominator,
                    args.out / f"kappa_ratios_vs_g.{suffix}")
    plot_double_ratios(groups, numerators, args.ratio_denominator,
                       args.out / f"double_ratios_vs_g.{suffix}")
    print(f"Wrote plots to {args.out}")


if __name__ == "__main__":
    main()
