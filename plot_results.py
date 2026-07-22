#!/usr/bin/env python3
"""!
@file plot_results.py
@brief Plot MRTA scan results written by run_scan.py.

@details
The script collects JSON records from a results directory and creates one
figure for the response coefficients kappa_n and another for ratios such as
kappa_3/kappa_2. Curves are grouped by collision-kernel spectrum.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    """!
    @brief Load valid scan records from a directory.
    @param results_dir Directory containing JSON files from run_scan.py.
    @return Records sorted by spectrum and opacity parameter g.
    @throws ValueError If no valid scan records are found.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            if "spectrum" not in record or "g" not in record or "kappa" not in record:
                print(f"Skipping {path}: not an MRTA scan record")
                continue
            record["_source"] = str(path)
            records.append(record)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {path}: {error}")

    if not records:
        raise ValueError(f"No valid scan JSON files found in {results_dir}")
    return sorted(records, key=lambda item: (curve_label(item), float(item["g"])))


def curve_label(record: dict[str, Any]) -> str:
    """!
    @brief Construct the legend label for a scan record.
    @param record MRTA scan record.
    @return Spectrum label, including b/a for a mixed spectrum.
    """
    spectrum = str(record["spectrum"])
    if spectrum == "mixed":
        return f"mixed (b/a={record.get('b_over_a', 1.0):g})"
    return spectrum


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """!
    @brief Group records into curves with the same collision kernel.
    @param records Valid MRTA scan records.
    @return Mapping from curve labels to opacity-ordered records.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[curve_label(record)].append(record)
    for curve in grouped.values():
        curve.sort(key=lambda item: float(item["g"]))
    return dict(grouped)


def available_harmonics(records: list[dict[str, Any]]) -> list[int]:
    """!
    @brief Find harmonics available in at least one scan record.
    @param records Valid MRTA scan records.
    @return Sorted harmonic numbers.
    """
    return sorted({int(n) for record in records for n in record["kappa"]})


def plot_kappa(
    grouped: dict[str, list[dict[str, Any]]], harmonics: list[int], output: Path
) -> None:
    """!
    @brief Plot response coefficients as functions of opacity.
    @param grouped Scan records grouped by collision kernel.
    @param harmonics Harmonic numbers to plot.
    @param output Destination image path.
    """
    figure, axes = plt.subplots(
        len(harmonics), 1, figsize=(7.2, 3.2 * len(harmonics)), squeeze=False,
        sharex=True,
    )
    for axis, harmonic in zip(axes[:, 0], harmonics):
        key = str(harmonic)
        for label, records in grouped.items():
            points = [(float(r["g"]), float(r["kappa"][key])) for r in records
                      if key in r["kappa"]]
            if points:
                g_values, kappa_values = zip(*points)
                axis.plot(g_values, kappa_values, marker="o", label=label)
        axis.set_xscale("log")
        axis.set_ylabel(rf"$\kappa_{{{harmonic}}}$")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    axes[-1, 0].set_xlabel(r"opacity parameter $g$")
    figure.suptitle("MRTA harmonic response")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_ratios(
    grouped: dict[str, list[dict[str, Any]]], numerator_harmonics: list[int],
    denominator: int, output: Path,
) -> None:
    """!
    @brief Plot harmonic-response ratios as functions of opacity.
    @param grouped Scan records grouped by collision kernel.
    @param numerator_harmonics Harmonics used in ratio numerators.
    @param denominator Harmonic used in all ratio denominators.
    @param output Destination image path.
    """
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    denominator_key = str(denominator)
    line_styles = ["-", "--", ":", "-."]
    for index, numerator in enumerate(numerator_harmonics):
        numerator_key = str(numerator)
        for label, records in grouped.items():
            points = []
            for record in records:
                kappa = record["kappa"]
                if numerator_key not in kappa or denominator_key not in kappa:
                    continue
                denominator_value = float(kappa[denominator_key])
                if denominator_value != 0.0:
                    points.append((float(record["g"]),
                                   float(kappa[numerator_key]) / denominator_value))
            if points:
                g_values, ratio_values = zip(*points)
                axis.plot(
                    g_values, ratio_values, marker="o",
                    linestyle=line_styles[index % len(line_styles)],
                    label=rf"{label}: $\kappa_{{{numerator}}}/\kappa_{{{denominator}}}$",
                )
    axis.set_xscale("log")
    axis.set_xlabel(r"opacity parameter $g$")
    axis.set_ylabel("response ratio")
    axis.set_title("MRTA harmonic-response ratios")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """!
    @brief Parse command-line arguments.
    @return Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="directory containing scan JSON files")
    parser.add_argument("--out", type=Path, default=Path("plots"),
                        help="directory in which figures are written")
    parser.add_argument("--harmonics", type=int, nargs="+", default=None,
                        help="harmonics to plot; default: all available")
    parser.add_argument("--ratio-denominator", type=int, default=2,
                        help="denominator harmonic for the ratio plot")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png",
                        help="output figure format")
    return parser.parse_args()


def main() -> None:
    """! @brief Load scan data and generate the requested figures."""
    args = parse_arguments()
    records = load_records(args.results)
    grouped = group_records(records)
    harmonics = args.harmonics or available_harmonics(records)
    args.out.mkdir(parents=True, exist_ok=True)

    kappa_path = args.out / f"kappa_vs_g.{args.format}"
    plot_kappa(grouped, harmonics, kappa_path)

    numerators = [n for n in harmonics if n != args.ratio_denominator]
    if numerators:
        ratio_path = args.out / f"kappa_ratios_vs_g.{args.format}"
        plot_ratios(grouped, numerators, args.ratio_denominator, ratio_path)
        print(f"Wrote {ratio_path}")
    print(f"Wrote {kappa_path}")


if __name__ == "__main__":
    main()
