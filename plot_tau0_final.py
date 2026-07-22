#!/usr/bin/env python3
## @file plot_tau0_final.py
## @brief Produce publication-oriented matched-opacity tau0 figures.

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Point:
    """! @brief One matched-opacity n=4 response record."""

    tau0: float
    spectrum: str
    one_hit: bool
    kappa4: float
    source: Path


## @brief Parse command-line arguments.
## @return Parsed argument namespace.
def parse_args() -> argparse.Namespace:
    """! @brief Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot the matched-opacity initialization-time dependence."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("plots/tau0_final"))
    parser.add_argument("--reference-tau0", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


## @brief Load unique n=4 response records from a matched-opacity scan.
## @param root Root of the matched-opacity result tree.
## @return Response records sorted by spectrum, mode, and tau0.
def load_points(root: Path) -> list[Point]:
    """! @brief Load unique n=4 response records."""
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")

    records: dict[tuple[float, str, bool], Point] = {}
    for path in sorted(root.rglob("*.json")):
        with path.open(encoding="utf-8") as stream:
            item = json.load(stream)
        if "4" not in item.get("kappa", {}):
            continue

        point = Point(
            tau0=float(item["tau0"]),
            spectrum=str(item["spectrum"]),
            one_hit=bool(item["one_hit"]),
            kappa4=float(item["kappa"]["4"]),
            source=path,
        )
        key = (point.tau0, point.spectrum, point.one_hit)
        previous = records.get(key)
        if previous is not None and not math.isclose(
            previous.kappa4, point.kappa4, rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise ValueError(
                f"Conflicting n=4 records in {previous.source} and {path}"
            )
        records[key] = point

    if not records:
        raise ValueError(f"No n=4 records found below {root}")
    return sorted(
        records.values(), key=lambda point: (point.spectrum, point.one_hit, point.tau0)
    )


## @brief Return one spectrum and solver-mode series.
## @param points All loaded response points.
## @param spectrum Spectrum identifier.
## @param one_hit Select the one-hit result when true.
## @return Series sorted by tau0.
def select_series(
    points: list[Point], spectrum: str, one_hit: bool
) -> list[Point]:
    """! @brief Return one spectrum and solver-mode series."""
    return sorted(
        (
            point
            for point in points
            if point.spectrum == spectrum and point.one_hit == one_hit
        ),
        key=lambda point: point.tau0,
    )


## @brief Find the reference response for normalization.
## @param series Response series.
## @param reference_tau0 Requested reference initialization time.
## @return Reference kappa4 value.
def reference_value(series: list[Point], reference_tau0: float) -> float:
    """! @brief Find the response at the normalization point."""
    for point in series:
        if math.isclose(point.tau0, reference_tau0, rel_tol=0.0, abs_tol=1.0e-12):
            if abs(point.kappa4) < 1.0e-30:
                raise ValueError("Cannot normalize by a vanishing kappa_4")
            return point.kappa4
    raise ValueError(f"No result exists at reference tau0={reference_tau0:g}")


## @brief Build compact legends separating spectrum and solver mode.
## @param figure Figure on which to draw the legends.
## @param colors Spectrum color mapping.
## @param markers Spectrum marker mapping.
def add_compact_legends(figure, colors: dict[str, str], markers: dict[str, str]) -> None:
    """! @brief Add independent spectrum and line-style legends."""
    spectrum_handles = [
        Line2D(
            [0], [0], color=colors[name], marker=markers[name], linestyle="-",
            linewidth=1.8, markersize=6, label=name
        )
        for name in ("flat", "diff", "mcdiff")
    ]
    mode_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.8, label="full"),
        Line2D(
            [0], [0], color="0.25", linestyle="--", linewidth=1.8,
            label="one-hit"
        ),
    ]
    figure.legend(
        handles=spectrum_handles + mode_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
    )


## @brief Plot absolute and normalized matched-opacity responses.
## @param points Loaded n=4 response records.
## @param reference_tau0 Normalization time.
## @param output_base Output path without extension.
## @param dpi PNG resolution.
def plot_final_response(
    points: list[Point], reference_tau0: float, output_base: Path, dpi: int
) -> None:
    """! @brief Plot absolute and normalized matched-opacity responses."""
    colors = {"flat": "C0", "diff": "C1", "mcdiff": "C2"}
    markers = {"flat": "o", "diff": "s", "mcdiff": "^"}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    absolute_axis, normalized_axis = axes

    all_kappa: list[float] = []
    all_normalized: list[float] = []
    for spectrum in ("flat", "diff", "mcdiff"):
        for one_hit in (False, True):
            series = select_series(points, spectrum, one_hit)
            if not series:
                continue
            reference = reference_value(series, reference_tau0)
            tau0 = [point.tau0 for point in series]
            kappa4 = [point.kappa4 for point in series]
            normalized = [value / reference for value in kappa4]
            all_kappa.extend(kappa4)
            all_normalized.extend(normalized)

            style = "--" if one_hit else "-"
            alpha = 0.78 if one_hit else 1.0
            common = {
                "color": colors[spectrum],
                "marker": markers[spectrum],
                "linestyle": style,
                "linewidth": 1.8,
                "markersize": 5.5,
                "alpha": alpha,
            }
            absolute_axis.plot(tau0, kappa4, **common)
            normalized_axis.plot(tau0, normalized, **common)

    absolute_axis.axhline(0.0, color="0.25", linewidth=1.1)
    normalized_axis.axhline(1.0, color="0.25", linewidth=1.1)
    normalized_axis.axvline(
        reference_tau0, color="0.55", linestyle=":", linewidth=1.0
    )

    minimum = min(all_kappa)
    absolute_axis.set_ylim(1.08 * minimum, 0.08 * abs(minimum))
    norm_min, norm_max = min(all_normalized), max(all_normalized)
    norm_pad = max(0.025, 0.12 * (norm_max - norm_min))
    normalized_axis.set_ylim(norm_min - norm_pad, norm_max + norm_pad)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"initialization time $\tau_0/R$")
        axis.grid(True, which="both", alpha=0.24)

    absolute_axis.set_ylabel(r"linear response $\kappa_4$")
    absolute_axis.set_title(r"Absolute response and sign")
    normalized_axis.set_ylabel(
        rf"$\kappa_4(\tau_0)/\kappa_4(\tau_0={reference_tau0:g}R)$"
    )
    normalized_axis.set_title(r"Residual initialization-time dependence")
    add_compact_legends(figure, colors, markers)

    figure.suptitle(r"Matched-opacity initialization-time dependence")
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


## @brief Load the matched coupling table.
## @param path Path to matched_g.csv.
## @return Sorted pairs of tau0 and matched g.
def load_matched_g(path: Path) -> list[tuple[float, float]]:
    """! @brief Load the matched coupling table."""
    if not path.is_file():
        raise ValueError(f"Matched coupling table does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        mapping = [
            (float(row["tau0"]), float(row["matched_g"]))
            for row in csv.DictReader(stream)
        ]
    if not mapping:
        raise ValueError(f"Matched coupling table is empty: {path}")
    return sorted(mapping)


## @brief Plot the supplementary opacity-matching curve.
## @param mapping Matched coupling pairs.
## @param output_base Output path without extension.
## @param dpi PNG resolution.
def plot_matching_curve(
    mapping: list[tuple[float, float]], output_base: Path, dpi: int
) -> None:
    """! @brief Plot the supplementary opacity-matching curve."""
    tau0 = [item[0] for item in mapping]
    coupling = [item[1] for item in mapping]
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.plot(tau0, coupling, marker="o", linewidth=1.8)
    axis.axhline(0.03, color="0.30", linestyle="--", linewidth=1.0,
                 label=r"reference $g=0.03$")
    axis.set_xscale("log")
    axis.set_xlabel(r"initialization time $\tau_0/R$")
    axis.set_ylabel(r"matched coupling $g(\tau_0)$")
    axis.set_title(r"Flat one-hit $\kappa_2$ opacity matching")
    axis.grid(True, which="both", alpha=0.24)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


## @brief Program entry point.
def main() -> None:
    """! @brief Load matched-opacity records and write final figures."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    points = load_points(args.results)
    mapping = load_matched_g(args.results / "matched_g.csv")

    response_base = args.out / "kappa4_tau0_matched"
    matching_base = args.out / "matched_g_tau0_supplement"
    plot_final_response(points, args.reference_tau0, response_base, args.dpi)
    plot_matching_curve(mapping, matching_base, args.dpi)
    for base in (response_base, matching_base):
        print(f"wrote {base.with_suffix('.png')}")
        print(f"wrote {base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
