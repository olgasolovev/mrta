#!/usr/bin/env python3
## @file plot_tau0_results.py
## @brief Plot fixed-g and matched-opacity initialization-time scans.

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Point:
    """! @brief One n=4 response point loaded from an MRTA JSON record."""

    tau0: float
    spectrum: str
    one_hit: bool
    kappa4: float
    source: Path


## @brief Parse command-line options.
## @return Populated argument namespace.
def parse_args() -> argparse.Namespace:
    """! @brief Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Plot MRTA kappa_4 dependence on the initialization time."
    )
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("plots/tau0"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


## @brief Load n=4 records and reject ambiguous duplicates.
## @param root Root directory of one tau0 scan.
## @return Sorted response points.
def load_points(root: Path) -> list[Point]:
    """! @brief Load n=4 records and reject ambiguous duplicates."""
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")

    points: dict[tuple[float, str, bool], Point] = {}
    for path in sorted(root.rglob("*.json")):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)

        kappa = record.get("kappa", {})
        if "4" not in kappa:
            continue

        point = Point(
            tau0=float(record["tau0"]),
            spectrum=str(record["spectrum"]),
            one_hit=bool(record["one_hit"]),
            kappa4=float(kappa["4"]),
            source=path,
        )
        key = (point.tau0, point.spectrum, point.one_hit)
        previous = points.get(key)
        if previous is not None and not math.isclose(
            previous.kappa4, point.kappa4, rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise ValueError(
                "Conflicting kappa[4] records for "
                f"tau0={point.tau0}, spectrum={point.spectrum}, "
                f"one_hit={point.one_hit}: {previous.source} and {path}"
            )
        points[key] = point

    if not points:
        raise ValueError(f"No n=4 MRTA records found below {root}")
    return sorted(points.values(), key=lambda item: (item.spectrum, item.one_hit, item.tau0))


## @brief Draw one fixed-g or matched-opacity response panel.
## @param axis Matplotlib axis to populate.
## @param points Response records for the panel.
## @param title Panel title.
def plot_response_panel(axis, points: list[Point], title: str) -> None:
    """! @brief Draw one fixed-g or matched-opacity response panel."""
    colors = {"flat": "C0", "diff": "C1", "mcdiff": "C2"}
    markers = {"flat": "o", "diff": "s", "mcdiff": "^"}

    for spectrum in ("flat", "diff", "mcdiff"):
        for one_hit in (False, True):
            series = [
                point
                for point in points
                if point.spectrum == spectrum and point.one_hit == one_hit
            ]
            if not series:
                continue
            axis.plot(
                [point.tau0 for point in series],
                [point.kappa4 for point in series],
                color=colors[spectrum],
                marker=markers[spectrum],
                linestyle="--" if one_hit else "-",
                linewidth=1.8,
                markersize=5,
                label=f"{spectrum} {'one-hit' if one_hit else 'full'}",
            )

    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel(r"initialization time $\tau_0/R$")
    axis.set_ylabel(r"$\kappa_4$")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)


## @brief Plot fixed-g and matched-opacity kappa_4 side by side.
## @param fixed Fixed-g response records.
## @param matched Matched-opacity response records.
## @param output Destination image path.
## @param dpi Raster resolution.
def plot_kappa4_comparison(
    fixed: list[Point], matched: list[Point], output: Path, dpi: int
) -> None:
    """! @brief Plot fixed-g and matched-opacity kappa_4 side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    plot_response_panel(axes[0], fixed, r"Fixed code coupling $g=0.03$")
    plot_response_panel(axes[1], matched, r"Matched one-hit $\kappa_2$")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(r"Initialization-time dependence of the $n=4$ response")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


## @brief Load the matched-g calibration table.
## @param path CSV file written by 09_tau0_matched_opacity.sh.
## @return Sorted (tau0, matched_g) pairs.
def load_matched_g(path: Path) -> list[tuple[float, float]]:
    """! @brief Load the matched-g calibration table."""
    if not path.is_file():
        raise ValueError(f"Matched-g table does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            (float(row["tau0"]), float(row["matched_g"]))
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError(f"Matched-g table is empty: {path}")
    return sorted(rows)


## @brief Plot the coupling required to hold one-hit kappa_2 fixed.
## @param mapping Sorted (tau0, matched_g) pairs.
## @param output Destination image path.
## @param dpi Raster resolution.
def plot_matched_g(
    mapping: list[tuple[float, float]], output: Path, dpi: int
) -> None:
    """! @brief Plot the coupling required to hold one-hit kappa_2 fixed."""
    tau0 = [item[0] for item in mapping]
    matched_g = [item[1] for item in mapping]
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.plot(tau0, matched_g, marker="o", linewidth=1.8)
    axis.axhline(0.03, color="0.35", linestyle="--", linewidth=1.0,
                 label=r"reference $g=0.03$")
    axis.set_xscale("log")
    axis.set_xlabel(r"initialization time $\tau_0/R$")
    axis.set_ylabel(r"matched code coupling $g(\tau_0)$")
    axis.set_title(r"Opacity matching through flat one-hit $\kappa_2$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


## @brief Program entry point.
def main() -> None:
    """! @brief Load scan results and write the tau0 figures."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fixed = load_points(args.fixed_root)
    matched = load_points(args.matched_root)
    mapping = load_matched_g(args.matched_root / "matched_g.csv")

    comparison_path = args.out / "kappa4_vs_tau0.png"
    mapping_path = args.out / "matched_g_vs_tau0.png"
    plot_kappa4_comparison(fixed, matched, comparison_path, args.dpi)
    plot_matched_g(mapping, mapping_path, args.dpi)
    print(f"wrote {comparison_path}")
    print(f"wrote {mapping_path}")


if __name__ == "__main__":
    main()
