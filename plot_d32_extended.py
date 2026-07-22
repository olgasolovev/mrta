#!/usr/bin/env python3
## @file plot_d32_extended.py
## @brief Plot the MRTA D_32 approach to its common-rate limit.

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Response:
    """! @brief Harmonic responses at one spectrum and opacity."""

    spectrum: str
    g: float
    kappa2: float
    kappa3: float
    source: Path


## @brief Parse command-line arguments.
## @return Parsed argument namespace.
def parse_args() -> argparse.Namespace:
    """! @brief Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot D_32 using baseline and extended-opacity records."
    )
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("plots/extended_g"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


## @brief Load full-solver n=2,3 records from explicitly selected roots.
## @param roots Input result directories.
## @return Unique responses indexed by spectrum and g.
def load_responses(roots: list[Path]) -> dict[tuple[str, float], Response]:
    """! @brief Load full-solver n=2,3 response records."""
    responses: dict[tuple[str, float], Response] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Result directory does not exist: {root}")
        for path in sorted(root.rglob("*.json")):
            with path.open(encoding="utf-8") as stream:
                item = json.load(stream)
            if bool(item.get("one_hit", False)):
                continue
            kappa = item.get("kappa", {})
            if "2" not in kappa or "3" not in kappa:
                continue

            response = Response(
                spectrum=str(item["spectrum"]),
                g=float(item["g"]),
                kappa2=float(kappa["2"]),
                kappa3=float(kappa["3"]),
                source=path,
            )
            key = (response.spectrum, response.g)
            previous = responses.get(key)
            if previous is not None and not (
                math.isclose(previous.kappa2, response.kappa2, rel_tol=1.0e-10)
                and math.isclose(previous.kappa3, response.kappa3, rel_tol=1.0e-10)
            ):
                raise ValueError(
                    f"Conflicting response records in {previous.source} and {path}"
                )
            responses[key] = response

    if not responses:
        raise ValueError("No full-solver n=2,3 records were found")
    return responses


## @brief Calculate spectrum-to-flat D_32 values at common g points.
## @param responses Loaded response mapping.
## @return Rows (g, spectrum, D32) sorted by g and spectrum.
def calculate_d32(
    responses: dict[tuple[str, float], Response]
) -> list[tuple[float, str, float]]:
    """! @brief Calculate spectrum-to-flat D_32 values."""
    rows: list[tuple[float, str, float]] = []
    for (spectrum, g), response in responses.items():
        if spectrum not in {"diff", "mcdiff"}:
            continue
        flat = responses.get(("flat", g))
        if flat is None:
            continue
        if abs(response.kappa2) < 1.0e-30 or abs(flat.kappa2) < 1.0e-30:
            continue
        spectrum_ratio = response.kappa3 / response.kappa2
        flat_ratio = flat.kappa3 / flat.kappa2
        if abs(flat_ratio) < 1.0e-30:
            continue
        rows.append((g, spectrum, spectrum_ratio / flat_ratio))

    if not rows:
        raise ValueError("No common flat/diff or flat/mcdiff g points were found")
    return sorted(rows)


## @brief Write the derived double ratios to CSV.
## @param rows Derived D_32 records.
## @param path Destination CSV path.
def write_csv(rows: list[tuple[float, str, float]], path: Path) -> None:
    """! @brief Write the derived double ratios to CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("g", "spectrum", "D32"))
        writer.writerows(rows)


## @brief Plot D_32 and its distance from the common-rate limit.
## @param rows Derived D_32 records.
## @param output_base Output path without extension.
## @param dpi PNG resolution.
def plot_d32(
    rows: list[tuple[float, str, float]], output_base: Path, dpi: int
) -> None:
    """! @brief Plot D_32 and its distance from unity."""
    colors = {"diff": "C0", "mcdiff": "C1"}
    markers = {"diff": "o", "mcdiff": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    ratio_axis, deviation_axis = axes

    for spectrum in ("diff", "mcdiff"):
        series = [(g, value) for g, name, value in rows if name == spectrum]
        if not series:
            continue
        g_values = [item[0] for item in series]
        d32 = [item[1] for item in series]
        deviation = [abs(value - 1.0) for value in d32]
        common = {
            "color": colors[spectrum],
            "marker": markers[spectrum],
            "linewidth": 1.8,
            "markersize": 5.5,
            "label": spectrum,
        }
        ratio_axis.plot(g_values, d32, **common)
        positive = [(g, value) for g, value in zip(g_values, deviation) if value > 0.0]
        deviation_axis.plot(
            [item[0] for item in positive],
            [item[1] for item in positive],
            **common,
        )

    ratio_axis.axhline(1.0, color="0.25", linewidth=1.1,
                       label=r"common-rate limit $D_{32}=1$")
    ratio_axis.axvline(8.0, color="0.55", linestyle=":", linewidth=1.0)
    deviation_axis.axvline(8.0, color="0.55", linestyle=":", linewidth=1.0)

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"code opacity parameter $g$")
        axis.grid(True, which="both", alpha=0.24)
    deviation_axis.set_yscale("log")

    ratio_axis.set_ylabel(r"double ratio $D_{32}$")
    deviation_axis.set_ylabel(r"distance from completion $|D_{32}-1|$")
    ratio_axis.set_title(r"Approach to the common-rate limit")
    deviation_axis.set_title(r"Residual spectrum dependence")
    ratio_axis.legend(frameon=False)

    figure.suptitle(r"Extended-opacity evolution of $D_{32}$")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


## @brief Program entry point.
def main() -> None:
    """! @brief Load results, derive D_32, and write figures and CSV."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    responses = load_responses(args.results)
    rows = calculate_d32(responses)
    csv_path = args.out / "d32_extended.csv"
    output_base = args.out / "d32_extended"
    write_csv(rows, csv_path)
    plot_d32(rows, output_base, args.dpi)
    print(f"wrote {csv_path}")
    print(f"wrote {output_base.with_suffix('.png')}")
    print(f"wrote {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
