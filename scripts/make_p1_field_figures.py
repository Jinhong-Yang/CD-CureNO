"""Create reproducible P1 manuscript figures from frozen field artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#24272B"


def load_field(project_root: Path, summary_path: Path) -> tuple[dict, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    archive = np.load(project_root / summary["field_artifact"])
    return summary, {key: archive[key] for key in archive.files}


def make_case_figure(
    project_root: Path,
    exact_summary_path: Path,
    corrected_summary_path: Path,
    case_id: int,
    output_stem: Path,
) -> dict:
    exact_summary, exact = load_field(project_root, exact_summary_path)
    corrected_summary, corrected = load_field(project_root, corrected_summary_path)
    exact_matches = np.flatnonzero(exact["case_ids"] == case_id)
    corrected_matches = np.flatnonzero(corrected["case_ids"] == case_id)
    if len(exact_matches) != 1 or len(corrected_matches) != 1:
        raise ValueError(f"Case ID {case_id} is not unique in both field artifacts.")
    exact_index = int(exact_matches[0])
    corrected_index = int(corrected_matches[0])
    truth_exact = exact["target"][exact_index]
    truth_corrected = corrected["target"][corrected_index]
    if not np.array_equal(truth_exact, truth_corrected):
        raise ValueError("Exact and corrected truth fields differ for the fixed case.")
    exact_prediction = exact["prediction"][exact_index]
    corrected_prediction = corrected["prediction"][corrected_index]
    exact_error = exact_prediction - truth_exact
    corrected_error = corrected_prediction - truth_exact
    absolute_error_difference = np.abs(corrected_error) - np.abs(exact_error)

    temperature_min = float(
        min(truth_exact.min(), exact_prediction.min(), corrected_prediction.min())
    )
    temperature_max = float(
        max(truth_exact.max(), exact_prediction.max(), corrected_prediction.max())
    )
    error_limit = float(max(np.abs(exact_error).max(), np.abs(corrected_error).max()))
    difference_limit = float(np.abs(absolute_error_difference).max())
    extent = [0.0, 222.0, 0.0, 50.0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titleweight": "semibold",
        }
    )
    fig = plt.figure(figsize=(14.0, 7.5))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 0.05, 0.05],
        left=0.065,
        right=0.955,
        bottom=0.09,
        top=0.84,
        wspace=0.28,
        hspace=0.30,
    )
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            axes[row, column] = fig.add_subplot(grid[row, column])
    temperature_colorbar_axis = fig.add_subplot(grid[0, 3])
    unused_top_axis = fig.add_subplot(grid[0, 4])
    unused_top_axis.axis("off")
    error_colorbar_axis = fig.add_subplot(grid[1, 3])
    difference_colorbar_axis = fig.add_subplot(grid[1, 4])
    temperature_panels = [
        (axes[0, 0], truth_exact, "(a) FEM truth"),
        (axes[0, 1], exact_prediction, "(b) Exact prediction"),
        (axes[0, 2], corrected_prediction, "(c) Corrected prediction"),
    ]
    temperature_images = []
    for axis, values, title in temperature_panels:
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=temperature_min,
            vmax=temperature_max,
            interpolation="nearest",
        )
        temperature_images.append(image)
        axis.set_title(title)

    error_panels = [
        (axes[1, 0], exact_error, "(d) Exact error"),
        (axes[1, 1], corrected_error, "(e) Corrected error"),
    ]
    error_images = []
    for axis, values, title in error_panels:
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            interpolation="nearest",
        )
        error_images.append(image)
        axis.set_title(title)

    difference_image = axes[1, 2].imshow(
        absolute_error_difference,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
        interpolation="nearest",
    )
    axes[1, 2].set_title("(f) Corrected − exact absolute error")

    for row in range(2):
        for column in range(3):
            axis = axes[row, column]
            axis.set_xlabel("Time (min)")
            if column == 0:
                axis.set_ylabel("Through-thickness position (mm)")
            axis.set_xlim(0, 222)
            axis.set_ylim(0, 50)
            axis.grid(False)

    fig.colorbar(
        temperature_images[0],
        cax=temperature_colorbar_axis,
        label="Temperature (K)",
    )
    fig.colorbar(
        error_images[0],
        cax=error_colorbar_axis,
        label="Prediction error (K)",
    )
    fig.colorbar(
        difference_image,
        cax=difference_colorbar_axis,
        label="Absolute-error difference (K)",
    )
    fig.suptitle(
        f"Case {case_id} temperature-field reconstruction (seed 1)",
        y=0.965,
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.915,
        "51 independent position-wise models; negative panel (f) values favor corrected",
        ha="center",
        va="top",
        fontsize=9,
        color="#555B63",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "case_id": case_id,
        "exact_summary": str(exact_summary_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "corrected_summary": str(
            corrected_summary_path.relative_to(project_root)
        ).replace("\\", "/"),
        "exact_seed": exact_summary["seed"],
        "corrected_seed": corrected_summary["seed"],
        "temperature_limits_kelvin": [temperature_min, temperature_max],
        "shared_error_limit_kelvin": error_limit,
        "absolute_error_difference_limit_kelvin": difference_limit,
        "png": str(png_path.relative_to(project_root)).replace("\\", "/"),
        "pdf": str(pdf_path.relative_to(project_root)).replace("\\", "/"),
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the frozen P1 temperature-field comparison figure."
    )
    parser.add_argument("--case-id", type=int, default=100)
    parser.add_argument(
        "--exact-summary",
        type=Path,
        default=Path("outputs/tables/p1_exact_field_seed1_summary.json"),
    )
    parser.add_argument(
        "--corrected-summary",
        type=Path,
        default=Path("outputs/tables/p1_corrected_field_seed1_summary.json"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("outputs/figures/p1_case100_temperature_field"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    exact = (project_root / args.exact_summary).resolve()
    corrected = (project_root / args.corrected_summary).resolve()
    output_stem = (project_root / args.output_stem).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "case_id": args.case_id,
                    "exact_summary": str(exact),
                    "exact_exists": exact.is_file(),
                    "corrected_summary": str(corrected),
                    "corrected_exists": corrected.is_file(),
                    "output_stem": str(output_stem),
                },
                indent=2,
            )
        )
        return 0
    metadata = make_case_figure(
        project_root, exact, corrected, args.case_id, output_stem
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
