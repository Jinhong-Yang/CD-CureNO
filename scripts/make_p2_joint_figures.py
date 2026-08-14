"""Create reproducible P2 joint temperature and cure-field figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#24272B"


def _load_case(run_dir: Path, case_id: int) -> dict[str, np.ndarray]:
    archive = np.load(run_dir / "predictions" / "test_predictions.npz")
    matches = np.flatnonzero(archive["case_ids"] == case_id)
    if len(matches) != 1:
        raise ValueError(f"Case ID {case_id} is not unique in {run_dir}.")
    index = int(matches[0])
    return {
        name: archive[name][index]
        for name in (
            "temperature_prediction",
            "temperature_target",
            "alpha_prediction",
            "alpha_target",
        )
    }


def _field_figure(
    truth: np.ndarray,
    factorized: np.ndarray,
    causal: np.ndarray,
    case_id: int,
    quantity: str,
    unit: str,
    field_cmap: str,
    output_stem: Path,
) -> dict:
    if truth.shape != (223, 51):
        raise ValueError(f"Expected [time,position]=(223,51), got {truth.shape}.")
    factorized_error = np.abs(factorized - truth)
    causal_error = np.abs(causal - truth)
    difference = causal_error - factorized_error
    field_min = float(min(truth.min(), factorized.min(), causal.min()))
    field_max = float(max(truth.max(), factorized.max(), causal.max()))
    error_max = float(max(factorized_error.max(), causal_error.max()))
    difference_limit = float(np.abs(difference).max())
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
    field_colorbar_axis = fig.add_subplot(grid[0, 3])
    fig.add_subplot(grid[0, 4]).axis("off")
    error_colorbar_axis = fig.add_subplot(grid[1, 3])
    difference_colorbar_axis = fig.add_subplot(grid[1, 4])

    field_images = []
    for axis, values, title in (
        (axes[0, 0], truth, f"(a) {quantity} truth"),
        (axes[0, 1], factorized, "(b) Factorized prediction"),
        (axes[0, 2], causal, "(c) Causal prediction"),
    ):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=field_cmap,
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
        )
        field_images.append(image)
        axis.set_title(title)

    error_images = []
    for axis, values, title in (
        (axes[1, 0], factorized_error, "(d) Factorized absolute error"),
        (axes[1, 1], causal_error, "(e) Causal absolute error"),
    ):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
            interpolation="nearest",
        )
        error_images.append(image)
        axis.set_title(title)
    difference_image = axes[1, 2].imshow(
        difference.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
        interpolation="nearest",
    )
    axes[1, 2].set_title("(f) Causal - factorized absolute error")

    for row in range(2):
        for column in range(3):
            axis = axes[row, column]
            axis.set_xlabel("Time (min)")
            if column == 0:
                axis.set_ylabel("Through-thickness position (mm)")
            axis.set_xlim(0, 222)
            axis.set_ylim(0, 50)

    label = f"{quantity} ({unit})" if unit else quantity
    fig.colorbar(field_images[0], cax=field_colorbar_axis, label=label)
    error_colorbar = fig.colorbar(error_images[0], cax=error_colorbar_axis)
    difference_colorbar = fig.colorbar(
        difference_image,
        cax=difference_colorbar_axis,
    )
    unit_suffix = f" ({unit})" if unit else ""
    error_colorbar.ax.set_title(
        f"Absolute\nerror{unit_suffix}", fontsize=8, pad=6
    )
    difference_colorbar.ax.set_title(
        f"Error\ndifference{unit_suffix}", fontsize=8, pad=6
    )
    fig.suptitle(
        f"Case {case_id} joint {quantity.lower()} reconstruction (seed 0)",
        y=0.965,
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.915,
        "Negative panel (f) values favor the causal operator",
        ha="center",
        va="top",
        fontsize=9,
        color="#555B63",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "quantity": quantity,
        "unit": unit,
        "field_limits": [field_min, field_max],
        "shared_absolute_error_limit": error_max,
        "absolute_error_difference_limit": difference_limit,
        "png": str(png),
        "pdf": str(pdf),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create fixed-case P2 joint temperature and alpha figures."
    )
    parser.add_argument("--case-id", type=int, default=100)
    parser.add_argument(
        "--factorized-run-id",
        default="p2extended200-factorized-w40d4-seed0-bbdc829",
    )
    parser.add_argument(
        "--causal-run-id",
        default="p2extended200-causal-w40d8-seed0-8ae6702",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/figures")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    factorized_dir = project_root / "outputs" / "runs" / args.factorized_run_id
    causal_dir = project_root / "outputs" / "runs" / args.causal_run_id
    output_dir = (project_root / args.output_dir).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "case_id": args.case_id,
                    "factorized_run_exists": factorized_dir.is_dir(),
                    "causal_run_exists": causal_dir.is_dir(),
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return 0
    factorized = _load_case(factorized_dir, args.case_id)
    causal = _load_case(causal_dir, args.case_id)
    for target in ("temperature_target", "alpha_target"):
        if not np.array_equal(factorized[target], causal[target]):
            raise ValueError(f"Frozen model targets differ for {target}.")
    temperature = _field_figure(
        factorized["temperature_target"],
        factorized["temperature_prediction"],
        causal["temperature_prediction"],
        args.case_id,
        "Temperature",
        "K",
        "viridis",
        output_dir / f"p2_case{args.case_id}_temperature",
    )
    alpha = _field_figure(
        factorized["alpha_target"],
        factorized["alpha_prediction"],
        causal["alpha_prediction"],
        args.case_id,
        "Degree of cure",
        "",
        "cividis",
        output_dir / f"p2_case{args.case_id}_alpha",
    )
    metadata = {
        "case_id": args.case_id,
        "factorized_run_id": args.factorized_run_id,
        "causal_run_id": args.causal_run_id,
        "temperature": temperature,
        "alpha": alpha,
    }
    metadata_path = output_dir / f"p2_case{args.case_id}_joint_fields.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
