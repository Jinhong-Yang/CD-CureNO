"""Auditable P3 pretraining and held-out-family evaluation."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cdcureno.data.source_1d import PreparedSource1D, prepare_source_1d
from cdcureno.models.joint_operators import FactorizedFNO, parameter_count
from cdcureno.training.joint import joint_loss_components


@dataclass(frozen=True)
class SourcePretrainConfig:
    data_path: Path
    split_manifest: Path
    output_root: Path
    summary_path: Path
    cases_path: Path
    seed: int = 0
    time_stride: int = 2
    width: int = 24
    depth: int = 3
    modes_time: int = 12
    modes_space: int = 10
    epochs: int = 80
    minimum_epochs: int = 40
    early_stopping_patience: int = 20
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature_weight: float = 1.0
    alpha_weight: float = 0.5
    gradient_weight: float = 0.1
    temperature_l4_weight: float = 0.2
    gradient_clip: float = 1.0
    num_threads: int = 8


ACCEPTANCE = {
    "in_family_temperature_relative_l2_mean_max": 5.0e-3,
    "held_out_temperature_relative_l2_mean_max": 1.0e-2,
    "held_out_each_family_temperature_relative_l2_mean_max": 1.25e-2,
    "held_out_alpha_relative_l2_mean_max": 1.2e-1,
    "held_out_temperature_linf_K_max": 20.0,
    "alpha_bound_violation_count_max": 0,
    "alpha_monotonicity_violation_count_max": 0,
}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _source_loss_components(
    outputs: dict[str, torch.Tensor],
    temperature: torch.Tensor,
    alpha: torch.Tensor,
    material_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    components = joint_loss_components(
        outputs, temperature, alpha, material_mask
    )
    normalized_error = torch.abs(outputs["temperature"] - temperature)
    components["temperature_l4"] = torch.mean(
        torch.mean(normalized_error**4, dim=(1, 2)) ** 0.25
    )
    return components


def _weighted_loss(
    components: dict[str, torch.Tensor], config: SourcePretrainConfig
) -> torch.Tensor:
    return (
        config.temperature_weight * components["temperature"]
        + config.temperature_l4_weight * components["temperature_l4"]
        + config.alpha_weight * components["alpha"]
        + config.gradient_weight * components["gradient"]
    )


def _objective(
    model: torch.nn.Module,
    prepared: PreparedSource1D,
    split: str,
    config: SourcePretrainConfig,
) -> float:
    loader = torch.utils.data.DataLoader(
        prepared.dataset(split),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, temperature, alpha, _, _ in loader:
            components = _source_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            batch = len(inputs)
            total += float(_weighted_loss(components, config)) * batch
            count += batch
    return total / count


def _decode_temperature(
    values: np.ndarray, prepared: PreparedSource1D
) -> np.ndarray:
    metadata = prepared.normalization["field_temperature"]
    return values * (metadata["maximum"] - metadata["minimum"]) + metadata[
        "minimum"
    ]


def _evaluate(
    model: torch.nn.Module,
    prepared: PreparedSource1D,
    split: str,
    config: SourcePretrainConfig,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    loader = torch.utils.data.DataLoader(
        prepared.dataset(split),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    rows: list[dict[str, float | int | str]] = []
    model.eval()
    with torch.no_grad():
        for inputs, target_temperature, target_alpha, case_ids, family_ids in loader:
            outputs = model(inputs)
            prediction_temperature = _decode_temperature(
                outputs["temperature"].numpy(), prepared
            )
            target_temperature_K = _decode_temperature(
                target_temperature.numpy(), prepared
            )
            prediction_alpha = outputs["alpha"].numpy()
            target_alpha_array = target_alpha.numpy()
            masks = inputs[..., 4].numpy()
            for index in range(len(inputs)):
                temperature_error = (
                    prediction_temperature[index] - target_temperature_K[index]
                )
                mask = masks[index] > 0.5
                predicted_alpha_masked = prediction_alpha[index][mask]
                target_alpha_masked = target_alpha_array[index][mask]
                alpha_difference = (
                    predicted_alpha_masked - target_alpha_masked
                )
                rows.append(
                    {
                        "split": split,
                        "case_id": int(case_ids[index]),
                        "family_id": int(family_ids[index]),
                        "temperature_relative_l2": float(
                            np.linalg.norm(temperature_error)
                            / np.linalg.norm(target_temperature_K[index])
                        ),
                        "temperature_mae_K": float(
                            np.mean(np.abs(temperature_error))
                        ),
                        "temperature_linf_K": float(
                            np.max(np.abs(temperature_error))
                        ),
                        "alpha_relative_l2": float(
                            np.linalg.norm(alpha_difference)
                            / max(
                                np.linalg.norm(target_alpha_masked),
                                np.finfo(np.float64).eps,
                            )
                        ),
                        "alpha_mae": float(np.mean(np.abs(alpha_difference))),
                        "alpha_bound_violation_count": int(
                            np.count_nonzero(
                                (prediction_alpha[index] < -1.0e-7)
                                | (prediction_alpha[index] > 1.0 + 1.0e-7)
                            )
                        ),
                        "alpha_monotonicity_violation_count": int(
                            np.count_nonzero(
                                np.diff(prediction_alpha[index], axis=0)
                                < -1.0e-7
                            )
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    summary: dict[str, float | int] = {
        "case_count": int(len(frame)),
        "temperature_relative_l2_mean": float(
            frame["temperature_relative_l2"].mean()
        ),
        "temperature_relative_l2_median": float(
            frame["temperature_relative_l2"].median()
        ),
        "temperature_mae_K_mean": float(frame["temperature_mae_K"].mean()),
        "temperature_linf_K_max": float(frame["temperature_linf_K"].max()),
        "alpha_relative_l2_mean": float(frame["alpha_relative_l2"].mean()),
        "alpha_mae_mean": float(frame["alpha_mae"].mean()),
        "alpha_bound_violation_count": int(
            frame["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            frame["alpha_monotonicity_violation_count"].sum()
        ),
    }
    return summary, frame


def run_source_pretraining(
    config: SourcePretrainConfig,
) -> dict[str, Any]:
    _seed(config.seed)
    torch.set_num_threads(config.num_threads)
    prepared = prepare_source_1d(
        config.data_path,
        config.split_manifest,
        time_stride=config.time_stride,
    )
    model = FactorizedFNO(
        input_channels=len(prepared.channel_names),
        width=config.width,
        depth=config.depth,
        modes_time=config.modes_time,
        modes_space=config.modes_space,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        prepared.dataset("train"),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_root / "best.pt"
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = 0
    bad_epochs = 0
    wall_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_count = 0
        for inputs, temperature, alpha, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            components = _source_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            loss = _weighted_loss(components, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_total += float(loss.detach()) * len(inputs)
            train_count += len(inputs)
        validation = _objective(model, prepared, "validation", config)
        scheduler.step()
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            bad_epochs = 0
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_objective": validation,
                    "normalization": prepared.normalization,
                    "channel_names": prepared.channel_names,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_objective": train_total / train_count,
                "validation_objective": validation,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        if (
            epoch >= config.minimum_epochs
            and bad_epochs >= config.early_stopping_patience
        ):
            break
    wall_seconds = time.perf_counter() - wall_started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    split_summaries: dict[str, dict[str, float | int]] = {}
    frames = []
    for split in (
        "in_family_test",
        *sorted(prepared.held_out_families),
    ):
        split_summary, frame = _evaluate(model, prepared, split, config)
        split_summaries[split] = split_summary
        frames.append(frame)
    cases = pd.concat(frames, ignore_index=True)
    heldout_names = sorted(prepared.held_out_families)
    heldout = cases[cases["split"].isin(heldout_names)]
    heldout_summary = {
        "case_count": int(len(heldout)),
        "temperature_relative_l2_mean": float(
            heldout["temperature_relative_l2"].mean()
        ),
        "temperature_linf_K_max": float(heldout["temperature_linf_K"].max()),
        "alpha_relative_l2_mean": float(heldout["alpha_relative_l2"].mean()),
        "alpha_bound_violation_count": int(
            heldout["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            heldout["alpha_monotonicity_violation_count"].sum()
        ),
    }
    checks = {
        "in_family_temperature": (
            split_summaries["in_family_test"][
                "temperature_relative_l2_mean"
            ]
            <= ACCEPTANCE["in_family_temperature_relative_l2_mean_max"]
        ),
        "held_out_temperature": (
            heldout_summary["temperature_relative_l2_mean"]
            <= ACCEPTANCE["held_out_temperature_relative_l2_mean_max"]
        ),
        "held_out_each_family_temperature": all(
            split_summaries[name]["temperature_relative_l2_mean"]
            <= ACCEPTANCE[
                "held_out_each_family_temperature_relative_l2_mean_max"
            ]
            for name in heldout_names
        ),
        "held_out_alpha": (
            heldout_summary["alpha_relative_l2_mean"]
            <= ACCEPTANCE["held_out_alpha_relative_l2_mean_max"]
        ),
        "held_out_linf": (
            heldout_summary["temperature_linf_K_max"]
            <= ACCEPTANCE["held_out_temperature_linf_K_max"]
        ),
        "alpha_bounds": (
            heldout_summary["alpha_bound_violation_count"]
            <= ACCEPTANCE["alpha_bound_violation_count_max"]
        ),
        "alpha_monotonicity": (
            heldout_summary["alpha_monotonicity_violation_count"]
            <= ACCEPTANCE["alpha_monotonicity_violation_count_max"]
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P3",
        "experiment": "source_factorized_cdcureno",
        "seed": config.seed,
        "model": {
            "family": "factorized_fno",
            "input_channels": len(prepared.channel_names),
            "width": config.width,
            "depth": config.depth,
            "modes_time": config.modes_time,
            "modes_space": config.modes_space,
            "parameter_count": parameter_count(model),
        },
        "training": {
            "selected_epoch": best_epoch,
            "executed_epochs": len(history),
            "best_validation_objective": best_validation,
            "wall_seconds": wall_seconds,
            "summed_epoch_seconds": float(
                sum(float(row["epoch_seconds"]) for row in history)
            ),
            "train_case_count": len(prepared.splits["train"]),
            "validation_case_count": len(prepared.splits["validation"]),
            "held_out_labels_used": False,
        },
        "normalization": prepared.normalization,
        "split_metrics": split_summaries,
        "held_out_combined": heldout_summary,
        "acceptance_thresholds_prespecified": ACCEPTANCE,
        "acceptance_checks": checks,
        "passed": bool(all(checks.values())),
    }
    pd.DataFrame(history).to_csv(
        config.output_root / "history.csv", index=False
    )
    config.cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(config.cases_path, index=False)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
