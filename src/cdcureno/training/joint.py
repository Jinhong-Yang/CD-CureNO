"""Auditable P2 training for joint Case1 temperature/cure operators."""

from __future__ import annotations

import csv
import ctypes
import json
import logging
import os
import random
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from cdcureno.data.joint_case1 import PreparedJointCase1, prepare_joint_case1
from cdcureno.legacy.audit import sha256_file
from cdcureno.models.joint_operators import (
    JointOperatorName,
    build_joint_operator,
    parameter_count,
)


JointExperiment = Literal[
    "source_joint_noncausal_fno2d",
    "source_joint_factorized_fno",
    "source_joint_causal",
]
EXPERIMENT_MODELS: dict[JointExperiment, JointOperatorName] = {
    "source_joint_noncausal_fno2d": "noncausal_fno2d",
    "source_joint_factorized_fno": "factorized_fno",
    "source_joint_causal": "causal_factorized",
}


@dataclass(frozen=True)
class JointTrainConfig:
    experiment: JointExperiment
    data_path: Path
    split_manifest: Path
    output_root: Path
    project_root: Path
    epochs: int = 300
    batch_size: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature_weight: float = 1.0
    alpha_weight: float = 0.5
    gradient_weight: float = 0.1
    gradient_clip: float = 1.0
    width: int | None = None
    depth: int | None = None
    modes_time: int = 16
    modes_space: int = 12
    seed: int = 0
    device: str = "cpu"
    num_threads: int = 4
    early_stopping_patience: int = 60
    minimum_epochs: int = 50
    run_id: str | None = None
    resume: bool = False
    register_result: bool = True

    def validated(self) -> "JointTrainConfig":
        if self.experiment not in EXPERIMENT_MODELS:
            raise ValueError(f"Unsupported joint experiment: {self.experiment}")
        if min(self.epochs, self.batch_size, self.num_threads) < 1:
            raise ValueError("epochs, batch_size, and num_threads must be positive.")
        if self.width is not None and self.width < 1:
            raise ValueError("width must be positive when specified.")
        if self.depth is not None and self.depth < 1:
            raise ValueError("depth must be positive when specified.")
        if min(self.modes_time, self.modes_space) < 1:
            raise ValueError("Fourier mode counts must be positive.")
        if min(
            self.temperature_weight,
            self.alpha_weight,
            self.gradient_weight,
        ) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        if self.temperature_weight + self.alpha_weight + self.gradient_weight <= 0:
            raise ValueError("At least one loss weight must be positive.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")
        if self.early_stopping_patience < 1 or self.minimum_epochs < 1:
            raise ValueError("Early-stopping settings must be positive.")
        return self


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _config_payload(config: JointTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("data_path", "split_manifest", "output_root", "project_root"):
        payload[key] = str(payload[key])
    return payload


def _scientific_config(config: JointTrainConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    for key in ("resume", "register_result"):
        payload.pop(key, None)
    return payload


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def make_joint_run_id(config: JointTrainConfig, git_sha: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    return (
        f"{timestamp}__P2__{config.experiment}__Case1joint__n50__"
        f"seed{config.seed}__{git_sha[:7]}"
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _peak_process_rss_bytes() -> int | None:
    """Return OS peak working set without adding a runtime dependency."""

    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        ok = get_process_memory_info(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _relative_l2_per_case(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    difference = (prediction - target).reshape(len(target), -1)
    flattened_target = target.reshape(len(target), -1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.clamp_min(
        torch.linalg.vector_norm(flattened_target, dim=1), 1e-12
    )


def joint_loss_components(
    outputs: dict[str, torch.Tensor],
    target_temperature: torch.Tensor,
    target_alpha: torch.Tensor,
    material_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Dimensionless complete-field losses, averaged over complete cases."""

    temperature = _relative_l2_per_case(
        outputs["temperature"], target_temperature
    ).mean()
    prediction_alpha = outputs["alpha"] * material_mask
    target_alpha_masked = target_alpha * material_mask
    alpha = _relative_l2_per_case(prediction_alpha, target_alpha_masked).mean()
    prediction_gradient = (
        outputs["temperature"][:, :, 1:] - outputs["temperature"][:, :, :-1]
    )
    target_gradient = target_temperature[:, :, 1:] - target_temperature[:, :, :-1]
    gradient = _relative_l2_per_case(prediction_gradient, target_gradient).mean()
    return {"temperature": temperature, "alpha": alpha, "gradient": gradient}


def _weighted_loss(
    components: dict[str, torch.Tensor], config: JointTrainConfig
) -> torch.Tensor:
    return (
        config.temperature_weight * components["temperature"]
        + config.alpha_weight * components["alpha"]
        + config.gradient_weight * components["gradient"]
    )


def _decode_temperature(
    normalized: torch.Tensor, prepared: PreparedJointCase1
) -> torch.Tensor:
    metadata = prepared.normalization["field_temperature"]
    minimum = metadata["minimum"]
    maximum = metadata["maximum"]
    return normalized * (maximum - minimum) + minimum


def compute_joint_field_metrics(
    case_ids: np.ndarray,
    temperature_prediction: np.ndarray,
    temperature_target: np.ndarray,
    alpha_prediction: np.ndarray,
    alpha_target: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Compute P1-comparable field metrics at the complete-case level."""

    expected = temperature_target.shape
    for name, values in (
        ("temperature_prediction", temperature_prediction),
        ("alpha_prediction", alpha_prediction),
        ("alpha_target", alpha_target),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} does not match {expected}.")
    if expected[0] != len(case_ids) or expected[2] != 51:
        raise ValueError("Expected case-aligned canonical [case,time,51] fields.")

    temperature_error = temperature_prediction - temperature_target
    composite_prediction = temperature_prediction[:, :, 21:]
    composite_target = temperature_target[:, :, 21:]
    tool_prediction = temperature_prediction[:, :, :21]
    tool_target = temperature_target[:, :, :21]
    alpha_error = alpha_prediction[:, :, 21:] - alpha_target[:, :, 21:]
    prediction_gradient = np.diff(temperature_prediction, axis=2)
    target_gradient = np.diff(temperature_target, axis=2)

    def relative(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        difference = (prediction - target).reshape(len(target), -1)
        flattened_target = target.reshape(len(target), -1)
        return np.linalg.norm(difference, axis=1) / np.maximum(
            np.linalg.norm(flattened_target, axis=1), 1e-12
        )

    flattened_error = temperature_error.reshape(len(case_ids), -1)
    flattened_alpha_error = alpha_error.reshape(len(case_ids), -1)
    frame = pd.DataFrame(
        {
            "case_id": case_ids,
            "field_relative_l2": relative(
                temperature_prediction, temperature_target
            ),
            "composite_relative_l2": relative(
                composite_prediction, composite_target
            ),
            "tool_relative_l2": relative(tool_prediction, tool_target),
            "field_mae": np.mean(np.abs(flattened_error), axis=1),
            "field_rmse": np.sqrt(np.mean(flattened_error**2, axis=1)),
            "field_linf": np.max(np.abs(flattened_error), axis=1),
            "peak_composite_temperature_error": np.max(
                composite_prediction, axis=(1, 2)
            )
            - np.max(composite_target, axis=(1, 2)),
            "alpha_relative_l2": relative(
                alpha_prediction[:, :, 21:], alpha_target[:, :, 21:]
            ),
            "alpha_mae": np.mean(np.abs(flattened_alpha_error), axis=1),
            "alpha_rmse": np.sqrt(np.mean(flattened_alpha_error**2, axis=1)),
            "alpha_linf": np.max(np.abs(flattened_alpha_error), axis=1),
            "spatial_gradient_relative_l2": relative(
                prediction_gradient, target_gradient
            ),
            "spatial_gradient_mae": np.mean(
                np.abs(prediction_gradient - target_gradient), axis=(1, 2)
            ),
            "alpha_monotonic_violation_count": np.sum(
                np.diff(alpha_prediction[:, :, 21:], axis=1) < -1e-7,
                axis=(1, 2),
            ),
            "alpha_bound_violation_count": np.sum(
                (alpha_prediction < -1e-7) | (alpha_prediction > 1.0 + 1e-7),
                axis=(1, 2),
            ),
        }
    )
    summary: dict[str, float | int] = {
        "case_count": int(len(frame)),
        "field_relative_l2_mean": float(frame["field_relative_l2"].mean()),
        "field_relative_l2_median": float(frame["field_relative_l2"].median()),
        "composite_relative_l2_mean": float(
            frame["composite_relative_l2"].mean()
        ),
        "tool_relative_l2_mean": float(frame["tool_relative_l2"].mean()),
        "field_mae_mean": float(frame["field_mae"].mean()),
        "field_rmse_mean": float(frame["field_rmse"].mean()),
        "field_linf_max": float(frame["field_linf"].max()),
        "peak_composite_temperature_error_mae": float(
            frame["peak_composite_temperature_error"].abs().mean()
        ),
        "alpha_relative_l2_mean": float(frame["alpha_relative_l2"].mean()),
        "alpha_mae_mean": float(frame["alpha_mae"].mean()),
        "alpha_rmse_mean": float(frame["alpha_rmse"].mean()),
        "alpha_linf_max": float(frame["alpha_linf"].max()),
        "spatial_gradient_relative_l2_mean": float(
            frame["spatial_gradient_relative_l2"].mean()
        ),
        "spatial_gradient_mae_mean": float(frame["spatial_gradient_mae"].mean()),
        "alpha_monotonic_violation_count": int(
            frame["alpha_monotonic_violation_count"].sum()
        ),
        "alpha_bound_violation_count": int(
            frame["alpha_bound_violation_count"].sum()
        ),
    }
    return summary, frame


def _evaluate_objective(
    model: torch.nn.Module,
    prepared: PreparedJointCase1,
    split: str,
    config: JointTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        prepared.dataset(split),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    totals = {"temperature": 0.0, "alpha": 0.0, "gradient": 0.0, "weighted": 0.0}
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, target_temperature, target_alpha, _ in loader:
            inputs = inputs.to(device)
            target_temperature = target_temperature.to(device)
            target_alpha = target_alpha.to(device)
            components = joint_loss_components(
                model(inputs),
                target_temperature,
                target_alpha,
                inputs[..., 4],
            )
            weighted = _weighted_loss(components, config)
            batch_count = len(inputs)
            for key, value in components.items():
                totals[key] += float(value) * batch_count
            totals["weighted"] += float(weighted) * batch_count
            count += batch_count
    return {key: value / count for key, value in totals.items()}


def _evaluate_test(
    model: torch.nn.Module,
    prepared: PreparedJointCase1,
    config: JointTrainConfig,
    device: torch.device,
) -> tuple[dict[str, float | int], pd.DataFrame, dict[str, np.ndarray]]:
    loader = torch.utils.data.DataLoader(
        prepared.dataset("test"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    temperature_predictions: list[np.ndarray] = []
    temperature_targets: list[np.ndarray] = []
    alpha_predictions: list[np.ndarray] = []
    alpha_targets: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    air_inputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, target_temperature, target_alpha, batch_case_ids in loader:
            device_inputs = inputs.to(device)
            outputs = model(device_inputs)
            temperature_predictions.append(
                _decode_temperature(outputs["temperature"].cpu(), prepared).numpy()
            )
            temperature_targets.append(
                _decode_temperature(target_temperature, prepared).numpy()
            )
            alpha_predictions.append(outputs["alpha"].cpu().numpy())
            alpha_targets.append(target_alpha.numpy())
            case_ids.append(batch_case_ids.numpy())
            air_inputs.append(inputs[:, :, 0, 0].numpy())
    arrays = {
        "case_ids": np.concatenate(case_ids),
        "temperature_prediction": np.concatenate(temperature_predictions),
        "temperature_target": np.concatenate(temperature_targets),
        "alpha_prediction": np.concatenate(alpha_predictions),
        "alpha_target": np.concatenate(alpha_targets),
        "air_temperature_normalized": np.concatenate(air_inputs),
    }
    summary, frame = compute_joint_field_metrics(
        arrays["case_ids"],
        arrays["temperature_prediction"],
        arrays["temperature_target"],
        arrays["alpha_prediction"],
        arrays["alpha_target"],
    )
    return summary, frame, arrays


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation: float,
    bad_epochs: int,
    generator: torch.Generator,
    config: JointTrainConfig,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation": best_validation,
        "bad_epochs": bad_epochs,
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "config": _scientific_config(config),
    }


def _write_provenance(
    run_dir: Path,
    config: JointTrainConfig,
    project_root: Path,
    git_sha: str,
) -> None:
    (run_dir / "config_resolved.yaml").write_text(
        _canonical_json(_config_payload(config)), encoding="utf-8"
    )
    status = _git(project_root, "status", "--short")
    (run_dir / "git_state.txt").write_text(
        f"HEAD {git_sha}\n\nstatus --short\n{status}\n", encoding="utf-8"
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"torch={torch.__version__}",
                f"numpy={np.__version__}",
                f"device_request={config.device}",
                f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
                f"num_threads={config.num_threads}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = {
        "data": {
            "path": str(config.data_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": sha256_file(config.data_path),
        },
        "split_manifest": {
            "path": str(config.split_manifest.relative_to(project_root)).replace(
                "\\", "/"
            ),
            "sha256": sha256_file(config.split_manifest),
        },
    }
    (run_dir / "data_checksums.json").write_text(
        _canonical_json(checksums), encoding="utf-8"
    )


def _register_result(
    project_root: Path,
    config: JointTrainConfig,
    model_name: str,
    run_id: str,
    git_sha: str,
    started_at: str,
    completed_at: str,
    metrics_path: Path,
) -> None:
    index_path = project_root / "RESULTS_INDEX.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["run_id"] == run_id for row in rows):
        return
    row = {
        "run_id": run_id,
        "phase": "P2",
        "experiment": config.experiment,
        "model": model_name,
        "dataset": "Case1_joint",
        "split": config.split_manifest.stem,
        "budget": "50",
        "seed": str(config.seed),
        "git_sha": git_sha,
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "metrics_path": str(metrics_path.relative_to(project_root)).replace("\\", "/"),
        "notes": "P2 joint temperature/cure operator",
    }
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writerow(row)


def _train_joint_impl(config: JointTrainConfig) -> dict[str, Any]:
    config = config.validated()
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_joint_run_id(config, git_sha)
    config = replace(config, run_id=run_id)
    run_dir = config.output_root.resolve() / run_id
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"cdcureno.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "stdout.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    session_started_at = datetime.now().astimezone().isoformat()
    started_at_path = run_dir / "STARTED_AT"
    if config.resume and started_at_path.is_file():
        started_at = started_at_path.read_text(encoding="utf-8").strip()
    else:
        started_at = session_started_at
        started_at_path.write_text(f"{started_at}\n", encoding="utf-8")
    with (run_dir / "run_sessions.jsonl").open(
        "a", encoding="utf-8", newline=""
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "event": "session_started",
                    "timestamp": session_started_at,
                    "resume": config.resume,
                },
                sort_keys=True,
            )
            + "\n"
        )
    wall_start = time.perf_counter()
    tracemalloc.start()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if not config.resume:
        _write_provenance(run_dir, config, project_root, git_sha)

    prepared = prepare_joint_case1(config.data_path, config.split_manifest)
    model_family = EXPERIMENT_MODELS[config.experiment]
    model = build_joint_operator(
        model_family,
        input_channels=prepared.inputs.shape[-1],
        width=config.width,
        depth=config.depth,
        modes_time=config.modes_time,
        modes_space=config.modes_space,
    ).to(device)
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
        drop_last=False,
    )
    history_rows: list[dict[str, Any]] = []
    start_epoch = 0
    best_validation = float("inf")
    bad_epochs = 0
    if config.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")
        saved_config = json.loads(
            (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
        )
        if _scientific_config(config) != _scientific_config(
            JointTrainConfig(
                **{
                    **saved_config,
                    "data_path": Path(saved_config["data_path"]),
                    "split_manifest": Path(saved_config["split_manifest"]),
                    "output_root": Path(saved_config["output_root"]),
                    "project_root": Path(saved_config["project_root"]),
                }
            )
        ):
            raise ValueError("Resume configuration differs from the frozen run config.")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history_path = run_dir / "history.parquet"
        if history_path.is_file():
            history_rows = pd.read_parquet(history_path).to_dict(orient="records")

    selected_epoch = -1
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        totals = {
            "temperature": 0.0,
            "alpha": 0.0,
            "gradient": 0.0,
            "weighted": 0.0,
        }
        case_count = 0
        final_gradient_norm = 0.0
        for inputs, target_temperature, target_alpha, _ in train_loader:
            inputs = inputs.to(device)
            target_temperature = target_temperature.to(device)
            target_alpha = target_alpha.to(device)
            optimizer.zero_grad(set_to_none=True)
            components = joint_loss_components(
                model(inputs),
                target_temperature,
                target_alpha,
                inputs[..., 4],
            )
            loss = _weighted_loss(components, config)
            loss.backward()
            final_gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
            )
            optimizer.step()
            batch_count = len(inputs)
            for key, value in components.items():
                totals[key] += float(value.detach()) * batch_count
            totals["weighted"] += float(loss.detach()) * batch_count
            case_count += batch_count
        scheduler.step()
        validation = _evaluate_objective(
            model, prepared, "validation", config, device
        )
        improved = validation["weighted"] < best_validation
        if improved:
            best_validation = validation["weighted"]
            bad_epochs = 0
            selected_epoch = epoch
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_temperature_relative_l2": totals["temperature"] / case_count,
            "train_alpha_relative_l2": totals["alpha"] / case_count,
            "train_spatial_gradient_relative_l2": totals["gradient"] / case_count,
            "train_weighted_objective": totals["weighted"] / case_count,
            "validation_temperature_relative_l2": validation["temperature"],
            "validation_alpha_relative_l2": validation["alpha"],
            "validation_spatial_gradient_relative_l2": validation["gradient"],
            "validation_weighted_objective": validation["weighted"],
            "gradient_norm_before_clip_last_batch": final_gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation,
            bad_epochs,
            generator,
            config,
        )
        _atomic_torch_save(checkpoint, last_path)
        if improved:
            _atomic_torch_save(checkpoint, best_path)
        pd.DataFrame(history_rows).to_parquet(run_dir / "history.parquet", index=False)
        if epoch == start_epoch or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            logger.info(
                "epoch=%d/%d train=%.6f validation=%.6f grad_norm=%.4f seconds=%.3f",
                epoch + 1,
                config.epochs,
                row["train_weighted_objective"],
                validation["weighted"],
                final_gradient_norm,
                row["duration_seconds"],
            )
        if (
            epoch + 1 >= config.minimum_epochs
            and bad_epochs >= config.early_stopping_patience
        ):
            logger.info(
                "early_stop epoch=%d bad_epochs=%d",
                epoch + 1,
                bad_epochs,
            )
            break

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    selected_epoch = int(selected["epoch"])
    test_metrics, test_frame, arrays = _evaluate_test(
        model, prepared, config, device
    )
    test_frame.insert(0, "split", "test")
    test_frame.to_parquet(run_dir / "metrics_per_case.parquet", index=False)
    np.savez_compressed(
        run_dir / "predictions" / "test_predictions.npz",
        **arrays,
    )

    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    completed_at = datetime.now().astimezone().isoformat()
    final_session_wall_seconds = time.perf_counter() - wall_start
    with (run_dir / "run_sessions.jsonl").open(
        "a", encoding="utf-8", newline=""
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "event": "session_completed",
                    "timestamp": completed_at,
                    "wall_seconds": final_session_wall_seconds,
                },
                sort_keys=True,
            )
            + "\n"
        )
    session_records = [
        json.loads(line)
        for line in (run_dir / "run_sessions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    metrics = {
        "run_id": run_id,
        "phase": "P2",
        "experiment": config.experiment,
        "model_family": model_family,
        "seed": config.seed,
        "selection": "best_validation_weighted_objective",
        "selected_epoch": selected_epoch,
        "normalization": prepared.normalization,
        "loss_weights": {
            "temperature": config.temperature_weight,
            "alpha": config.alpha_weight,
            "gradient": config.gradient_weight,
        },
        "test": test_metrics,
        "parameter_count": parameter_count(model),
        "field_shape": list(arrays["temperature_prediction"].shape),
        "device": str(device),
        "num_threads": config.num_threads,
        "summed_epoch_seconds": float(
            sum(row["duration_seconds"] for row in history_rows)
        ),
        "final_session_wall_seconds": final_session_wall_seconds,
        "recorded_session_count": sum(
            record["event"] == "session_started" for record in session_records
        ),
        "resume_session_count": sum(
            record["event"] == "session_started" and record["resume"]
            for record in session_records
        ),
        "peak_process_rss_bytes": _peak_process_rss_bytes(),
        "peak_python_tracemalloc_bytes": tracemalloc_peak,
        "peak_accelerator_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "started_at": started_at,
        "completed_at": completed_at,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(_canonical_json(metrics), encoding="utf-8")
    (run_dir / "DONE").write_text(f"{completed_at}\n", encoding="utf-8")
    if config.register_result:
        _register_result(
            project_root,
            config,
            f"{model_family}_w{model.width}_d{len(model.blocks)}",
            run_id,
            git_sha,
            started_at,
            completed_at,
            metrics_path,
        )
    logger.info(
        "completed run_id=%s field_relative_l2=%.6f alpha_relative_l2=%.6f",
        run_id,
        test_metrics["field_relative_l2_mean"],
        test_metrics["alpha_relative_l2_mean"],
    )
    return metrics


def train_joint(config: JointTrainConfig) -> dict[str, Any]:
    """Run P2 training and preserve a FAILED marker on any exception."""

    config = config.validated()
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_joint_run_id(config, git_sha)
    resolved = replace(config, run_id=run_id)
    try:
        return _train_joint_impl(resolved)
    except Exception as error:
        run_dir = resolved.output_root.resolve() / run_id
        if not (run_dir / "DONE").exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILED").write_text(
                f"{datetime.now().astimezone().isoformat()}\n"
                f"{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
        raise
