"""Auditable exact and corrected legacy ResFNO training wrappers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from cdcureno.legacy.audit import load_legacy_module, sha256_file


Experiment = Literal["legacy_resfno_exact", "legacy_resfno_corrected"]


@dataclass(frozen=True)
class LegacyTrainConfig:
    experiment: Experiment
    data_path: Path
    split_manifest: Path
    output_root: Path
    project_root: Path
    location: int = 35
    task: Literal["T", "A"] = "T"
    epochs: int = 300
    batch_size: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    modes: int = 16
    width: int = 64
    smoothness_weight: float = 0.5
    seed: int = 0
    device: str = "cpu"
    run_id: str | None = None
    resume: bool = False
    register_result: bool = True
    num_threads: int = 4

    def validated(self) -> "LegacyTrainConfig":
        if self.experiment not in {
            "legacy_resfno_exact",
            "legacy_resfno_corrected",
        }:
            raise ValueError(f"Unsupported experiment: {self.experiment}")
        if self.task not in {"T", "A"}:
            raise ValueError(f"task must be T or A, got {self.task}")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.location < 0:
            raise ValueError("location must be nonnegative")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        if self.experiment == "legacy_resfno_exact" and self.task == "A":
            raise ValueError(
                "The exact alpha path is a confirmed crashing legacy behavior; "
                "use legacy_resfno_corrected for task A."
            )
        if self.experiment == "legacy_resfno_exact" and self.smoothness_weight != 0:
            return replace(self, smoothness_weight=0.0)
        return self


@dataclass(frozen=True)
class TorchRangeNormalizer:
    minimum: torch.Tensor
    maximum: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "TorchRangeNormalizer":
        minimum = torch.min(values)
        maximum = torch.max(values)
        if not torch.isfinite(minimum) or not torch.isfinite(maximum):
            raise ValueError("Normalizer values must be finite.")
        if maximum <= minimum:
            raise ValueError("Normalizer maximum must exceed minimum.")
        return cls(minimum=minimum, maximum=maximum)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.minimum) / (self.maximum - self.minimum)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.maximum - self.minimum) + self.minimum

    def metadata(self) -> dict[str, float]:
        return {
            "minimum": float(self.minimum.detach().cpu()),
            "maximum": float(self.maximum.detach().cpu()),
        }


def _jsonable_config(config: LegacyTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("data_path", "split_manifest", "output_root", "project_root"):
        payload[key] = str(payload[key])
    return payload


def _scientific_config(config: LegacyTrainConfig) -> dict[str, Any]:
    payload = _jsonable_config(config)
    payload.pop("resume", None)
    payload.pop("register_result", None)
    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def make_run_id(config: LegacyTrainConfig, git_sha: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    budget = "n50"
    return (
        f"{timestamp}__P1__{config.experiment}__Case1__{budget}__"
        f"seed{config.seed}__{git_sha[:7]}"
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    observed = [case for name in ("train", "validation", "test") for case in splits[name]]
    if len(observed) != len(set(observed)):
        raise ValueError("Split manifest contains duplicate case IDs.")
    return payload


def prepare_data(
    config: LegacyTrainConfig,
) -> tuple[
    dict[str, torch.Tensor],
    TorchRangeNormalizer,
    TorchRangeNormalizer | None,
    dict[str, Any],
]:
    arrays = sio.loadmat(config.data_path)
    data_t = arrays["dataT"]
    data_a = arrays["dataA"]
    data_air = arrays["dataTair"]
    if data_t.shape != data_a.shape:
        raise ValueError(f"dataT/dataA shape mismatch: {data_t.shape} vs {data_a.shape}")
    if not 0 <= config.location < data_t.shape[1]:
        raise ValueError(
            f"location {config.location} outside valid range [0,{data_t.shape[1] - 1}]"
        )

    manifest = load_manifest(config.split_manifest)
    splits = manifest["splits"]
    train_ids = torch.tensor(splits["train"], dtype=torch.long)
    x_all = torch.from_numpy(data_air.astype(np.float32))
    target = data_t if config.task == "T" else data_a
    y_all = torch.from_numpy(target[:, config.location, :].astype(np.float32))

    if config.experiment == "legacy_resfno_exact":
        x_normalizer = TorchRangeNormalizer.fit(x_all)
        y_normalizer = (
            TorchRangeNormalizer.fit(y_all) if config.task == "T" else None
        )
        normalization_scope = "all_200_cases_legacy"
    else:
        x_normalizer = TorchRangeNormalizer.fit(x_all[train_ids])
        y_normalizer = (
            TorchRangeNormalizer.fit(y_all[train_ids]) if config.task == "T" else None
        )
        normalization_scope = "train_cases_only"

    x_encoded = x_normalizer.encode(x_all)
    y_encoded = y_normalizer.encode(y_all) if y_normalizer is not None else y_all
    prepared = {
        "x_encoded": x_encoded,
        "y_encoded": y_encoded,
        "x_physical": x_all,
        "y_physical": y_all,
    }
    metadata = {
        "normalization_scope": normalization_scope,
        "x": x_normalizer.metadata(),
        "y": y_normalizer.metadata() if y_normalizer is not None else None,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
    }
    return prepared, x_normalizer, y_normalizer, metadata


def _relative_l2_per_case(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm((prediction - target).reshape(len(target), -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(len(target), -1), dim=1)
    return difference / torch.clamp_min(denominator, 1e-12)


def _evaluate(
    model: torch.nn.Module,
    case_ids: list[int],
    prepared: dict[str, torch.Tensor],
    y_normalizer: TorchRangeNormalizer | None,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    if not case_ids:
        return {}, pd.DataFrame(), np.empty((0,))
    model.eval()
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(case_ids), batch_size):
            batch_ids = case_ids[start : start + batch_size]
            index = torch.tensor(batch_ids, dtype=torch.long)
            x = prepared["x_encoded"][index].reshape(len(batch_ids), -1, 1).to(device)
            encoded = model(x).squeeze(-1).cpu()
            physical = y_normalizer.decode(encoded) if y_normalizer is not None else encoded
            target = prepared["y_physical"][index]
            rel_l2 = _relative_l2_per_case(physical, target)
            difference = physical - target
            for offset, case_id in enumerate(batch_ids):
                error = difference[offset]
                row = {
                    "case_id": case_id,
                    "relative_l2": float(rel_l2[offset]),
                    "mae": float(torch.mean(torch.abs(error))),
                    "rmse": float(torch.sqrt(torch.mean(error**2))),
                    "linf": float(torch.max(torch.abs(error))),
                    "peak_value_error": float(
                        torch.max(physical[offset]) - torch.max(target[offset])
                    ),
                    "time_to_peak_index_error": int(
                        torch.argmax(physical[offset]) - torch.argmax(target[offset])
                    ),
                }
                rows.append(row)
            predictions.append(physical.numpy())
    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    summary = {
        "relative_l2_mean": float(frame["relative_l2"].mean()),
        "relative_l2_median": float(frame["relative_l2"].median()),
        "mae_mean": float(frame["mae"].mean()),
        "rmse_mean": float(frame["rmse"].mean()),
        "linf_max": float(frame["linf"].max()),
        "peak_value_error_mae": float(frame["peak_value_error"].abs().mean()),
        "time_to_peak_index_error_mae": float(
            frame["time_to_peak_index_error"].abs().mean()
        ),
        "case_count": int(len(frame)),
    }
    return summary, frame, np.concatenate(predictions, axis=0)


def _encoded_relative_l2_mean(
    model: torch.nn.Module,
    case_ids: list[int],
    prepared: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(case_ids), batch_size):
            batch_ids = case_ids[start : start + batch_size]
            index = torch.tensor(batch_ids, dtype=torch.long)
            x = prepared["x_encoded"][index].reshape(len(batch_ids), -1, 1).to(device)
            y = prepared["y_encoded"][index].to(device)
            prediction = model(x).squeeze(-1)
            total += float(_relative_l2_per_case(prediction, y).sum())
            count += len(batch_ids)
    return total / count


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation: float,
    generator: torch.Generator,
    config: LegacyTrainConfig,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation": best_validation,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator_state": generator.get_state(),
        "config": _scientific_config(config),
    }


def _restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    config: LegacyTrainConfig,
) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["config"] != _scientific_config(config):
        raise ValueError("Refusing to resume with a different resolved configuration.")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    generator.set_state(payload["loader_generator_state"])
    return int(payload["epoch"]) + 1, float(payload["best_validation"])


def _write_run_provenance(
    run_dir: Path,
    config: LegacyTrainConfig,
    project_root: Path,
    git_sha: str,
) -> None:
    (run_dir / "config_resolved.yaml").write_text(
        _canonical_json(_jsonable_config(config)), encoding="utf-8"
    )
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (run_dir / "environment.txt").write_text(
        f"python={sys.version}\nexecutable={sys.executable}\n"
        f"torch={torch.__version__}\ncuda_available={torch.cuda.is_available()}\n"
        + environment,
        encoding="utf-8",
    )
    status = _git(project_root, "status", "--short")
    (run_dir / "git_state.txt").write_text(
        f"commit={git_sha}\nstatus:\n{status}\n", encoding="utf-8"
    )
    checksums = {
        "data": {
            "path": str(config.data_path),
            "sha256": sha256_file(config.data_path),
        },
        "split_manifest": {
            "path": str(config.split_manifest),
            "sha256": sha256_file(config.split_manifest),
        },
    }
    (run_dir / "data_checksums.json").write_text(
        _canonical_json(checksums), encoding="utf-8"
    )


def _validate_resume_configuration(
    run_dir: Path, config: LegacyTrainConfig
) -> None:
    resolved_path = run_dir / "config_resolved.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Resolved configuration is missing: {resolved_path}")
    existing = json.loads(resolved_path.read_text(encoding="utf-8"))
    existing.pop("resume", None)
    existing.pop("register_result", None)
    if existing != _scientific_config(config):
        raise ValueError("Refusing to resume with a different resolved configuration.")


def _register_result(
    project_root: Path,
    config: LegacyTrainConfig,
    run_id: str,
    git_sha: str,
    started_at: str,
    completed_at: str,
    metrics_path: Path,
) -> None:
    index_path = project_root / "RESULTS_INDEX.csv"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["run_id"] == run_id for row in rows):
        return
    row = {
        "run_id": run_id,
        "phase": "P1",
        "experiment": config.experiment,
        "model": f"FNO1d_m{config.modes}_w{config.width}_x{config.location}",
        "dataset": "Case1",
        "split": config.split_manifest.stem,
        "budget": "50",
        "seed": str(config.seed),
        "git_sha": git_sha,
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "metrics_path": str(metrics_path.relative_to(project_root)).replace("\\", "/"),
        "notes": "P1 one-location baseline",
    }
    fields = list(row)
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(row)


def _train_legacy_impl(config: LegacyTrainConfig) -> dict[str, Any]:
    config = config.validated()
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_run_id(config, git_sha)
    config = replace(config, run_id=run_id)
    run_dir = config.output_root.resolve() / run_id
    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
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

    started_at = datetime.now().astimezone().isoformat()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    if config.resume:
        _validate_resume_configuration(run_dir, config)
    else:
        _write_run_provenance(run_dir, config, project_root, git_sha)

    prepared, _, y_normalizer, normalization = prepare_data(config)
    manifest = load_manifest(config.split_manifest)
    splits = manifest["splits"]
    legacy = load_legacy_module(project_root / "external" / "ResFNO")
    model = legacy.FNO1d(config.modes, config.width, config.task).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_ids = torch.tensor(splits["train"], dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(
        prepared["x_encoded"][train_ids].reshape(len(train_ids), -1, 1),
        prepared["y_encoded"][train_ids],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    history_rows: list[dict[str, Any]] = []
    start_epoch = 0
    best_validation = float("inf")
    if config.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")
        start_epoch, best_validation = _restore_checkpoint(
            last_path, model, optimizer, scheduler, generator, config
        )
        history_path = run_dir / "history.parquet"
        if history_path.is_file():
            history_rows = pd.read_parquet(history_path).to_dict(orient="records")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_rel_sum = 0.0
        train_objective_sum = 0.0
        case_count = 0
        final_smoothness = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            encoded = output.squeeze(-1)
            per_case = _relative_l2_per_case(encoded, y)
            relative_l2 = per_case.sum()
            first = output[:, 1:, :] - output[:, :-1, :]
            second = first[:, 1:, :] - first[:, :-1, :]
            smoothness = torch.max(torch.abs(second))
            if config.experiment == "legacy_resfno_exact":
                objective = relative_l2
            else:
                objective = relative_l2 + config.smoothness_weight * smoothness
            objective.backward()
            optimizer.step()
            train_rel_sum += float(relative_l2.detach())
            train_objective_sum += float(objective.detach())
            final_smoothness = float(smoothness.detach())
            case_count += len(x)
        scheduler.step()

        train_relative_l2 = train_rel_sum / case_count
        train_objective = train_objective_sum / case_count
        validation_metrics: dict[str, float] = {}
        test_epoch_metrics: dict[str, float] = {}
        if config.experiment == "legacy_resfno_corrected":
            validation_metrics, _, _ = _evaluate(
                model,
                splits["validation"],
                prepared,
                y_normalizer,
                config.batch_size,
                device,
            )
            selection_metric = validation_metrics["relative_l2_mean"]
        else:
            legacy_encoded_test = _encoded_relative_l2_mean(
                model,
                splits["test"],
                prepared,
                config.batch_size,
                device,
            )
            test_epoch_metrics = {"relative_l2_mean": legacy_encoded_test}
            selection_metric = float("nan")

        row = {
            "epoch": epoch,
            "train_relative_l2": train_relative_l2,
            "train_objective": train_objective,
            "last_batch_smoothness": final_smoothness,
            "validation_relative_l2": validation_metrics.get("relative_l2_mean"),
            "legacy_test_relative_l2": test_epoch_metrics.get("relative_l2_mean"),
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
            generator,
            config,
        )
        if config.experiment == "legacy_resfno_corrected" and selection_metric < best_validation:
            best_validation = selection_metric
            checkpoint["best_validation"] = best_validation
            _atomic_torch_save(checkpoint, best_path)
        _atomic_torch_save(checkpoint, last_path)
        pd.DataFrame(history_rows).to_parquet(run_dir / "history.parquet", index=False)
        if epoch == start_epoch or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            logger.info(
                "epoch=%d/%d train_rel_l2=%.6f val_rel_l2=%s seconds=%.3f",
                epoch + 1,
                config.epochs,
                train_relative_l2,
                (
                    f"{validation_metrics['relative_l2_mean']:.6f}"
                    if validation_metrics
                    else "n/a"
                ),
                row["duration_seconds"],
            )

    if config.experiment == "legacy_resfno_exact":
        # The legacy code has no validation selection. "best" is explicitly the last epoch.
        final_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        _atomic_torch_save(final_checkpoint, best_path)
        selection = "last_epoch_legacy_no_validation"
    else:
        selection = "best_validation_relative_l2"

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    test_metrics, test_frame, test_predictions = _evaluate(
        model,
        splits["test"],
        prepared,
        y_normalizer,
        config.batch_size,
        device,
    )
    test_frame.insert(0, "split", "test")
    test_frame.to_parquet(run_dir / "metrics_per_case.parquet", index=False)
    np.savez_compressed(
        run_dir / "predictions" / "test_predictions.npz",
        case_ids=np.asarray(splits["test"], dtype=np.int64),
        prediction=test_predictions,
        target=prepared["y_physical"][torch.tensor(splits["test"])].numpy(),
        input_air=prepared["x_physical"][torch.tensor(splits["test"])].numpy(),
    )

    metrics = {
        "run_id": run_id,
        "phase": "P1",
        "experiment": config.experiment,
        "task": config.task,
        "location": config.location,
        "seed": config.seed,
        "selection": selection,
        "selected_epoch": int(selected["epoch"]),
        "normalization": normalization,
        "test": test_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "num_threads": config.num_threads,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(_canonical_json(metrics), encoding="utf-8")
    completed_at = metrics["completed_at"]
    (run_dir / "DONE").write_text(f"{completed_at}\n", encoding="utf-8")
    if config.register_result:
        _register_result(
            project_root,
            config,
            run_id,
            git_sha,
            started_at,
            completed_at,
            metrics_path,
        )
    logger.info(
        "completed run_id=%s test_relative_l2=%.6f",
        run_id,
        test_metrics["relative_l2_mean"],
    )
    return metrics


def train_legacy(config: LegacyTrainConfig) -> dict[str, Any]:
    """Run training and preserve an explicit FAILED marker on any exception."""

    config = config.validated()
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_run_id(config, git_sha)
    resolved = replace(config, run_id=run_id)
    try:
        return _train_legacy_impl(resolved)
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
