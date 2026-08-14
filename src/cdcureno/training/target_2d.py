"""Auditable P5 RP-FFNO target-pilot training.

This module is intentionally limited to the pre-registered one-seed P5 pilot.
It exposes only the frozen P4 training budget and validation split through
``prepare_target_2d_training``.  ID-test and OOD loaders are neither imported
nor constructed.

The two supported methods differ only in initialization:

``scratch_ffno``
    Deterministic random initialization of the exact target architecture.

``restriction_transfer_ffno``
    The already verified restriction-preserving inflated P3 checkpoint.

Both methods train every target parameter (T2) with identical optimizer
groups, data, losses, epoch budget, micro-batch size, and gradient
accumulation.  A CUDA full-resolution resource preflight freezes the latter
two settings before either method can run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from cdcureno.data.source_1d import INPUT_CHANNELS, prepare_source_1d
from cdcureno.data.target_2d import (
    TARGET_ADAPTER_CHANNELS,
    TARGET_INPUT_CHANNELS,
    PreparedTarget2DTraining,
    lift_homogeneous_source_input,
    prepare_target_2d_training,
)
from cdcureno.models.checkpoint_inflation import (
    load_inflated_target,
    load_target_config,
)
from cdcureno.models.target_operators import AxisFactorized2DOperator


EXPERIMENT = "p5_rp_ffno_target_pilot_v1"
MODEL_FAMILY = "axis_factorized_2d"
CHECKPOINT_SCHEMA_VERSION = 1
RESOURCE_PROFILE_SCHEMA_VERSION = 1
PILOT_METHODS = ("scratch_ffno", "restriction_transfer_ffno")
PILOT_BUDGETS = (8, 16)
PILOT_SEEDS = (0,)
PARAMETER_GROUP_RATIOS = {
    "adapter": 1.0,
    "lift_and_heads": 0.5,
    "shared_core": 0.1,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _path_string(path: Path | None) -> str | None:
    return None if path is None else path.resolve().as_posix()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _artifact_record(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected run artifact is missing: {resolved}")
    return {
        "path": _portable_path(resolved, project_root),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _implementation_state(project_root: Path) -> dict[str, Any]:
    relative_paths = (
        "src/cdcureno/training/target_2d.py",
        "src/cdcureno/data/target_2d.py",
        "src/cdcureno/data/source_1d.py",
        "src/cdcureno/data/normalization.py",
        "src/cdcureno/models/target_operators.py",
        "src/cdcureno/models/checkpoint_inflation.py",
        "src/cdcureno/models/joint_operators.py",
        "src/cdcureno/physics/__init__.py",
        "src/cdcureno/physics/as4_8552.py",
        "src/cdcureno/solvers/__init__.py",
        "src/cdcureno/solvers/conservative_1d.py",
    )
    implementation_paths = tuple(
        (project_root / relative_path).resolve()
        for relative_path in relative_paths
    )
    source_files = {
        _portable_path(path, project_root): _sha256_file(path)
        for path in implementation_paths
    }
    return {
        "head_sha": _git(project_root, "rev-parse", "HEAD"),
        "status_short": _git(project_root, "status", "--short"),
        "source_file_sha256": source_files,
        "implementation_sha256": _sha256_json(source_files),
    }


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; "
                f"device_count={torch.cuda.device_count()}."
            )
        return torch.device("cuda", index)
    return device


def _device_payload(device: torch.device, requested: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        properties = torch.cuda.get_device_properties(index)
        payload["active_device"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "multi_processor_count": int(properties.multi_processor_count),
        }
    return payload


def _runtime_fingerprint(
    device: torch.device, requested: str
) -> dict[str, Any]:
    return {
        "python": sys.version.replace(os.linesep, " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "device": _device_payload(device, requested),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
        ),
        "num_threads": torch.get_num_threads(),
    }


@dataclass(frozen=True)
class TargetPilotTrainConfig:
    """Frozen scientific and operational contract for one P5 pilot run."""

    project_root: Path
    target_split_manifest: Path
    source_checkpoint: Path
    inflated_checkpoint: Path
    target_model_config: Path
    source_data_path: Path
    source_split_manifest: Path
    output_root: Path
    resource_profile_path: Path
    config_file: Path | None = None
    run_id: str | None = None
    method: str = "scratch_ffno"
    label_budget: int = 8
    seed: int = 0
    expected_target_split_sha256: str | None = None
    expected_source_checkpoint_sha256: str | None = None
    expected_inflated_checkpoint_sha256: str | None = None
    expected_target_model_config_sha256: str | None = None
    expected_source_data_sha256: str | None = None
    expected_source_split_sha256: str | None = None
    width: int = 32
    depth: int = 4
    modes_time: int = 24
    modes_z: int = 12
    modes_x: int = 12
    lateral_rank: int = 4
    expected_parameter_count: int | None = 172_610
    epochs: int = 120
    minimum_epochs: int = 40
    early_stopping_patience: int = 20
    effective_batch_size: int = 4
    preflight_candidate_batch_sizes: tuple[int, ...] = (2, 1)
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature_weight: float = 1.0
    alpha_weight: float = 0.5
    gradient_x_weight: float = 0.05
    gradient_z_weight: float = 0.05
    restriction_weight: float = 0.1
    restriction_validation_case_count: int = 4
    restriction_lateral_invariance_max: float = 1.0e-6
    restriction_lateral_range_max: float = 1.0e-5
    gradient_clip: float = 1.0
    device: str = "cuda"
    num_threads: int = 8
    verify_array_checksums: bool = True
    require_resource_profile: bool = True
    resume: bool = False

    def validated(
        self, *, require_resolved_resources: bool = False
    ) -> "TargetPilotTrainConfig":
        if self.method not in PILOT_METHODS:
            raise ValueError(
                f"method must be one of {list(PILOT_METHODS)}, got "
                f"{self.method!r}."
            )
        if self.label_budget not in PILOT_BUDGETS:
            raise ValueError(
                f"label_budget must be one of {list(PILOT_BUDGETS)}."
            )
        if self.seed not in PILOT_SEEDS:
            raise ValueError(
                f"The frozen P5 pilot permits only seed {PILOT_SEEDS[0]}."
            )
        positive_integers = {
            "width": self.width,
            "depth": self.depth,
            "modes_time": self.modes_time,
            "modes_z": self.modes_z,
            "modes_x": self.modes_x,
            "lateral_rank": self.lateral_rank,
            "epochs": self.epochs,
            "minimum_epochs": self.minimum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "effective_batch_size": self.effective_batch_size,
            "num_threads": self.num_threads,
            "restriction_validation_case_count": (
                self.restriction_validation_case_count
            ),
        }
        invalid = [
            name for name, value in positive_integers.items() if value < 1
        ]
        if invalid:
            raise ValueError(f"These settings must be positive: {invalid}.")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs.")
        candidates = tuple(
            int(value) for value in self.preflight_candidate_batch_sizes
        )
        if (
            not candidates
            or any(value < 1 for value in candidates)
            or tuple(sorted(set(candidates), reverse=True)) != candidates
            or any(self.effective_batch_size % value for value in candidates)
        ):
            raise ValueError(
                "Preflight batch candidates must be unique descending positive "
                "divisors of effective_batch_size."
            )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("gradient_clip", self.gradient_clip),
            (
                "restriction_lateral_invariance_max",
                self.restriction_lateral_invariance_max,
            ),
            (
                "restriction_lateral_range_max",
                self.restriction_lateral_range_max,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        weights = {
            "temperature_weight": self.temperature_weight,
            "alpha_weight": self.alpha_weight,
            "gradient_x_weight": self.gradient_x_weight,
            "gradient_z_weight": self.gradient_z_weight,
            "restriction_weight": self.restriction_weight,
        }
        if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("Every loss weight must be finite and nonnegative.")
        if (
            self.temperature_weight
            + self.alpha_weight
            + self.gradient_x_weight
            + self.gradient_z_weight
            <= 0.0
        ):
            raise ValueError("Validation objective weights cannot all be zero.")
        if self.lateral_rank > self.width:
            raise ValueError("lateral_rank cannot exceed width.")
        for name, value in (
            ("micro_batch_size", self.micro_batch_size),
            (
                "gradient_accumulation_steps",
                self.gradient_accumulation_steps,
            ),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when resolved.")
        if (
            self.micro_batch_size is not None
            and self.gradient_accumulation_steps is not None
            and self.micro_batch_size * self.gradient_accumulation_steps
            != self.effective_batch_size
        ):
            raise ValueError(
                "micro_batch_size * gradient_accumulation_steps must equal "
                "effective_batch_size."
            )
        if require_resolved_resources and (
            self.micro_batch_size is None
            or self.gradient_accumulation_steps is None
        ):
            raise ValueError(
                "Training requires the preflight-resolved micro-batch and "
                "gradient-accumulation settings."
            )
        for name, value in (
            (
                "expected_target_split_sha256",
                self.expected_target_split_sha256,
            ),
            (
                "expected_source_checkpoint_sha256",
                self.expected_source_checkpoint_sha256,
            ),
            (
                "expected_inflated_checkpoint_sha256",
                self.expected_inflated_checkpoint_sha256,
            ),
            (
                "expected_target_model_config_sha256",
                self.expected_target_model_config_sha256,
            ),
            (
                "expected_source_data_sha256",
                self.expected_source_data_sha256,
            ),
            (
                "expected_source_split_sha256",
                self.expected_source_split_sha256,
            ),
        ):
            if value is not None and (
                len(value) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in value
                )
            ):
                raise ValueError(f"{name} must be a 64-character SHA256.")
        if self.expected_parameter_count is not None and (
            self.expected_parameter_count < 1
        ):
            raise ValueError("expected_parameter_count must be positive.")
        return replace(
            self, preflight_candidate_batch_sizes=candidates
        )


def _config_payload(config: TargetPilotTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for name in (
        "project_root",
        "target_split_manifest",
        "source_checkpoint",
        "inflated_checkpoint",
        "target_model_config",
        "source_data_path",
        "source_split_manifest",
        "output_root",
        "resource_profile_path",
        "config_file",
    ):
        payload[name] = _path_string(getattr(config, name))
    payload["preflight_candidate_batch_sizes"] = list(
        config.preflight_candidate_batch_sizes
    )
    return payload


def _scientific_config(config: TargetPilotTrainConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    payload.pop("resume", None)
    return payload


def _checked_file(
    path: Path, expected_sha256: str | None, *, label: str
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    digest = _sha256_file(resolved)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f"{label} SHA256 differs from the frozen config.")
    return {
        "path": resolved.as_posix(),
        "sha256": digest,
        "bytes": resolved.stat().st_size,
    }


def _input_checksums(config: TargetPilotTrainConfig) -> dict[str, Any]:
    payload = {
        "target_id_manifest": _checked_file(
            config.target_split_manifest,
            config.expected_target_split_sha256,
            label="target ID split manifest",
        ),
        "source_checkpoint": _checked_file(
            config.source_checkpoint,
            config.expected_source_checkpoint_sha256,
            label="source checkpoint",
        ),
        "inflated_checkpoint": _checked_file(
            config.inflated_checkpoint,
            config.expected_inflated_checkpoint_sha256,
            label="inflated target checkpoint",
        ),
        "target_model_config": _checked_file(
            config.target_model_config,
            config.expected_target_model_config_sha256,
            label="target model config",
        ),
        "source_virtual_input_data": _checked_file(
            config.source_data_path,
            config.expected_source_data_sha256,
            label="source virtual-input data",
        ),
        "source_virtual_input_split": _checked_file(
            config.source_split_manifest,
            config.expected_source_split_sha256,
            label="source virtual-input split",
        ),
    }
    if config.config_file is not None:
        payload["declared_experiment_config"] = _checked_file(
            config.config_file, None, label="declared experiment config"
        )
    return payload


def _model_spec_from_config(
    config: TargetPilotTrainConfig,
) -> dict[str, Any]:
    payload = load_target_config(config.target_model_config)
    required = {
        "family": MODEL_FAMILY,
        "temporal_family": "spectral_noncausal",
        "causal": False,
        "source_channel_names": list(INPUT_CHANNELS),
        "new_channel_names": list(TARGET_ADAPTER_CHANNELS),
        "width": config.width,
        "depth": config.depth,
        "modes_time": config.modes_time,
        "modes_z": config.modes_z,
        "modes_x": config.modes_x,
        "lateral_rank": config.lateral_rank,
    }
    mismatches = {
        name: {"expected": expected, "actual": payload.get(name)}
        for name, expected in required.items()
        if payload.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "Target model YAML differs from the frozen trainer contract: "
            f"{mismatches}."
        )
    return {
        **required,
        "input_channels": len(TARGET_INPUT_CHANNELS),
        "channel_names": list(TARGET_INPUT_CHANNELS),
        "axis_order": payload.get("axis_order"),
        "output_order": payload.get("output_order"),
        "config_sha256": payload["_config_sha256"],
    }


def _new_target_model(
    config: TargetPilotTrainConfig,
) -> tuple[AxisFactorized2DOperator, dict[str, Any]]:
    spec = _model_spec_from_config(config)
    model = AxisFactorized2DOperator(
        source_channel_names=spec["source_channel_names"],
        new_channel_names=spec["new_channel_names"],
        width=spec["width"],
        depth=spec["depth"],
        modes_time=spec["modes_time"],
        modes_z=spec["modes_z"],
        modes_x=spec["modes_x"],
        lateral_rank=spec["lateral_rank"],
        adapter_seed=config.seed,
    )
    return model, spec


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _clone_state_to_cpu(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def _build_initialized_model(
    config: TargetPilotTrainConfig,
    device: torch.device,
) -> tuple[AxisFactorized2DOperator, dict[str, Any], dict[str, Any]]:
    expected_spec = _model_spec_from_config(config)
    if config.method == "scratch_ffno":
        # Global RNG was already frozen by _configure_determinism.  Keep the
        # architecture-specific zero lateral output factor intact.
        model, spec = _new_target_model(config)
        initialization = {
            "method": config.method,
            "deterministic_seed": config.seed,
            "source_checkpoint_weights_loaded": False,
            "inflated_checkpoint_weights_loaded": False,
            "architecture_specific_lateral_zero_residual_preserved": True,
        }
    else:
        model, payload = load_inflated_target(
            config.inflated_checkpoint, device="cpu"
        )
        spec = dict(payload["model_config"])
        for name in (
            "width",
            "depth",
            "modes_time",
            "modes_z",
            "modes_x",
            "lateral_rank",
            "source_channel_names",
            "new_channel_names",
        ):
            actual = (
                list(spec[name])
                if name in {"source_channel_names", "new_channel_names"}
                else spec[name]
            )
            if actual != expected_spec[name]:
                raise ValueError(
                    f"Inflated checkpoint {name} differs from target YAML."
                )
        report = payload.get("inflation_report")
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise ValueError(
                "Transfer initialization requires a passed inflation report."
            )
        initialization = {
            "method": config.method,
            "deterministic_seed": report.get("deterministic_seed"),
            "source_checkpoint_weights_loaded": True,
            "inflated_checkpoint_weights_loaded": True,
            "inflation_verification_passed": True,
            "inflated_checkpoint_sha256": _sha256_file(
                config.inflated_checkpoint
            ),
        }
    stage = model.set_transfer_stage("T2")
    if stage["trainable_parameters"] != stage["total_parameters"]:
        raise AssertionError("The P5 pilot must use full T2 trainability.")
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        config.expected_parameter_count is not None
        and actual_count != config.expected_parameter_count
    ):
        raise ValueError(
            f"Target parameter count {actual_count} differs from pinned "
            f"{config.expected_parameter_count}."
        )
    initialization["initial_state_sha256"] = _state_dict_sha256(
        model.state_dict()
    )
    initialization["trainability"] = stage
    model.to(device)
    return model, expected_spec, initialization


def build_parameter_groups(
    model: AxisFactorized2DOperator,
    *,
    base_learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build exhaustive, non-overlapping P5 discriminative LR groups."""

    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in PARAMETER_GROUP_RATIOS
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(
                "P5 pilot parameter grouping expects full T2 trainability."
            )
        if name.startswith("lift.geometry.") or ".lateral." in name:
            group = "adapter"
        elif name.startswith("lift.") or name.startswith("head."):
            group = "lift_and_heads"
        else:
            group = "shared_core"
        grouped[group].append((name, parameter))
    all_names = [name for values in grouped.values() for name, _ in values]
    expected_names = [name for name, _ in model.named_parameters()]
    if len(all_names) != len(set(all_names)) or set(all_names) != set(
        expected_names
    ):
        raise AssertionError("Parameter groups are not exhaustive and disjoint.")
    optimizer_groups: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for group_name, ratio in PARAMETER_GROUP_RATIOS.items():
        values = grouped[group_name]
        if not values:
            raise AssertionError(f"Parameter group {group_name} is empty.")
        learning_rate = base_learning_rate * ratio
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in values],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": group_name,
                "lr_ratio": ratio,
            }
        )
        report[group_name] = {
            "lr_ratio": ratio,
            "initial_learning_rate": learning_rate,
            "parameter_tensor_count": len(values),
            "parameter_count": sum(
                parameter.numel() for _, parameter in values
            ),
            "parameter_names": [name for name, _ in values],
        }
    report["total_parameter_count"] = sum(
        value["parameter_count"]
        for key, value in report.items()
        if key in PARAMETER_GROUP_RATIOS
    )
    return optimizer_groups, report


def _masked_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Relative-L2 tensors and mask must have equal shapes.")
    weights = mask.to(dtype=prediction.dtype)
    dimensions = tuple(range(1, prediction.ndim))
    numerator = torch.linalg.vector_norm(
        (prediction - target) * weights, dim=dimensions
    )
    denominator = torch.linalg.vector_norm(
        target * weights, dim=dimensions
    ).clamp_min(torch.finfo(prediction.dtype).eps)
    return torch.mean(numerator / denominator)


def target_loss_components(
    outputs: Mapping[str, torch.Tensor],
    temperature: torch.Tensor,
    alpha: torch.Tensor,
    composite_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return composite-only T/alpha and explicit x/z gradient losses."""

    predicted_temperature = outputs["temperature"]
    predicted_alpha = outputs["alpha"]
    if (
        predicted_temperature.shape != temperature.shape
        or predicted_alpha.shape != alpha.shape
        or composite_mask.shape != temperature.shape
    ):
        raise ValueError("Target output, labels, and mask shapes must match.")
    mask = composite_mask > 0.5
    temperature_loss = _masked_relative_l2(
        predicted_temperature, temperature, mask
    )
    alpha_loss = _masked_relative_l2(predicted_alpha, alpha, mask)

    temperature_x = torch.diff(temperature, dim=3)
    prediction_x = torch.diff(predicted_temperature, dim=3)
    mask_x = mask[..., 1:] & mask[..., :-1]
    temperature_z = torch.diff(temperature, dim=2)
    prediction_z = torch.diff(predicted_temperature, dim=2)
    mask_z = mask[:, :, 1:, :] & mask[:, :, :-1, :]
    gradient_x = _masked_relative_l2(
        prediction_x, temperature_x, mask_x
    )
    gradient_z = _masked_relative_l2(
        prediction_z, temperature_z, mask_z
    )
    return {
        "temperature": temperature_loss,
        "alpha": alpha_loss,
        "gradient_x": gradient_x,
        "gradient_z": gradient_z,
    }


def weighted_validation_objective(
    components: Mapping[str, torch.Tensor],
    config: TargetPilotTrainConfig,
) -> torch.Tensor:
    """Frozen checkpoint-selection objective; no restriction term."""

    return (
        config.temperature_weight * components["temperature"]
        + config.alpha_weight * components["alpha"]
        + config.gradient_x_weight * components["gradient_x"]
        + config.gradient_z_weight * components["gradient_z"]
    )


def homogeneous_restriction_loss(
    model: AxisFactorized2DOperator,
    source_inputs: torch.Tensor,
    *,
    nx: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Label-free lateral-invariance loss on valid extruded source inputs.

    No source prediction or target label is used.  This avoids turning the
    scratch comparator into a distillation method.  The six target-only
    channels are appended as exact zeros by the trusted extrusion helper.
    """

    if source_inputs.ndim != 4 or source_inputs.shape[-1] != len(
        INPUT_CHANNELS
    ):
        raise ValueError("Virtual source input must have shape [B,Nt,Nz,14].")
    target_inputs = lift_homogeneous_source_input(source_inputs, nx)
    suffix = target_inputs[..., len(INPUT_CHANNELS) :]
    if torch.count_nonzero(suffix):
        raise AssertionError(
            "Restriction virtual inputs must have exact-zero target suffix."
        )
    outputs = model(target_inputs)
    terms: dict[str, torch.Tensor] = {}
    for field in ("temperature", "alpha"):
        values = outputs[field]
        centered = values - torch.mean(values, dim=3, keepdim=True)
        numerator = torch.mean(centered**2)
        denominator = torch.mean(values**2).clamp_min(
            torch.finfo(values.dtype).eps
        )
        terms[field] = numerator / denominator
    total = terms["temperature"] + terms["alpha"]
    return total, terms


class StatefulShuffleSampler(Sampler[int]):
    """Generator-backed sampler whose exact position is checkpointable."""

    def __init__(self, size: int, generator: torch.Generator) -> None:
        if size < 1:
            raise ValueError("Sampler size must be positive.")
        self.size = int(size)
        self.generator = generator
        self.epoch = 0
        self.position = 0
        self.order: list[int] = []

    def __iter__(self) -> Iterator[int]:
        if not self.order or self.position >= len(self.order):
            self.order = torch.randperm(
                self.size, generator=self.generator
            ).tolist()
            self.position = 0
            self.epoch += 1
        while self.position < len(self.order):
            value = int(self.order[self.position])
            self.position += 1
            yield value

    def __len__(self) -> int:
        return self.size - self.position if self.order else self.size

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "epoch": self.epoch,
            "position": self.position,
            "order": list(self.order),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("size", -1)) != self.size:
            raise ValueError("Sampler size differs from checkpoint.")
        order = [int(value) for value in payload.get("order", ())]
        position = int(payload.get("position", -1))
        epoch = int(payload.get("epoch", -1))
        if (
            (order and sorted(order) != list(range(self.size)))
            or not 0 <= position <= len(order)
            or epoch < 0
        ):
            raise ValueError("Checkpoint sampler state is invalid.")
        self.order = order
        self.position = position
        self.epoch = epoch


def _next_virtual_batch(
    loader: DataLoader[tuple[torch.Tensor]],
    iterator: Iterator[tuple[torch.Tensor]],
) -> tuple[torch.Tensor, Iterator[tuple[torch.Tensor]]]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch[0], iterator


def _prepare_virtual_source_inputs(
    config: TargetPilotTrainConfig,
) -> tuple[torch.Tensor, tuple[int, ...], dict[str, Any]]:
    prepared = prepare_source_1d(
        config.source_data_path,
        config.source_split_manifest,
        time_stride=2,
    )
    if tuple(prepared.channel_names) != tuple(INPUT_CHANNELS):
        raise ValueError("Virtual source channel order violates P3.")
    checkpoint = torch.load(
        config.source_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_normalization = checkpoint.get("normalization")
    if checkpoint_normalization != prepared.normalization:
        raise ValueError(
            "Virtual source inputs do not use the checkpoint's exact frozen "
            "normalization."
        )
    training_indices = tuple(int(value) for value in prepared.splits["train"])
    if not training_indices:
        raise ValueError("Virtual source training split is empty.")
    inputs = prepared.inputs[torch.tensor(training_indices, dtype=torch.long)]
    metadata = {
        "split": "source_train_only",
        "case_ids": [
            int(value)
            for value in prepared.case_ids[
                torch.tensor(training_indices, dtype=torch.long)
            ]
        ],
        "labels_loaded_into_preparation_but_used_by_restriction_loss": False,
        "source_teacher_predictions_used": False,
        "target_labels_used": False,
        "target_suffix": "six_exact_zero_channels",
    }
    return inputs, training_indices, metadata


def _resource_contract(
    config: TargetPilotTrainConfig, checksums: Mapping[str, Any]
) -> dict[str, Any]:
    implementation = _implementation_state(config.project_root)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "architecture": _model_spec_from_config(config),
        "effective_batch_size": config.effective_batch_size,
        "candidate_micro_batch_sizes": list(
            config.preflight_candidate_batch_sizes
        ),
        "full_resolution_required": True,
        "mixed_precision": False,
        "full_t2_trainability": True,
        "implementation_source_file_sha256": implementation[
            "source_file_sha256"
        ],
        "implementation_sha256": implementation[
            "implementation_sha256"
        ],
        "optimizer": "AdamW",
        "parameter_group_lr_ratios": dict(PARAMETER_GROUP_RATIOS),
        "loss_weights": {
            "temperature_relative_l2": config.temperature_weight,
            "alpha_relative_l2": config.alpha_weight,
            "gradient_x_relative_l2": config.gradient_x_weight,
            "gradient_z_relative_l2": config.gradient_z_weight,
            "homogeneous_lateral_invariance": config.restriction_weight,
        },
        "restriction_validation_case_count": (
            config.restriction_validation_case_count
        ),
        "restriction_validation_thresholds": {
            "lateral_invariance_score_max": (
                config.restriction_lateral_invariance_max
            ),
            "maximum_lateral_range_max": (
                config.restriction_lateral_range_max
            ),
        },
        "input_sha256": {
            name: value["sha256"]
            for name, value in checksums.items()
            if isinstance(value, dict) and "sha256" in value
        },
    }


def _resolve_resources_from_profile(
    config: TargetPilotTrainConfig,
    checksums: Mapping[str, Any],
    device: torch.device,
) -> tuple[TargetPilotTrainConfig, dict[str, Any] | None]:
    if not config.require_resource_profile:
        return config.validated(require_resolved_resources=True), None
    path = config.resource_profile_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Frozen CUDA resource profile is missing. Run the target trainer "
            "with --resource-preflight before either pilot method."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        profile.get("schema_version") != RESOURCE_PROFILE_SCHEMA_VERSION
        or profile.get("experiment") != EXPERIMENT
        or profile.get("passed") is not True
    ):
        raise ValueError("Resource profile schema, experiment, or status is invalid.")
    contract = _resource_contract(config, checksums)
    if profile.get("resource_contract_sha256") != _sha256_json(contract):
        raise ValueError(
            "Resource profile does not match the current model/data/loss contract."
        )
    current_device = _device_payload(device, config.device)
    profiled_device = profile.get("device")
    for key in ("resolved", "torch", "cuda_runtime"):
        if profiled_device.get(key) != current_device.get(key):
            raise ValueError(f"Resource profile device field {key} changed.")
    if profiled_device.get("active_device") != current_device.get(
        "active_device"
    ):
        raise ValueError("Active GPU differs from the frozen resource profile.")
    micro = int(profile["selected_micro_batch_size"])
    accumulation = int(profile["gradient_accumulation_steps"])
    resolved = replace(
        config,
        micro_batch_size=micro,
        gradient_accumulation_steps=accumulation,
    ).validated(require_resolved_resources=True)
    return resolved, profile


def run_target_resource_preflight(
    config: TargetPilotTrainConfig,
    *,
    overwrite: bool = False,
    prepared_override: PreparedTarget2DTraining | None = None,
    virtual_inputs_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run one full-resolution CUDA optimizer step and freeze resources."""

    config = config.validated()
    device = _resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError(
            "The canonical P5 resource preflight must run on CUDA."
        )
    output_path = config.resource_profile_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite resource profile: {output_path}"
        )
    _configure_determinism(config.seed)
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    torch.cuda.set_device(device)
    checksums = _input_checksums(config)
    prepared = prepared_override or prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=max(PILOT_BUDGETS),
        project_root=config.project_root,
        verify_array_checksums=config.verify_array_checksums,
    )
    # The shared array files are checksum-read and memory-mapped by the data
    # contract. Only one training case's label values are indexed/materialized;
    # validation/test/OOD label values are not.
    sample = prepared.dataset("train")[0]
    virtual_inputs, _, virtual_metadata = (
        (virtual_inputs_override, (), {"injected": True})
        if virtual_inputs_override is not None
        else _prepare_virtual_source_inputs(config)
    )
    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    selected_peak_allocated: int | None = None
    selected_peak_reserved: int | None = None
    full_shape = list(sample[0].shape)
    nx = int(sample[0].shape[2])
    for candidate in config.preflight_candidate_batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            _configure_determinism(config.seed)
            scratch_config = replace(config, method="scratch_ffno")
            model, _, _ = _build_initialized_model(scratch_config, device)
            parameter_groups, _ = build_parameter_groups(
                model,
                base_learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer = torch.optim.AdamW(parameter_groups)
            inputs = sample[0].unsqueeze(0).repeat(
                candidate, 1, 1, 1, 1
            ).to(device)
            temperature = sample[1].unsqueeze(0).repeat(
                candidate, 1, 1, 1
            ).to(device)
            alpha = sample[2].unsqueeze(0).repeat(
                candidate, 1, 1, 1
            ).to(device)
            source_values = virtual_inputs[:candidate]
            if len(source_values) < candidate:
                repeats = math.ceil(candidate / max(len(source_values), 1))
                source_values = virtual_inputs.repeat(
                    repeats, 1, 1, 1
                )[:candidate]
            source_values = source_values.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            components = target_loss_components(
                outputs, temperature, alpha, inputs[..., 4]
            )
            restriction, _ = homogeneous_restriction_loss(
                model, source_values, nx=nx
            )
            objective = (
                weighted_validation_objective(components, config)
                + config.restriction_weight * restriction
            )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            attempts.append(
                {
                    "micro_batch_size": candidate,
                    "status": "passed",
                    "optimizer_step_completed": True,
                    "peak_memory_allocated_bytes": peak_allocated,
                    "peak_memory_reserved_bytes": peak_reserved,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            selected = candidate
            selected_peak_allocated = peak_allocated
            selected_peak_reserved = peak_reserved
            del model, optimizer, inputs, temperature, alpha, outputs
            break
        except torch.OutOfMemoryError as error:
            attempts.append(
                {
                    "micro_batch_size": candidate,
                    "status": "cuda_out_of_memory",
                    "error": str(error),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        finally:
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError(
            "No configured full-resolution micro-batch fits the active GPU."
        )
    accumulation = config.effective_batch_size // selected
    contract = _resource_contract(config, checksums)
    profile = {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "phase": "P5",
        "experiment": EXPERIMENT,
        "profile_role": "method_independent_full_resolution_cuda_preflight",
        "passed": True,
        "resource_contract": contract,
        "resource_contract_sha256": _sha256_json(contract),
        "device": _device_payload(device, config.device),
        "full_resolution_input_shape_without_batch": full_shape,
        "selected_micro_batch_size": selected,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": config.effective_batch_size,
        "selected_peak_memory_allocated_bytes": selected_peak_allocated,
        "selected_peak_memory_reserved_bytes": selected_peak_reserved,
        "attempts": attempts,
        "preflight_training_case_id": int(sample[3]),
        "materialized_target_label_splits": ["train"],
        "target_test_or_ood_label_values_indexed_or_materialized": False,
        "restriction_virtual_input": virtual_metadata,
        "initialization_used_for_memory_preflight": "scratch_ffno",
        "architecture_and_trainability_identical_between_methods": True,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    _atomic_write_text(output_path, _canonical_json(profile))
    return profile


def _decode_temperature(
    values: np.ndarray, normalization: Mapping[str, Any]
) -> np.ndarray:
    metadata = normalization["field_temperature"]
    return (
        values * (float(metadata["maximum"]) - float(metadata["minimum"]))
        + float(metadata["minimum"])
    )


def _evaluate_objective(
    model: AxisFactorized2DOperator,
    prepared: PreparedTarget2DTraining,
    config: TargetPilotTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        prepared.dataset("validation"),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    totals = {
        "temperature": 0.0,
        "alpha": 0.0,
        "gradient_x": 0.0,
        "gradient_z": 0.0,
        "weighted": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, temperature, alpha, _ in loader:
            inputs = inputs.to(
                device, non_blocking=device.type == "cuda"
            )
            temperature = temperature.to(
                device, non_blocking=device.type == "cuda"
            )
            alpha = alpha.to(device, non_blocking=device.type == "cuda")
            components = target_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            objective = weighted_validation_objective(components, config)
            batch_count = len(inputs)
            for name, value in components.items():
                totals[name] += float(value) * batch_count
            totals["weighted"] += float(objective) * batch_count
            count += batch_count
    if count != len(prepared.validation_case_ids):
        raise ValueError("Validation loader did not evaluate every frozen case.")
    return {name: value / count for name, value in totals.items()}


def _load_target_coordinates(
    config: TargetPilotTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    split = json.loads(
        config.target_split_manifest.read_text(encoding="utf-8")
    )
    source_path = (
        config.project_root / str(split["source_manifest"])
    ).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    artifact_root = (
        config.project_root / str(source["array_artifact_root"])
    ).resolve()
    z = np.load(artifact_root / "z_m.npy", allow_pickle=False)
    x = np.load(artifact_root / "x_m.npy", allow_pickle=False)
    if (
        z.ndim != 1
        or x.ndim != 1
        or len(z) < 2
        or len(x) < 2
        or not np.all(np.diff(z) > 0.0)
        or not np.all(np.diff(x) > 0.0)
    ):
        raise ValueError("Frozen target coordinates are invalid.")
    return np.asarray(z, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _relative_l2_numpy(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    error = (prediction - target)[mask]
    reference = target[mask]
    return float(
        np.linalg.norm(error)
        / max(np.linalg.norm(reference), np.finfo(np.float64).eps)
    )


def _evaluate_validation_metrics(
    model: AxisFactorized2DOperator,
    prepared: PreparedTarget2DTraining,
    config: TargetPilotTrainConfig,
    device: torch.device,
    *,
    coordinates_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    loader = DataLoader(
        prepared.dataset("validation"),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    z, x = (
        coordinates_override
        if coordinates_override is not None
        else _load_target_coordinates(config)
    )
    rows: list[dict[str, Any]] = []
    prediction_temperature_rows: list[np.ndarray] = []
    prediction_alpha_rows: list[np.ndarray] = []
    target_temperature_rows: list[np.ndarray] = []
    target_alpha_rows: list[np.ndarray] = []
    prediction_case_ids: list[int] = []
    canonical_composite_mask: np.ndarray | None = None
    model.eval()
    with torch.no_grad():
        for inputs, temperature, alpha, case_ids in loader:
            output = model(
                inputs.to(device, non_blocking=device.type == "cuda")
            )
            predicted_temperature = _decode_temperature(
                output["temperature"].cpu().numpy(),
                prepared.normalization,
            )
            true_temperature = _decode_temperature(
                temperature.numpy(), prepared.normalization
            )
            predicted_alpha = output["alpha"].cpu().numpy()
            true_alpha = alpha.numpy()
            masks = inputs[..., 4].numpy() > 0.5
            for index in range(len(inputs)):
                mask = masks[index]
                if not np.array_equal(
                    mask,
                    np.broadcast_to(mask[:1], mask.shape),
                ):
                    raise ValueError(
                        "Validation composite mask unexpectedly varies in time."
                    )
                case_mask = np.asarray(mask[0], dtype=np.bool_)
                if canonical_composite_mask is None:
                    canonical_composite_mask = case_mask.copy()
                elif not np.array_equal(
                    canonical_composite_mask, case_mask
                ):
                    raise ValueError(
                        "Validation cases do not share the frozen material mask."
                    )
                composite_mask = np.broadcast_to(
                    mask, predicted_temperature[index].shape
                )
                case_id = int(case_ids[index])
                temperature_prediction = predicted_temperature[index]
                temperature_target = true_temperature[index]
                alpha_prediction = predicted_alpha[index]
                alpha_target = true_alpha[index]
                predicted_gradient_x = np.diff(
                    temperature_prediction, axis=2
                ) / np.diff(x)[None, None, :]
                true_gradient_x = np.diff(
                    temperature_target, axis=2
                ) / np.diff(x)[None, None, :]
                predicted_gradient_z = np.diff(
                    temperature_prediction, axis=1
                ) / np.diff(z)[None, :, None]
                true_gradient_z = np.diff(
                    temperature_target, axis=1
                ) / np.diff(z)[None, :, None]
                mask_x = composite_mask[..., 1:] & composite_mask[..., :-1]
                mask_z = (
                    composite_mask[:, 1:, :] & composite_mask[:, :-1, :]
                )
                rows.append(
                    {
                        "split": "validation",
                        "case_id": case_id,
                        "temperature_relative_l2_K_composite": (
                            _relative_l2_numpy(
                                temperature_prediction,
                                temperature_target,
                                composite_mask,
                            )
                        ),
                        "temperature_mae_K_composite": float(
                            np.mean(
                                np.abs(
                                    temperature_prediction
                                    - temperature_target
                                )[composite_mask]
                            )
                        ),
                        "temperature_rmse_K_composite": float(
                            np.sqrt(
                                np.mean(
                                    (
                                        temperature_prediction
                                        - temperature_target
                                    )[composite_mask]
                                    ** 2
                                )
                            )
                        ),
                        "temperature_linf_K_composite": float(
                            np.max(
                                np.abs(
                                    temperature_prediction
                                    - temperature_target
                                )[composite_mask]
                            )
                        ),
                        "peak_temperature_absolute_error_K": float(
                            abs(
                                np.max(
                                    temperature_prediction[composite_mask]
                                )
                                - np.max(temperature_target[composite_mask])
                            )
                        ),
                        "alpha_relative_l2_composite": _relative_l2_numpy(
                            alpha_prediction,
                            alpha_target,
                            composite_mask,
                        ),
                        "alpha_mae_composite": float(
                            np.mean(
                                np.abs(alpha_prediction - alpha_target)[
                                    composite_mask
                                ]
                            )
                        ),
                        "temperature_gradient_x_relative_l2": (
                            _relative_l2_numpy(
                                predicted_gradient_x,
                                true_gradient_x,
                                mask_x,
                            )
                        ),
                        "temperature_gradient_z_relative_l2": (
                            _relative_l2_numpy(
                                predicted_gradient_z,
                                true_gradient_z,
                                mask_z,
                            )
                        ),
                        "maximum_lateral_gradient_error_K_per_m": float(
                            np.max(
                                np.abs(
                                    predicted_gradient_x
                                    - true_gradient_x
                                )[mask_x]
                            )
                        ),
                    }
                )
                prediction_temperature_rows.append(
                    temperature_prediction.astype(np.float32, copy=False)
                )
                prediction_alpha_rows.append(
                    alpha_prediction.astype(np.float32, copy=False)
                )
                target_temperature_rows.append(
                    temperature_target.astype(np.float32, copy=False)
                )
                target_alpha_rows.append(
                    alpha_target.astype(np.float32, copy=False)
                )
                prediction_case_ids.append(case_id)
    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    expected_ids = list(prepared.validation_case_ids)
    if frame["case_id"].tolist() != expected_ids:
        raise ValueError(
            "Per-case validation metrics do not match frozen case order."
        )
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"split", "case_id"}
    ]
    summary: dict[str, Any] = {"case_count": len(frame)}
    for column in metric_columns:
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_median"] = float(frame[column].median())
        summary[f"{column}_max"] = float(frame[column].max())
    predictions = {
        "case_ids": np.asarray(prediction_case_ids, dtype=np.int64),
        "composite_mask": np.asarray(
            canonical_composite_mask, dtype=np.bool_
        ),
        "x_m": np.asarray(x, dtype=np.float64),
        "z_m": np.asarray(z, dtype=np.float64),
        "temperature_prediction_K": np.stack(
            prediction_temperature_rows
        ),
        "temperature_target_K": np.stack(target_temperature_rows),
        "alpha_prediction": np.stack(prediction_alpha_rows),
        "alpha_target": np.stack(target_alpha_rows),
    }
    return summary, frame, predictions


def evaluate_restriction_validation(
    model: AxisFactorized2DOperator,
    virtual_inputs: torch.Tensor,
    *,
    nx: int,
    case_count: int,
    device: torch.device,
    case_ids: Sequence[int] | None = None,
    lateral_invariance_max: float = 1.0e-6,
    lateral_range_max: float = 1.0e-5,
) -> dict[str, Any]:
    """Audit the selected model on a frozen label-free homogeneous subset."""

    if case_count < 1 or case_count > len(virtual_inputs):
        raise ValueError(
            "restriction validation case_count is outside the virtual pool."
        )
    if (
        not np.isfinite(lateral_invariance_max)
        or lateral_invariance_max <= 0.0
        or not np.isfinite(lateral_range_max)
        or lateral_range_max <= 0.0
    ):
        raise ValueError("Restriction thresholds must be finite and positive.")
    source_values = virtual_inputs[:case_count].to(device)
    target_inputs = lift_homogeneous_source_input(source_values, nx)
    suffix = target_inputs[..., len(INPUT_CHANNELS) :]
    exact_zero_suffix = bool(torch.count_nonzero(suffix) == 0)
    model.eval()
    fields: dict[str, Any] = {}
    all_finite = True
    with torch.no_grad():
        outputs = model(target_inputs)
        for field in ("temperature", "alpha"):
            values = outputs[field]
            finite = bool(torch.all(torch.isfinite(values)))
            all_finite &= finite
            variance = torch.var(values, dim=3, unbiased=False)
            lateral_invariance = float(
                torch.sqrt(torch.mean(variance))
                / torch.std(values, unbiased=False).clamp_min(
                    torch.finfo(values.dtype).eps
                )
            )
            lateral_range = torch.amax(values, dim=3) - torch.amin(
                values, dim=3
            )
            fields[field] = {
                "finite": finite,
                "lateral_invariance_score": lateral_invariance,
                "maximum_lateral_range": float(torch.max(lateral_range)),
                "mean_lateral_range": float(torch.mean(lateral_range)),
                "checks": {
                    "lateral_invariance_score": (
                        lateral_invariance <= lateral_invariance_max
                    ),
                    "maximum_lateral_range": (
                        float(torch.max(lateral_range))
                        <= lateral_range_max
                    ),
                },
            }
    resolved_case_ids = (
        [int(value) for value in case_ids[:case_count]]
        if case_ids is not None and len(case_ids) >= case_count
        else list(range(case_count))
    )
    contract_checks = {
        "source_input_shape_is_canonical": (
            source_values.ndim == 4
            and source_values.shape[-1] == len(INPUT_CHANNELS)
        ),
        "target_suffix_is_exact_zero": exact_zero_suffix,
        "temperature_is_finite": fields["temperature"]["finite"],
        "alpha_is_finite": fields["alpha"]["finite"],
        "temperature_lateral_invariance_score": fields["temperature"][
            "checks"
        ]["lateral_invariance_score"],
        "temperature_maximum_lateral_range": fields["temperature"][
            "checks"
        ]["maximum_lateral_range"],
        "alpha_lateral_invariance_score": fields["alpha"]["checks"][
            "lateral_invariance_score"
        ],
        "alpha_maximum_lateral_range": fields["alpha"]["checks"][
            "maximum_lateral_range"
        ],
        "target_labels_absent": True,
        "source_teacher_predictions_absent": True,
    }
    passed = bool(all(contract_checks.values()) and all_finite)
    return {
        "schema_version": 1,
        "audit": "selected_model_label_free_homogeneous_restriction",
        "selection": "first_frozen_source_train_cases_in_manifest_order",
        "case_count": case_count,
        "case_ids": resolved_case_ids,
        "source_input_shape": list(source_values.shape),
        "target_input_shape": list(target_inputs.shape),
        "nx": nx,
        "fields": fields,
        "contract_checks": contract_checks,
        "provenance": {
            "target_labels_used": False,
            "source_teacher_predictions_used": False,
        },
        "thresholds": {
            "lateral_invariance_score_max": lateral_invariance_max,
            "maximum_lateral_range_max": lateral_range_max,
        },
        "threshold_note": (
            "Thresholds were frozen in the pilot YAML and protocol before "
            "either target method trained."
        ),
        "passed": passed,
    }


def _checkpoint_payload(
    model: AxisFactorized2DOperator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    best_validation: float,
    best_epoch: int,
    best_model_state: Mapping[str, torch.Tensor],
    bad_epochs: int,
    train_generator: torch.Generator,
    virtual_generator: torch.Generator,
    train_sampler: StatefulShuffleSampler,
    virtual_sampler: StatefulShuffleSampler,
    config: TargetPilotTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
    parameter_groups: Mapping[str, Any],
    initialization: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
    history_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen_best_state = _clone_state_to_cpu(best_model_state)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "P5",
        "experiment": EXPERIMENT,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "model_family": MODEL_FAMILY,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "best_model": frozen_best_state,
        "best_model_sha256": _state_dict_sha256(frozen_best_state),
        "bad_epochs": bad_epochs,
        "train_generator_state": train_generator.get_state(),
        "virtual_generator_state": virtual_generator.get_state(),
        "train_sampler_state": train_sampler.state_dict(),
        "virtual_sampler_state": virtual_sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda"
            else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "scientific_config": _scientific_config(config),
        "input_checksums": dict(checksums),
        "target_data_checksums": prepared.checksums,
        "normalization": prepared.normalization,
        "train_case_ids": tuple(prepared.train_case_ids),
        "validation_case_ids": tuple(prepared.validation_case_ids),
        "channel_names": tuple(TARGET_INPUT_CHANNELS),
        "selection_split": "validation",
        "selection_uses_test_or_ood_labels": False,
        "implementation": dict(implementation),
        "runtime_fingerprint": dict(runtime_fingerprint),
        "parameter_groups": dict(parameter_groups),
        "initialization": dict(initialization),
        "resource_profile_sha256": (
            None
            if resource_profile is None
            else _sha256_json(resource_profile)
        ),
        # The last checkpoint is the authoritative epoch transaction.  The
        # user-facing parquet history and best.pt are reproducible derivatives.
        "history_rows": [dict(row) for row in history_rows],
    }


def _best_artifact_payload(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    best_model = checkpoint["best_model"]
    if _state_dict_sha256(best_model) != checkpoint["best_model_sha256"]:
        raise ValueError("Authoritative best-model digest is invalid.")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "P5",
        "experiment": EXPERIMENT,
        "method": checkpoint["method"],
        "label_budget": checkpoint["label_budget"],
        "seed": checkpoint["seed"],
        "model_family": checkpoint["model_family"],
        "model": best_model,
        "epoch": int(checkpoint["best_epoch"]),
        "best_validation": float(checkpoint["best_validation"]),
        "scientific_config": checkpoint["scientific_config"],
        "input_checksums": checkpoint["input_checksums"],
        "target_data_checksums": checkpoint["target_data_checksums"],
        "normalization": checkpoint["normalization"],
        "channel_names": checkpoint["channel_names"],
        "selection_split": checkpoint["selection_split"],
        "selection_uses_test_or_ood_labels": False,
        "git_sha": checkpoint["implementation"]["head_sha"],
        "implementation_sha256": checkpoint["implementation"][
            "implementation_sha256"
        ],
        "runtime_fingerprint": checkpoint["runtime_fingerprint"],
        "model_sha256": checkpoint["best_model_sha256"],
        "derived_from_last_epoch": int(checkpoint["epoch"]),
    }


def _history_frame(
    history_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame([dict(row) for row in history_rows])
    if "learning_rates" in frame:
        frame["learning_rates"] = frame["learning_rates"].map(
            lambda value: (
                value
                if isinstance(value, str)
                else json.dumps(value, sort_keys=True)
            )
        )
    return frame


def _restore_derived_epoch_artifacts(
    checkpoint: Mapping[str, Any],
    *,
    best_path: Path,
    history_path: Path,
    recovery_log_path: Path,
) -> None:
    """Rebuild best/history atomically from the authoritative last checkpoint."""

    expected_best = _best_artifact_payload(checkpoint)
    best_needs_rebuild = True
    if best_path.is_file():
        try:
            actual_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_needs_rebuild = not (
                isinstance(actual_best, Mapping)
                and actual_best.get("schema_version")
                == CHECKPOINT_SCHEMA_VERSION
                and actual_best.get("scientific_config")
                == checkpoint["scientific_config"]
                and actual_best.get("input_checksums")
                == checkpoint["input_checksums"]
                and int(actual_best.get("epoch", -1))
                == int(checkpoint["best_epoch"])
                and actual_best.get("model_sha256")
                == checkpoint["best_model_sha256"]
                and _state_dict_sha256(actual_best.get("model", {}))
                == checkpoint["best_model_sha256"]
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            best_needs_rebuild = True
    expected_history_rows = list(checkpoint["history_rows"])
    history_needs_rebuild = True
    if history_path.is_file():
        try:
            actual_history = pd.read_parquet(history_path).to_dict(
                orient="records"
            )
            for row in actual_history:
                if isinstance(row.get("learning_rates"), str):
                    row["learning_rates"] = json.loads(
                        row["learning_rates"]
                    )
            history_needs_rebuild = (
                actual_history != expected_history_rows
            )
        except (OSError, ValueError, json.JSONDecodeError):
            history_needs_rebuild = True
    if best_needs_rebuild:
        _atomic_torch_save(expected_best, best_path)
    if history_needs_rebuild:
        _atomic_parquet(
            _history_frame(expected_history_rows), history_path
        )
    if best_needs_rebuild or history_needs_rebuild:
        _append_jsonl(
            recovery_log_path,
            {
                "event": "derived_epoch_artifacts_rebuilt",
                "timestamp": datetime.now().astimezone().isoformat(),
                "authoritative_last_epoch": int(checkpoint["epoch"]),
                "best_rebuilt": best_needs_rebuild,
                "history_rebuilt": history_needs_rebuild,
            },
        )


def _restore_rng(checkpoint: Mapping[str, Any]) -> None:
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from checkpoint RNG state."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def make_target_pilot_run_id(
    config: TargetPilotTrainConfig, git_sha: str
) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    method = config.method.replace("_", "-")
    return (
        f"{timestamp}__P5__{method}__p4-2d-core-v1__"
        f"budget{config.label_budget}__seed{config.seed}__{git_sha[:7]}"
    )


def _write_provenance(
    run_dir: Path,
    config: TargetPilotTrainConfig,
    device: torch.device,
    checksums: Mapping[str, Any],
    implementation: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    resource_profile: Mapping[str, Any] | None,
) -> None:
    _atomic_write_text(
        run_dir / "config_resolved.json",
        _canonical_json(
            {
                "schema_version": 1,
                "phase": "P5",
                "experiment": EXPERIMENT,
                "scientific_config": _scientific_config(config),
                "comparison_contract": {
                    "allowed_methods": list(PILOT_METHODS),
                    "allowed_budgets": list(PILOT_BUDGETS),
                    "allowed_seeds": list(PILOT_SEEDS),
                    "methods_differ_only_by_initialization": True,
                    "transfer_stage": "T2",
                    "parameter_group_lr_ratios": PARAMETER_GROUP_RATIOS,
                    "selection_split": "validation",
                    "test_or_ood_selection": False,
                },
            }
        ),
    )
    _atomic_write_text(
        run_dir / "data_checksums.json",
        _canonical_json(
            {
                "input_files": dict(checksums),
                "prepared_target_contract": prepared.checksums,
                "train_case_ids": list(prepared.train_case_ids),
                "validation_case_ids": list(prepared.validation_case_ids),
            }
        ),
    )
    _atomic_write_text(
        run_dir / "git_state.txt",
        (
            f"HEAD {implementation['head_sha']}\n"
            f"implementation_sha256 "
            f"{implementation['implementation_sha256']}\n\n"
            f"status --short\n{implementation['status_short']}\n"
        ),
    )
    environment = {
        "schema_version": 1,
        **dict(runtime_fingerprint),
        "command": sys.argv,
        "resource_profile": resource_profile,
    }
    _atomic_write_text(
        run_dir / "environment.json", _canonical_json(environment)
    )


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _train_impl(
    config: TargetPilotTrainConfig,
    *,
    session_epoch_limit: int | None,
    prepared_override: PreparedTarget2DTraining | None,
    virtual_inputs_override: torch.Tensor | None,
    coordinates_override: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    implementation = _implementation_state(project_root)
    git_sha = str(implementation["head_sha"])
    run_id = config.run_id or make_target_pilot_run_id(config, git_sha)
    config = replace(config, run_id=run_id)
    run_dir = config.output_root.resolve() / run_id
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    history_path = run_dir / "history.parquet"
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
    if config.resume and (run_dir / "DONE").is_file():
        raise RuntimeError(f"Completed run cannot be resumed: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"cdcureno.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        run_dir / "stdout.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    session_started_at = datetime.now().astimezone().isoformat()
    started_path = run_dir / "STARTED_AT"
    if config.resume and started_path.is_file():
        started_at = started_path.read_text(encoding="utf-8").strip()
    else:
        started_at = session_started_at
        _atomic_write_text(started_path, f"{started_at}\n")
    sessions_path = run_dir / "run_sessions.jsonl"
    _append_jsonl(
        sessions_path,
        {
            "event": "session_started",
            "timestamp": session_started_at,
            "resume": config.resume,
        },
    )
    _atomic_write_text(
        run_dir / "STATUS.json",
        _canonical_json(
            {"status": "running", "timestamp": session_started_at}
        ),
    )

    wall_started = time.perf_counter()
    tracemalloc.start()
    _configure_determinism(config.seed)
    device = _resolve_device(config.device)
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    runtime_fingerprint = _runtime_fingerprint(device, config.device)

    checksums = _input_checksums(config)
    config, resource_profile = _resolve_resources_from_profile(
        config, checksums, device
    )
    prepared = prepared_override or prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=config.label_budget,
        project_root=project_root,
        verify_array_checksums=config.verify_array_checksums,
    )
    if prepared.label_budget != config.label_budget:
        raise ValueError("Prepared target label budget differs from run config.")
    if tuple(prepared.channel_names) != tuple(TARGET_INPUT_CHANNELS):
        raise ValueError("Prepared target channel order violates P5.")
    virtual_inputs, _, virtual_metadata = (
        (virtual_inputs_override, (), {"injected": True})
        if virtual_inputs_override is not None
        else _prepare_virtual_source_inputs(config)
    )
    if virtual_inputs is None or len(virtual_inputs) < 1:
        raise ValueError("Restriction virtual-input pool is empty.")

    if not config.resume:
        _write_provenance(
            run_dir,
            config,
            device,
            checksums,
            implementation,
            runtime_fingerprint,
            prepared,
            resource_profile,
        )
        _atomic_write_text(
            run_dir / "restriction_virtual_input.json",
            _canonical_json(virtual_metadata),
        )
    else:
        frozen = json.loads(
            (run_dir / "config_resolved.json").read_text(encoding="utf-8")
        )
        if frozen["scientific_config"] != _scientific_config(config):
            raise ValueError(
                "Resume configuration differs from the frozen run."
            )
        frozen_checksums = json.loads(
            (run_dir / "data_checksums.json").read_text(encoding="utf-8")
        )
        if frozen_checksums["input_files"] != checksums:
            raise ValueError("Resume input checksums differ from frozen files.")
        if (
            frozen_checksums["prepared_target_contract"]
            != prepared.checksums
        ):
            raise ValueError(
                "Resume prepared target checksums differ from frozen contract."
            )

    model, model_spec, initialization = _build_initialized_model(
        config, device
    )
    optimizer_groups, parameter_group_report = build_parameter_groups(
        model,
        base_learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    train_generator = torch.Generator().manual_seed(config.seed + 10_001)
    virtual_generator = torch.Generator().manual_seed(config.seed + 20_003)
    train_dataset = prepared.dataset("train")
    virtual_dataset = TensorDataset(virtual_inputs)
    train_sampler = StatefulShuffleSampler(
        len(train_dataset), train_generator
    )
    virtual_sampler = StatefulShuffleSampler(
        len(virtual_dataset), virtual_generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    virtual_loader = DataLoader(
        virtual_dataset,
        batch_size=config.micro_batch_size,
        sampler=virtual_sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    history_rows: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    if config.resume:
        if not last_path.is_file():
            raise FileNotFoundError("Resume requires authoritative last.pt.")
        checkpoint = torch.load(
            last_path, map_location=device, weights_only=False
        )
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported P5 target checkpoint schema.")
        if checkpoint["scientific_config"] != _scientific_config(config):
            raise ValueError(
                "Checkpoint scientific config differs from this run."
            )
        if checkpoint["input_checksums"] != checksums:
            raise ValueError(
                "Checkpoint input checksums differ from current files."
            )
        checkpoint_implementation = checkpoint["implementation"]
        for name in ("head_sha", "implementation_sha256"):
            if checkpoint_implementation[name] != implementation[name]:
                raise ValueError(
                    f"Resume implementation field {name} changed."
                )
        if checkpoint.get("runtime_fingerprint") != runtime_fingerprint:
            raise ValueError(
                "Resume runtime, deterministic settings, or GPU changed."
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_model_state = _clone_state_to_cpu(
            checkpoint["best_model"]
        )
        if (
            _state_dict_sha256(best_model_state)
            != checkpoint["best_model_sha256"]
        ):
            raise ValueError(
                "Checkpoint authoritative best-model digest is invalid."
            )
        train_generator.set_state(checkpoint["train_generator_state"])
        virtual_generator.set_state(
            checkpoint["virtual_generator_state"]
        )
        train_sampler.load_state_dict(
            checkpoint["train_sampler_state"]
        )
        virtual_sampler.load_state_dict(
            checkpoint["virtual_sampler_state"]
        )
        _restore_rng(checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history_rows = [
            dict(row) for row in checkpoint["history_rows"]
        ]
        if len(history_rows) != int(checkpoint["epoch"]):
            raise ValueError(
                "Checkpoint history length differs from its epoch."
            )
        _restore_derived_epoch_artifacts(
            checkpoint,
            best_path=best_path,
            history_path=history_path,
            recovery_log_path=run_dir / "artifact_recovery.jsonl",
        )
        if [int(row["epoch"]) for row in history_rows] != list(
            range(1, start_epoch)
        ):
            raise ValueError(
                "Resume history is not contiguous with last.pt."
            )

    executed_this_session = 0
    stopped_early = False
    virtual_iterator = iter(virtual_loader)
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = {
            "temperature": 0.0,
            "alpha": 0.0,
            "gradient_x": 0.0,
            "gradient_z": 0.0,
            "restriction": 0.0,
            "weighted": 0.0,
        }
        train_count = 0
        optimizer_steps = 0
        last_gradient_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        micro_batches = list(train_loader)
        if not micro_batches:
            raise ValueError("Target training loader is empty.")
        for micro_index, (
            inputs,
            temperature,
            alpha,
            _,
        ) in enumerate(micro_batches, start=1):
            inputs = inputs.to(
                device, non_blocking=device.type == "cuda"
            )
            temperature = temperature.to(
                device, non_blocking=device.type == "cuda"
            )
            alpha = alpha.to(
                device, non_blocking=device.type == "cuda"
            )
            source_values, virtual_iterator = _next_virtual_batch(
                virtual_loader, virtual_iterator
            )
            source_values = source_values.to(
                device, non_blocking=device.type == "cuda"
            )
            components = target_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            restriction, _ = homogeneous_restriction_loss(
                model, source_values, nx=int(inputs.shape[3])
            )
            objective = (
                weighted_validation_objective(components, config)
                + config.restriction_weight * restriction
            )
            loss = objective / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}."
                )
            loss.backward()
            is_last = micro_index == len(micro_batches)
            accumulation_boundary = (
                micro_index % config.gradient_accumulation_steps == 0
            )
            if accumulation_boundary or is_last:
                last_gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                )
                if not np.isfinite(last_gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch {epoch}."
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            batch_count = len(inputs)
            for name, value in components.items():
                totals[name] += float(value.detach()) * batch_count
            totals["restriction"] += (
                float(restriction.detach()) * batch_count
            )
            totals["weighted"] += (
                float(objective.detach()) * batch_count
            )
            train_count += batch_count
        scheduler.step()
        validation = _evaluate_objective(
            model, prepared, config, device
        )
        if not all(np.isfinite(value) for value in validation.values()):
            raise FloatingPointError(
                f"Non-finite validation metric at epoch {epoch}."
            )
        improved = validation["weighted"] < best_validation
        if improved:
            best_validation = validation["weighted"]
            best_epoch = epoch
            best_model_state = _clone_state_to_cpu(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if best_model_state is None or best_epoch < 1:
            raise AssertionError(
                "The first finite validation epoch must select a best model."
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "epoch": epoch,
            "train_temperature_relative_l2": (
                totals["temperature"] / train_count
            ),
            "train_alpha_relative_l2": totals["alpha"] / train_count,
            "train_gradient_x_relative_l2": (
                totals["gradient_x"] / train_count
            ),
            "train_gradient_z_relative_l2": (
                totals["gradient_z"] / train_count
            ),
            "train_homogeneous_restriction": (
                totals["restriction"] / train_count
            ),
            "train_weighted_objective": (
                totals["weighted"] / train_count
            ),
            "validation_temperature_relative_l2": (
                validation["temperature"]
            ),
            "validation_alpha_relative_l2": validation["alpha"],
            "validation_gradient_x_relative_l2": (
                validation["gradient_x"]
            ),
            "validation_gradient_z_relative_l2": (
                validation["gradient_z"]
            ),
            "validation_weighted_objective": validation["weighted"],
            "optimizer_steps": optimizer_steps,
            "gradient_norm_before_clip_last_step": last_gradient_norm,
            "learning_rates": {
                str(group["group_name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "duration_seconds": time.perf_counter() - epoch_started,
            "improved": improved,
        }
        history_rows.append(row)
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_validation=best_validation,
            best_epoch=best_epoch,
            best_model_state=best_model_state,
            bad_epochs=bad_epochs,
            train_generator=train_generator,
            virtual_generator=virtual_generator,
            train_sampler=train_sampler,
            virtual_sampler=virtual_sampler,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime_fingerprint=runtime_fingerprint,
            parameter_groups=parameter_group_report,
            initialization=initialization,
            resource_profile=resource_profile,
            history_rows=history_rows,
        )
        _atomic_torch_save(checkpoint, last_path)
        _restore_derived_epoch_artifacts(
            checkpoint,
            best_path=best_path,
            history_path=history_path,
            recovery_log_path=run_dir / "artifact_recovery.jsonl",
        )
        executed_this_session += 1
        logger.info(
            "epoch=%d/%d train=%.6f validation=%.6f steps=%d seconds=%.3f",
            epoch,
            config.epochs,
            row["train_weighted_objective"],
            row["validation_weighted_objective"],
            optimizer_steps,
            row["duration_seconds"],
        )
        if (
            epoch >= config.minimum_epochs
            and bad_epochs >= config.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "early_stop epoch=%d bad_epochs=%d", epoch, bad_epochs
            )
            break
        if (
            session_epoch_limit is not None
            and executed_this_session >= session_epoch_limit
            and epoch < config.epochs
        ):
            paused_at = datetime.now().astimezone().isoformat()
            _append_jsonl(
                sessions_path,
                {
                    "event": "session_paused",
                    "timestamp": paused_at,
                    "last_epoch": epoch,
                },
            )
            _atomic_write_text(
                run_dir / "STATUS.json",
                _canonical_json(
                    {
                        "status": "paused",
                        "timestamp": paused_at,
                        "last_epoch": epoch,
                    }
                ),
            )
            _, peak_python = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _close_logger(logger)
            return {
                "status": "paused",
                "run_id": run_id,
                "last_epoch": epoch,
                "target_test_or_ood_labels_evaluated": False,
                "peak_python_tracemalloc_bytes": peak_python,
            }

    if not last_path.is_file():
        raise RuntimeError(
            "Training completed without an authoritative last checkpoint."
        )
    authority = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    _restore_derived_epoch_artifacts(
        authority,
        best_path=best_path,
        history_path=history_path,
        recovery_log_path=run_dir / "artifact_recovery.jsonl",
    )
    selected = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(selected["model"], strict=True)
    selected_epoch = int(selected["epoch"])
    validation_summary, per_case, predictions = (
        _evaluate_validation_metrics(
            model,
            prepared,
            config,
            device,
            coordinates_override=coordinates_override,
        )
    )
    per_case_path = run_dir / "metrics_per_case_validation.parquet"
    predictions_path = run_dir / "predictions" / "validation_best.npz"
    restriction_validation_path = run_dir / "restriction_validation.json"
    _atomic_parquet(per_case, per_case_path)
    _atomic_npz(predictions_path, **predictions)
    restriction_validation = evaluate_restriction_validation(
        model,
        virtual_inputs,
        nx=int(predictions["temperature_prediction_K"].shape[3]),
        case_count=config.restriction_validation_case_count,
        device=device,
        case_ids=(
            virtual_metadata.get("case_ids")
            if isinstance(virtual_metadata, Mapping)
            else None
        ),
        lateral_invariance_max=(
            config.restriction_lateral_invariance_max
        ),
        lateral_range_max=config.restriction_lateral_range_max,
    )
    _atomic_write_text(
        restriction_validation_path,
        _canonical_json(restriction_validation),
    )
    if not restriction_validation["passed"]:
        raise RuntimeError(
            "Selected-model homogeneous restriction contract audit failed; "
            f"evidence was preserved at {restriction_validation_path}."
        )
    if set(prepared.accessed_case_ids) != {"train", "validation"}:
        raise AssertionError("Unexpected target split access audit.")
    accessed = prepared.accessed_case_ids
    if not set(accessed["train"]).issubset(prepared.train_case_ids):
        raise PermissionError("Training accessed a case outside its budget.")
    if tuple(accessed["validation"]) != tuple(
        prepared.validation_case_ids
    ):
        raise PermissionError(
            "Validation access does not equal the frozen validation split."
        )

    completed_at = datetime.now().astimezone().isoformat()
    wall_seconds = time.perf_counter() - wall_started
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_accelerator = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    _append_jsonl(
        sessions_path,
        {
            "event": "session_completed",
            "timestamp": completed_at,
            "wall_seconds": wall_seconds,
        },
    )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "phase": "P5",
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "model": {
            **model_spec,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "transfer_stage": "T2",
        },
        "initialization": initialization,
        "parameter_groups": parameter_group_report,
        "selection": {
            "split": "validation",
            "objective": (
                "temperature_relL2_plus_alpha_relL2_plus_xz_gradient"
            ),
            "selected_epoch": selected_epoch,
            "best_validation_objective": float(
                selected["best_validation"]
            ),
            "target_test_or_ood_labels_used": False,
        },
        "training": {
            "configured_epochs": config.epochs,
            "executed_epochs": len(history_rows),
            "stopped_early": stopped_early,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
            "effective_batch_size": config.effective_batch_size,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "mixed_precision": False,
            "device": str(device),
            "summed_epoch_seconds": float(
                sum(float(row["duration_seconds"]) for row in history_rows)
            ),
            "all_session_wall_seconds": wall_seconds,
            "peak_accelerator_memory_bytes": peak_accelerator,
            "peak_python_tracemalloc_bytes": peak_python,
        },
        "loss_weights": {
            "temperature_relative_l2": config.temperature_weight,
            "alpha_relative_l2": config.alpha_weight,
            "gradient_x_relative_l2": config.gradient_x_weight,
            "gradient_z_relative_l2": config.gradient_z_weight,
            "homogeneous_lateral_invariance": config.restriction_weight,
        },
        "restriction_virtual_input": virtual_metadata,
        "restriction_validation": restriction_validation,
        "normalization": prepared.normalization,
        "input_checksums": checksums,
        "target_data_checksums": prepared.checksums,
        "target_label_access_audit": {
            "train": list(accessed["train"]),
            "validation": list(accessed["validation"]),
            "id_test": [],
            "ood": [],
        },
        "validation_metrics": validation_summary,
        "auditable_artifacts": {
            "history": _artifact_record(history_path, project_root),
            "validation_metrics_per_case": _artifact_record(
                per_case_path, project_root
            ),
            "validation_predictions": _artifact_record(
                predictions_path, project_root
            ),
            "restriction_validation": _artifact_record(
                restriction_validation_path, project_root
            ),
        },
        "checkpoints": {
            "best": {
                "path": _portable_path(best_path, project_root),
                "sha256": _sha256_file(best_path),
                "selected_epoch": selected_epoch,
            },
            "last": {
                "path": _portable_path(last_path, project_root),
                "sha256": _sha256_file(last_path),
                "epoch": int(
                    torch.load(
                        last_path,
                        map_location="cpu",
                        weights_only=False,
                    )["epoch"]
                ),
            },
        },
        "resource_profile": resource_profile,
        "started_at": started_at,
        "completed_at": completed_at,
        "git_sha": git_sha,
        "implementation_sha256": implementation[
            "implementation_sha256"
        ],
        # This individual run is evidence for the paired gate aggregator; it
        # cannot itself declare transfer superiority.
        "pilot_gate_decision": "requires_all_four_runs_and_paired_aggregator",
    }
    _atomic_write_text(run_dir / "metrics.json", _canonical_json(metrics))
    _atomic_write_text(
        run_dir / "STATUS.json",
        _canonical_json(
            {"status": "completed", "timestamp": completed_at}
        ),
    )
    _atomic_write_text(run_dir / "DONE", f"{completed_at}\n")
    failed_marker = run_dir / "FAILED"
    if failed_marker.is_file():
        os.replace(failed_marker, run_dir / "FAILED_RECOVERED")
    logger.info(
        "completed run_id=%s selected_epoch=%d validation_T=%.6f",
        run_id,
        selected_epoch,
        validation_summary[
            "temperature_relative_l2_K_composite_mean"
        ],
    )
    _close_logger(logger)
    return metrics


def train_target_pilot(
    config: TargetPilotTrainConfig,
    *,
    session_epoch_limit: int | None = None,
    prepared_override: PreparedTarget2DTraining | None = None,
    virtual_inputs_override: torch.Tensor | None = None,
    coordinates_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Train or resume one leakage-safe P5 RP-FFNO pilot run."""

    config = config.validated()
    if session_epoch_limit is not None and session_epoch_limit < 1:
        raise ValueError("session_epoch_limit must be positive.")
    implementation = _implementation_state(config.project_root)
    run_id = config.run_id or make_target_pilot_run_id(
        config, implementation["head_sha"]
    )
    resolved = replace(config, run_id=run_id)
    run_dir = resolved.output_root.resolve() / run_id
    if run_dir.exists() and not resolved.resume:
        raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
    try:
        return _train_impl(
            resolved,
            session_epoch_limit=session_epoch_limit,
            prepared_override=prepared_override,
            virtual_inputs_override=virtual_inputs_override,
            coordinates_override=coordinates_override,
        )
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        failed_at = datetime.now().astimezone().isoformat()
        _append_jsonl(
            run_dir / "failure_history.jsonl",
            {
                "event": "failure",
                "timestamp": failed_at,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _atomic_write_text(
            run_dir / "FAILED",
            f"{failed_at}\n{type(error).__name__}: {error}\n",
        )
        _atomic_write_text(
            run_dir / "STATUS.json",
            _canonical_json(
                {
                    "status": "failed",
                    "timestamp": failed_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
        )
        raise


def _resolve_repo_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (project_root / path).resolve()
    )


def load_target_pilot_config(
    path: Path,
    *,
    project_root: Path,
    method: str | None = None,
    label_budget: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    run_id: str | None = None,
    output_root: Path | None = None,
    epochs: int | None = None,
    resume: bool = False,
) -> TargetPilotTrainConfig:
    """Load and strictly validate the pre-registered P5 pilot YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P5 target pilot config must be a YAML mapping.")
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != "P5"
        or payload.get("experiment") != EXPERIMENT
    ):
        raise ValueError("Unsupported P5 target pilot config.")
    comparison = payload["comparison"]
    if (
        comparison.get("methods") != list(PILOT_METHODS)
        or comparison.get("budgets") != list(PILOT_BUDGETS)
        or comparison.get("seeds") != list(PILOT_SEEDS)
        or comparison.get("methods_differ_only_by_initialization") is not True
        or comparison.get("transfer_stage") != "T2"
        or comparison.get("test_or_ood_labels_used_for_selection") is not False
    ):
        raise ValueError("P5 YAML comparison declarations changed.")
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    loss = payload["loss"]
    resource = payload["resource_preflight"]
    outputs = payload["outputs"]
    if (
        model.get("parameter_group_lr_ratios")
        != PARAMETER_GROUP_RATIOS
        or training.get("mixed_precision") is not False
        or training.get("checkpoint_selection")
        != "frozen_validation_weighted_objective"
        or loss.get("temperature_region") != "composite_only"
        or loss.get("alpha_region") != "composite_only"
        or loss.get("gradient_axes") != ["x", "z"]
        or loss.get("restriction_virtual_input")
        != "label_free_homogeneous_source_train_inputs_only"
        or loss.get("source_teacher_predictions_used") is not False
        or resource.get("full_resolution") is not True
        or resource.get("device_type") != "cuda"
    ):
        raise ValueError("P5 YAML scientific declarations changed.")
    configured_output_root = (
        _resolve_repo_path(project_root, output_root)
        if output_root is not None
        else _resolve_repo_path(project_root, outputs["root"])
    )
    config = TargetPilotTrainConfig(
        project_root=project_root.resolve(),
        target_split_manifest=_resolve_repo_path(
            project_root, data["target_split_manifest"]
        ),
        source_checkpoint=_resolve_repo_path(
            project_root, data["source_checkpoint"]
        ),
        inflated_checkpoint=_resolve_repo_path(
            project_root, data["inflated_checkpoint"]
        ),
        target_model_config=_resolve_repo_path(
            project_root, model["config"]
        ),
        source_data_path=_resolve_repo_path(
            project_root, data["source_virtual_input_data"]
        ),
        source_split_manifest=_resolve_repo_path(
            project_root, data["source_virtual_input_split"]
        ),
        output_root=configured_output_root,
        resource_profile_path=_resolve_repo_path(
            project_root, resource["profile_path"]
        ),
        config_file=path.resolve(),
        run_id=run_id or outputs.get("run_id"),
        method=method or str(comparison["default_method"]),
        label_budget=int(
            label_budget
            if label_budget is not None
            else comparison["default_budget"]
        ),
        seed=int(
            seed if seed is not None else comparison["default_seed"]
        ),
        expected_target_split_sha256=data[
            "expected_target_split_sha256"
        ],
        expected_source_checkpoint_sha256=data[
            "expected_source_checkpoint_sha256"
        ],
        expected_inflated_checkpoint_sha256=data[
            "expected_inflated_checkpoint_sha256"
        ],
        expected_target_model_config_sha256=model[
            "expected_config_sha256"
        ],
        expected_source_data_sha256=data[
            "expected_source_virtual_input_data_sha256"
        ],
        expected_source_split_sha256=data[
            "expected_source_virtual_input_split_sha256"
        ],
        width=int(model["width"]),
        depth=int(model["depth"]),
        modes_time=int(model["modes_time"]),
        modes_z=int(model["modes_z"]),
        modes_x=int(model["modes_x"]),
        lateral_rank=int(model["lateral_rank"]),
        expected_parameter_count=int(model["expected_parameter_count"]),
        epochs=int(epochs if epochs is not None else training["epochs"]),
        minimum_epochs=int(training["minimum_epochs"]),
        early_stopping_patience=int(
            training["early_stopping_patience"]
        ),
        effective_batch_size=int(resource["effective_batch_size"]),
        preflight_candidate_batch_sizes=tuple(
            int(value)
            for value in resource["candidate_micro_batch_sizes"]
        ),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        temperature_weight=float(
            loss["weights"]["temperature_relative_l2"]
        ),
        alpha_weight=float(loss["weights"]["alpha_relative_l2"]),
        gradient_x_weight=float(
            loss["weights"]["gradient_x_relative_l2"]
        ),
        gradient_z_weight=float(
            loss["weights"]["gradient_z_relative_l2"]
        ),
        restriction_weight=float(
            loss["weights"]["homogeneous_lateral_invariance"]
        ),
        restriction_validation_case_count=int(
            loss["restriction_validation_case_count"]
        ),
        restriction_lateral_invariance_max=float(
            loss["restriction_validation_thresholds"][
                "lateral_invariance_score_max"
            ]
        ),
        restriction_lateral_range_max=float(
            loss["restriction_validation_thresholds"][
                "maximum_lateral_range_max"
            ]
        ),
        gradient_clip=float(training["gradient_clip"]),
        device=device or str(training["device"]),
        num_threads=int(training["num_threads"]),
        verify_array_checksums=bool(data["verify_array_checksums"]),
        require_resource_profile=True,
        resume=resume,
    )
    if epochs is not None and config.minimum_epochs > epochs:
        config = replace(config, minimum_epochs=epochs)
    return config.validated()


def target_pilot_dry_run(
    config: TargetPilotTrainConfig,
) -> dict[str, Any]:
    """Validate hashes, frozen data access, architecture, and resource lock."""

    config = config.validated()
    _configure_determinism(config.seed)
    checksums = _input_checksums(config)
    device = _resolve_device(config.device)
    prepared = prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=config.label_budget,
        project_root=config.project_root,
        verify_array_checksums=config.verify_array_checksums,
    )
    config_resolved: TargetPilotTrainConfig | None = None
    resource_profile: dict[str, Any] | None = None
    resource_error: str | None = None
    try:
        config_resolved, resource_profile = _resolve_resources_from_profile(
            config, checksums, device
        )
    except (FileNotFoundError, ValueError) as error:
        resource_error = f"{type(error).__name__}: {error}"
    model, model_spec, initialization = _build_initialized_model(
        config, torch.device("cpu")
    )
    _, groups = build_parameter_groups(
        model,
        base_learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checks = {
        "target_manifest_role_is_id_training_contract": True,
        "budget_is_exact_frozen_prefix": (
            len(prepared.train_case_ids) == config.label_budget
        ),
        "validation_is_frozen_32_cases": (
            tuple(prepared.validation_case_ids)
            == tuple(range(256, 288))
        ),
        "target_test_or_ood_loader_not_constructed": True,
        "target_label_values_not_indexed_during_dry_run": (
            prepared.accessed_case_ids
            == {"train": (), "validation": ()}
        ),
        "channel_contract_matches": (
            tuple(prepared.channel_names)
            == tuple(TARGET_INPUT_CHANNELS)
        ),
        "all_parameters_trainable_t2": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "parameter_count_matches": (
            groups["total_parameter_count"]
            == config.expected_parameter_count
        ),
        "resource_profile_valid": config_resolved is not None,
    }
    return {
        "schema_version": 1,
        "phase": "P5",
        "experiment": EXPERIMENT,
        "dry_run": True,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "model": model_spec,
        "initialization": initialization,
        "parameter_groups": groups,
        "train_case_ids": list(prepared.train_case_ids),
        "validation_case_ids": list(prepared.validation_case_ids),
        "input_checksums": checksums,
        "prepared_target_checksums": prepared.checksums,
        "resource_profile": resource_profile,
        "resource_profile_error": resource_error,
        "resolved_micro_batch_size": (
            None
            if config_resolved is None
            else config_resolved.micro_batch_size
        ),
        "resolved_gradient_accumulation_steps": (
            None
            if config_resolved is None
            else config_resolved.gradient_accumulation_steps
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
