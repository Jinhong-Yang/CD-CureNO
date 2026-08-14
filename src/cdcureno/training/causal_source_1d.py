"""Auditable, resumable training for the structurally causal P3 source model."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from cdcureno.data.source_1d import (
    INPUT_CHANNELS,
    PreparedSource1D,
    prepare_source_1d,
)
from cdcureno.legacy.audit import sha256_file
from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    parameter_count,
)
from cdcureno.training.joint import joint_loss_components
from cdcureno.training.source_1d import ACCEPTANCE as P3_SOURCE_ACCEPTANCE


ACCEPTANCE = dict(P3_SOURCE_ACCEPTANCE)
MODEL_FAMILY = "causal_factorized"
EXPERIMENT = "source_causal_cdcureno_v1"
CHECKPOINT_SCHEMA_VERSION = 2
IMPLEMENTATION_BINDING_SCHEMA_VERSION = 1
COMPUTE_TIMING_SCHEMA_VERSION = 1
HISTORY_TIMING_SCHEMA_VERSION = 1
COMPUTE_TIMING_PROTOCOL = "p3_causal_source_compute_timing"
OPTIMIZER_TIMING_CLOCK_SOURCE = "time.perf_counter"
CANDIDATE_TIMING_CLOCK_SOURCE = "time.perf_counter"
CUDA_TIMING_SYNCHRONIZATION = "torch.cuda.synchronize_before_and_after"
CPU_TIMING_SYNCHRONIZATION = "not_applicable_cpu"
COMPUTE_ACCOUNTING_FIELDS = (
    "parameter_count",
    "optimizer_update_count",
    "candidate_validation_seconds",
    "device_synchronized_training_seconds",
    "candidate_count",
    "timing_complete",
    "source_training_run_id",
)
COMPUTE_TIMING_CONFIG = {
    "schema_version": COMPUTE_TIMING_SCHEMA_VERSION,
    "clock_source": OPTIMIZER_TIMING_CLOCK_SOURCE,
    "cuda_device_synchronization": CUDA_TIMING_SYNCHRONIZATION,
    "cpu_device_synchronization": CPU_TIMING_SYNCHRONIZATION,
    "candidate_validation_separated": True,
    "legacy_epoch_duration_inference_allowed": False,
}
SCIENTIFIC_IMPLEMENTATION_FILES = (
    "src/cdcureno/data/joint_case1.py",
    "src/cdcureno/data/normalization.py",
    "src/cdcureno/data/source_1d.py",
    "src/cdcureno/models/joint_operators.py",
    "src/cdcureno/training/causal_source_1d.py",
    "src/cdcureno/training/joint.py",
    "src/cdcureno/training/source_1d.py",
)
_HISTORY_ROW_KEYS = {
    "epoch",
    "train_temperature_relative_l2",
    "train_temperature_normalized_l4",
    "train_alpha_relative_l2",
    "train_spatial_gradient_relative_l2",
    "train_weighted_objective",
    "validation_temperature_relative_l2",
    "validation_temperature_normalized_l4",
    "validation_alpha_relative_l2",
    "validation_spatial_gradient_relative_l2",
    "validation_weighted_objective",
    "gradient_norm_before_clip_last_batch",
    "learning_rate",
    "duration_seconds",
    "duration_seconds_semantics",
    "improved",
    "selected_best_epoch",
    "selected_best_validation_objective",
    "selected_best_model_sha256",
    "history_timing_schema_version",
    "parameter_count",
    "optimizer_update_count",
    "device_synchronized_training_seconds",
    "candidate_count",
    "candidate_validation_seconds",
    "optimizer_timing_clock_source",
    "optimizer_device_synchronization",
    "candidate_timing_clock_source",
    "candidate_device_synchronization",
    "candidate_validation_time_in_optimizer_total",
    "resume_invocation_count",
    "controlled_resume_invocation_count",
    "unobserved_interruption_resume_count",
    "replayed_optimizer_update_count",
    "replayed_optimizer_duration_seconds",
    "timing_complete",
}
_EPOCH_WALL_TIME_SEMANTICS = (
    "epoch_wall_including_training_validation_and_python_overhead_"
    "not_used_for_compute_accounting"
)


@dataclass(frozen=True)
class CausalSourceTrainConfig:
    """Frozen scientific and operational settings for one causal source run."""

    data_path: Path
    split_manifest: Path
    output_root: Path
    project_root: Path
    run_id: str | None = None
    config_file: Path | None = None
    expected_data_sha256: str | None = None
    expected_split_sha256: str | None = None
    seed: int = 0
    time_stride: int = 2
    input_channels: int = 14
    width: int = 34
    depth: int = 8
    modes_space: int = 12
    expected_parameter_count: int | None = 163_270
    epochs: int = 100
    minimum_epochs: int = 60
    early_stopping_patience: int = 25
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    temperature_weight: float = 1.0
    temperature_l4_weight: float = 0.2
    alpha_weight: float = 0.5
    gradient_weight: float = 0.1
    gradient_clip: float = 1.0
    device: str = "cuda"
    num_threads: int = 8
    causality_tolerance: float = 1.0e-7
    causality_case_count: int = 4
    causality_cutoff_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    compute_timing_schema_version: int = COMPUTE_TIMING_SCHEMA_VERSION
    resume: bool = False

    def validated(self) -> "CausalSourceTrainConfig":
        positive_integers = {
            "seed_plus_one": self.seed + 1,
            "time_stride": self.time_stride,
            "input_channels": self.input_channels,
            "width": self.width,
            "depth": self.depth,
            "modes_space": self.modes_space,
            "epochs": self.epochs,
            "minimum_epochs": self.minimum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "causality_case_count": self.causality_case_count,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"These settings must be positive: {invalid}.")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs.")
        positive_floats = {
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "causality_tolerance": self.causality_tolerance,
        }
        invalid_float = [
            name for name, value in positive_floats.items() if value <= 0.0
        ]
        if invalid_float:
            raise ValueError(f"These settings must be positive: {invalid_float}.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative.")
        weights = (
            self.temperature_weight,
            self.temperature_l4_weight,
            self.alpha_weight,
            self.gradient_weight,
        )
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError("Loss weights must be nonnegative with a positive sum.")
        fractions = tuple(float(value) for value in self.causality_cutoff_fractions)
        if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("Causality cutoff fractions must lie strictly in (0, 1).")
        for name, value in (
            ("expected_data_sha256", self.expected_data_sha256),
            ("expected_split_sha256", self.expected_split_sha256),
        ):
            if value is not None and (
                len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError(f"{name} must be a 64-character SHA256 digest.")
        if self.expected_parameter_count is not None and (
            self.expected_parameter_count < 1
        ):
            raise ValueError("expected_parameter_count must be positive.")
        if self.input_channels != len(INPUT_CHANNELS):
            raise ValueError(
                f"The causal source contract requires {len(INPUT_CHANNELS)} inputs."
            )
        if self.compute_timing_schema_version != COMPUTE_TIMING_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported causal-source compute timing schema version."
            )
        return replace(self, causality_cutoff_fractions=fractions)


def causal_receptive_field(
    depth: int, *, kernel_size: int = 3, dilation_cycle: int = 8
) -> int:
    """Return the exact temporal receptive field of the existing causal blocks."""

    if min(depth, kernel_size, dilation_cycle) < 1:
        raise ValueError("depth, kernel_size, and dilation_cycle must be positive.")
    return 1 + (kernel_size - 1) * sum(
        2 ** (layer_index % dilation_cycle) for layer_index in range(depth)
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Causal-source artifact is not canonical JSON."
        ) from error


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _semantic_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _artifact_reference(
    path: Path,
    project_root: Path,
    *,
    allow_external: bool = False,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Artifact is not one regular file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) and not allow_external:
        raise ValueError(f"Artifact escapes the project root: {path}")
    return {
        "path": (
            resolved.relative_to(root).as_posix()
            if resolved.is_relative_to(root)
            else resolved.as_posix()
        ),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat(follow_symlinks=False).st_size,
    }


def _implementation_file_manifest(project_root: Path) -> list[dict[str, Any]]:
    root = project_root.resolve()
    rows: list[dict[str, Any]] = []
    for relative in SCIENTIFIC_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Scientifically relevant implementation file is missing: {path}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _runtime_manifest(device: torch.device) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "device": str(device),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            raise ValueError("Resolved CUDA device must have an explicit index.")
        properties = torch.cuda.get_device_properties(index)
        runtime["cuda_device"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    else:
        runtime["cuda_device"] = None
    return runtime


def _implementation_runtime_binding(
    config: CausalSourceTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    manifest = {
        "implementation_files": _implementation_file_manifest(
            config.project_root
        ),
        "runtime": _runtime_manifest(device),
    }
    return {
        "schema_version": IMPLEMENTATION_BINDING_SCHEMA_VERSION,
        "manifest": manifest,
        "sha256": _semantic_sha256(manifest),
    }


def _validate_implementation_runtime_binding(
    binding: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version")
        != IMPLEMENTATION_BINDING_SCHEMA_VERSION
        or not isinstance(binding.get("manifest"), dict)
        or not isinstance(binding.get("sha256"), str)
    ):
        raise ValueError(f"{label} implementation/runtime binding is invalid.")
    actual = _semantic_sha256(binding["manifest"])
    if actual != binding["sha256"]:
        raise ValueError(
            f"{label} implementation/runtime binding hash is invalid."
        )
    return binding


def _config_payload(config: CausalSourceTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "data_path",
        "split_manifest",
        "output_root",
        "project_root",
        "config_file",
    ):
        payload[key] = _path_string(getattr(config, key))
    payload["causality_cutoff_fractions"] = list(
        config.causality_cutoff_fractions
    )
    return payload


def _scientific_config(config: CausalSourceTrainConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    payload.pop("resume", None)
    return payload


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _append_session_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Model state entry {name!r} is not a tensor.")
        values = tensor.detach().cpu().contiguous()
        descriptor = json.dumps(
            {
                "name": name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = values.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _clone_model_state_cpu(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _validate_model_state(
    model: torch.nn.Module,
    state: Any,
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} model state must be a mapping.")
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected).difference(state))
        extra = sorted(set(state).difference(expected))
        raise ValueError(
            f"{label} model state keys differ; missing={missing}, extra={extra}."
        )
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} model tensor {name!r} is invalid.")
        if (
            tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
        ):
            raise ValueError(
                f"{label} model tensor {name!r} has incompatible shape or dtype."
            )
        if (
            tensor.is_floating_point() or tensor.is_complex()
        ) and not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{label} model tensor {name!r} is non-finite.")
    return state


def _history_selection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, float, int]:
    if not rows:
        raise ValueError("Checkpoint authoritative history is empty.")
    best_value = float("inf")
    best_epoch = -1
    best_hash: str | None = None
    bad_epochs = 0
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or int(row.get("epoch", -1)) != position:
            raise ValueError("Checkpoint authoritative history is not contiguous.")
        value = float(row.get("validation_weighted_objective", float("nan")))
        if not np.isfinite(value):
            raise ValueError(
                "Checkpoint history has a non-finite validation objective."
            )
        improved = value < best_value
        if improved:
            best_value = value
            best_epoch = position
            bad_epochs = 0
        else:
            bad_epochs += 1
        recorded_epoch = int(row.get("selected_best_epoch", -1))
        recorded_value = float(
            row.get(
                "selected_best_validation_objective",
                float("nan"),
            )
        )
        recorded_hash = row.get("selected_best_model_sha256")
        if (
            recorded_epoch != best_epoch
            or recorded_value != best_value
            or not isinstance(recorded_hash, str)
            or len(recorded_hash) != 64
            or bool(row.get("improved")) is not improved
            or (not improved and recorded_hash != best_hash)
        ):
            raise ValueError(
                "Checkpoint history cumulative best selection is inconsistent."
            )
        best_hash = recorded_hash
    return best_epoch, best_value, bad_epochs


def _plain_history_rows(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    source = (
        rows.to_dict(orient="records")
        if isinstance(rows, pd.DataFrame)
        else [dict(row) for row in rows]
    )
    plain: list[dict[str, Any]] = []
    for row in source:
        converted = {
            str(name): (
                value.item() if isinstance(value, np.generic) else value
            )
            for name, value in row.items()
        }
        try:
            plain.append(json.loads(_canonical_json_bytes(converted)))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Causal-source timing history is not finite canonical JSON."
            ) from error
    return plain


def _strict_nonnegative_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def _strict_nonnegative_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number.")
    return float(value)


def _timing_device_synchronization(device_type: str) -> str:
    if device_type == "cuda":
        return CUDA_TIMING_SYNCHRONIZATION
    if device_type == "cpu":
        return CPU_TIMING_SYNCHRONIZATION
    raise ValueError(f"Unsupported timing device type: {device_type!r}.")


def _resume_evidence_from_sessions(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            "Causal-source run session evidence is missing."
        )
    starts: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(
                f"Run-session evidence has an empty line at {line_number}."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Run-session evidence line {line_number} is invalid."
            ) from error
        if not isinstance(record, dict):
            raise ValueError("Run-session evidence rows must be objects.")
        if record.get("event") == "session_started":
            starts.append(record)
    if not starts or starts[0].get("resume") is not False:
        raise ValueError("Run-session evidence lacks one initial session.")
    if any(record.get("resume") is not True for record in starts[1:]):
        raise ValueError("Run-session resume flags are not contiguous.")
    controlled = 0
    unobserved = 0
    for record in starts[1:]:
        previous_status = record.get("previous_status")
        possible = record.get(
            "unobserved_failed_attempt_overhead_possible"
        )
        expected_possible = previous_status != "paused"
        if type(previous_status) is not str or possible is not expected_possible:
            raise ValueError(
                "Run-session interruption classification is invalid."
            )
        if expected_possible:
            unobserved += 1
        else:
            controlled += 1
    return {
        "resume_invocation_count": len(starts) - 1,
        "controlled_resume_invocation_count": controlled,
        "unobserved_interruption_resume_count": unobserved,
        "unobserved_failed_attempt_overhead_possible": unobserved > 0,
        "timing_complete": unobserved == 0,
    }


def causal_source_compute_timing_from_history(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    expected_parameter_count: int,
    expected_optimizer_updates_per_epoch: int,
    device_type: str,
    expected_resume_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate history and derive exact separated compute totals."""

    parameter_count = _strict_nonnegative_integer(
        expected_parameter_count,
        label="Expected source parameter count",
    )
    updates_per_epoch = _strict_nonnegative_integer(
        expected_optimizer_updates_per_epoch,
        label="Expected optimizer updates per epoch",
    )
    if parameter_count < 1 or updates_per_epoch < 1:
        raise ValueError(
            "Expected parameter and optimizer-update counts must be positive."
        )
    synchronization = _timing_device_synchronization(device_type)
    history = _plain_history_rows(rows)
    if not history:
        raise ValueError("Causal-source timing history is empty.")
    previous_resume_count = 0
    previous_controlled_count = 0
    previous_unobserved_count = 0
    optimizer_durations: list[float] = []
    candidate_durations: list[float] = []
    for position, row in enumerate(history, start=1):
        if set(row) != _HISTORY_ROW_KEYS:
            missing = sorted(_HISTORY_ROW_KEYS.difference(row))
            extra = sorted(set(row).difference(_HISTORY_ROW_KEYS))
            raise ValueError(
                "Causal-source timing history keys differ; "
                f"missing={missing}, extra={extra}."
            )
        if type(row["epoch"]) is not int or row["epoch"] != position:
            raise ValueError(
                "Causal-source timing history epochs are not contiguous."
            )
        if (
            row["history_timing_schema_version"]
            != HISTORY_TIMING_SCHEMA_VERSION
            or row["duration_seconds_semantics"]
            != _EPOCH_WALL_TIME_SEMANTICS
            or row["parameter_count"] != parameter_count
            or type(row["parameter_count"]) is not int
            or row["optimizer_update_count"] != updates_per_epoch
            or type(row["optimizer_update_count"]) is not int
            or row["candidate_count"] != 1
            or type(row["candidate_count"]) is not int
            or row["optimizer_timing_clock_source"]
            != OPTIMIZER_TIMING_CLOCK_SOURCE
            or row["candidate_timing_clock_source"]
            != CANDIDATE_TIMING_CLOCK_SOURCE
            or row["optimizer_device_synchronization"] != synchronization
            or row["candidate_device_synchronization"] != synchronization
            or row["candidate_validation_time_in_optimizer_total"] is not False
        ):
            raise ValueError(
                f"Causal-source timing identity differs at epoch {position}."
            )
        optimizer_duration = _strict_nonnegative_number(
            row["device_synchronized_training_seconds"],
            label=f"Epoch {position} optimizer duration",
        )
        candidate_duration = _strict_nonnegative_number(
            row["candidate_validation_seconds"],
            label=f"Epoch {position} candidate duration",
        )
        epoch_wall = _strict_nonnegative_number(
            row["duration_seconds"],
            label=f"Epoch {position} wall duration",
        )
        if epoch_wall < optimizer_duration + candidate_duration:
            raise ValueError(
                f"Epoch {position} separated timing exceeds epoch wall time."
            )
        for name in (
            "train_temperature_relative_l2",
            "train_temperature_normalized_l4",
            "train_alpha_relative_l2",
            "train_spatial_gradient_relative_l2",
            "train_weighted_objective",
            "validation_temperature_relative_l2",
            "validation_temperature_normalized_l4",
            "validation_alpha_relative_l2",
            "validation_spatial_gradient_relative_l2",
            "validation_weighted_objective",
            "gradient_norm_before_clip_last_batch",
            "learning_rate",
            "selected_best_validation_objective",
        ):
            _strict_nonnegative_number(
                row[name],
                label=f"Epoch {position} {name}",
            )
        if (
            type(row["improved"]) is not bool
            or type(row["selected_best_epoch"]) is not int
            or type(row["selected_best_model_sha256"]) is not str
            or len(row["selected_best_model_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row["selected_best_model_sha256"]
            )
        ):
            raise ValueError(
                f"Epoch {position} selection metadata is invalid."
            )
        resume_count = _strict_nonnegative_integer(
            row["resume_invocation_count"],
            label=f"Epoch {position} resume count",
        )
        controlled_count = _strict_nonnegative_integer(
            row["controlled_resume_invocation_count"],
            label=f"Epoch {position} controlled-resume count",
        )
        unobserved_count = _strict_nonnegative_integer(
            row["unobserved_interruption_resume_count"],
            label=f"Epoch {position} unobserved-resume count",
        )
        if (
            controlled_count + unobserved_count != resume_count
            or resume_count < previous_resume_count
            or controlled_count < previous_controlled_count
            or unobserved_count < previous_unobserved_count
            or row["replayed_optimizer_update_count"] != 0
            or type(row["replayed_optimizer_update_count"]) is not int
            or row["replayed_optimizer_duration_seconds"] != 0.0
            or type(row["replayed_optimizer_duration_seconds"]) is not float
            or row["timing_complete"] is not (unobserved_count == 0)
        ):
            raise ValueError(
                f"Epoch {position} resume/replay timing evidence differs."
            )
        previous_resume_count = resume_count
        previous_controlled_count = controlled_count
        previous_unobserved_count = unobserved_count
        optimizer_durations.append(optimizer_duration)
        candidate_durations.append(candidate_duration)
    _history_selection(history)
    final_resume = {
        "resume_invocation_count": previous_resume_count,
        "controlled_resume_invocation_count": previous_controlled_count,
        "unobserved_interruption_resume_count": previous_unobserved_count,
        "unobserved_failed_attempt_overhead_possible": (
            previous_unobserved_count > 0
        ),
        "timing_complete": previous_unobserved_count == 0,
    }
    if expected_resume_evidence is not None and dict(
        expected_resume_evidence
    ) != final_resume:
        raise ValueError(
            "Causal-source history resume evidence differs from sessions."
        )
    successful_seconds = float(math.fsum(optimizer_durations))
    candidate_seconds = float(math.fsum(candidate_durations))
    history_sha256 = _canonical_sha256(history)
    timing_complete = final_resume["timing_complete"]
    return {
        "schema_version": COMPUTE_TIMING_SCHEMA_VERSION,
        "protocol": COMPUTE_TIMING_PROTOCOL,
        "history_timing_schema_version": HISTORY_TIMING_SCHEMA_VERSION,
        "history_payload_sha256": history_sha256,
        "history_row_count": len(history),
        "parameter_count": parameter_count,
        "successful_optimizer_update_count": (
            updates_per_epoch * len(history)
        ),
        "successful_optimizer_duration_seconds": successful_seconds,
        "candidate_evaluation_count": len(history),
        "candidate_evaluation_duration_seconds": candidate_seconds,
        "replayed_optimizer_update_count": 0,
        "replayed_optimizer_duration_seconds": 0.0,
        **final_resume,
        "optimizer_timing_clock_source": OPTIMIZER_TIMING_CLOCK_SOURCE,
        "candidate_timing_clock_source": CANDIDATE_TIMING_CLOCK_SOURCE,
        "device_synchronization": synchronization,
        "candidate_evaluation_time_in_optimizer_total": False,
        "failed_attempt_overhead_fully_observed": timing_complete,
        "timing_scope": {
            "successful_optimizer_updates": "complete_history_derived",
            "candidate_evaluations": "complete_history_derived",
            "replayed_optimizer_updates": (
                "no_durable_replay_observation_available"
            ),
            "failed_process_attempt_overhead": (
                "complete_no_uncontrolled_resume_observed"
                if timing_complete
                else "possibly_incomplete_due_to_process_termination"
            ),
            "legacy_epoch_duration": "excluded_from_compute_accounting",
        },
        "optimizer_update_count": updates_per_epoch * len(history),
        "candidate_validation_seconds": candidate_seconds,
        "device_synchronized_training_seconds": successful_seconds,
        "candidate_count": len(history),
        "source_training_run_id": None,
    }


def _compute_accounting(
    compute_timing: Mapping[str, Any],
) -> dict[str, Any]:
    accounting = {
        key: compute_timing[key] for key in COMPUTE_ACCOUNTING_FIELDS
    }
    if (
        type(accounting["parameter_count"]) is not int
        or accounting["parameter_count"] < 1
        or type(accounting["optimizer_update_count"]) is not int
        or accounting["optimizer_update_count"] < 1
        or type(accounting["candidate_count"]) is not int
        or accounting["candidate_count"] < 1
        or accounting["source_training_run_id"] is not None
        or type(accounting["timing_complete"]) is not bool
    ):
        raise ValueError("Causal-source compute accounting is invalid.")
    _strict_nonnegative_number(
        accounting["candidate_validation_seconds"],
        label="Candidate-validation time",
    )
    _strict_nonnegative_number(
        accounting["device_synchronized_training_seconds"],
        label="Device-synchronized training time",
    )
    return accounting


def _validate_checkpoint_compute_binding(
    checkpoint: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_parameter_count: int,
    expected_optimizer_updates_per_epoch: int,
    device_type: str,
    expected_resume_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = causal_source_compute_timing_from_history(
        history_rows,
        expected_parameter_count=expected_parameter_count,
        expected_optimizer_updates_per_epoch=(
            expected_optimizer_updates_per_epoch
        ),
        device_type=device_type,
        expected_resume_evidence=expected_resume_evidence,
    )
    if checkpoint.get("compute_timing") != expected:
        raise ValueError(f"{label} compute timing differs from its history.")
    accounting = _compute_accounting(expected)
    if checkpoint.get("compute_accounting") != accounting:
        raise ValueError(
            f"{label} compute accounting differs from its history."
        )
    return expected


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def _reconcile_history_artifact(
    path: Path,
    authoritative_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in authoritative_rows]
    expected = pd.DataFrame(rows)
    repair = False
    if not path.is_file():
        repair = True
    else:
        observed = pd.read_parquet(path)
        if len(observed) > len(expected):
            raise ValueError("Resume history is ahead of last.pt.")
        prefix = expected.iloc[: len(observed)].reset_index(drop=True)
        if not _frames_equal(observed, prefix):
            raise ValueError("Resume history differs from authoritative last.pt.")
        repair = len(observed) < len(expected)
    if repair:
        _atomic_parquet(expected, path)
    return rows


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
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "cuda":
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device_index} does not exist; "
                f"device_count={torch.cuda.device_count()}."
            )
        device = torch.device("cuda", device_index)
    return device


def _synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_previous_status(run_dir: Path) -> str | None:
    path = run_dir / "STATUS.json"
    if path.is_symlink():
        raise ValueError("Run STATUS.json cannot be a symlink.")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Previous run STATUS.json is invalid.") from error
    status = payload.get("status") if isinstance(payload, dict) else None
    if type(status) is not str or not status:
        raise ValueError("Previous run status is invalid.")
    return status


def _device_payload(device: torch.device, requested: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
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


def _model_spec(config: CausalSourceTrainConfig) -> dict[str, Any]:
    return {
        "family": MODEL_FAMILY,
        "input_channels": config.input_channels,
        "channel_names": list(INPUT_CHANNELS),
        "width": config.width,
        "depth": config.depth,
        "modes_space": config.modes_space,
        "temporal_kernel_size": 3,
        "temporal_dilations": [
            2 ** (index % 8) for index in range(config.depth)
        ],
        "temporal_receptive_field": causal_receptive_field(config.depth),
        "structurally_causal": True,
    }


def _input_checksums(config: CausalSourceTrainConfig) -> dict[str, Any]:
    data_sha = sha256_file(config.data_path)
    split_sha = sha256_file(config.split_manifest)
    if (
        config.expected_data_sha256 is not None
        and data_sha.lower() != config.expected_data_sha256.lower()
    ):
        raise ValueError(
            "Source data SHA256 differs from the pinned causal-source config."
        )
    if (
        config.expected_split_sha256 is not None
        and split_sha.lower() != config.expected_split_sha256.lower()
    ):
        raise ValueError(
            "Source split SHA256 differs from the pinned causal-source config."
        )
    payload: dict[str, Any] = {
        "data": {
            "path": _path_string(config.data_path),
            "sha256": data_sha,
            "bytes": config.data_path.stat().st_size,
        },
        "split_manifest": {
            "path": _path_string(config.split_manifest),
            "sha256": split_sha,
            "bytes": config.split_manifest.stat().st_size,
        },
    }
    if config.config_file is not None:
        payload["declared_config"] = {
            "path": _path_string(config.config_file),
            "sha256": sha256_file(config.config_file),
            "bytes": config.config_file.stat().st_size,
        }
    return payload


def _write_provenance(
    run_dir: Path,
    config: CausalSourceTrainConfig,
    device: torch.device,
    git_sha: str,
    checksums: dict[str, Any],
    implementation_runtime_binding: dict[str, Any],
) -> None:
    _atomic_write_text(
        run_dir / "config_resolved.json",
        _canonical_json(
            {
                "schema_version": 1,
                "experiment": EXPERIMENT,
                "scientific_config": _scientific_config(config),
                "compute_timing_contract": COMPUTE_TIMING_CONFIG,
                "acceptance": ACCEPTANCE,
            }
        ),
    )
    _atomic_write_text(
        run_dir / "data_checksums.json", _canonical_json(checksums)
    )
    _atomic_write_text(
        run_dir / "implementation_runtime.json",
        _canonical_json(implementation_runtime_binding),
    )
    git_status = _git(config.project_root, "status", "--short")
    _atomic_write_text(
        run_dir / "git_state.txt",
        f"HEAD {git_sha}\n\nstatus --short\n{git_status}\n",
    )
    environment = {
        "schema_version": 1,
        "python": sys.version.replace(os.linesep, " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "device": _device_payload(device, config.device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
        "command": sys.argv,
        "implementation_runtime_binding_sha256": (
            implementation_runtime_binding["sha256"]
        ),
    }
    _atomic_write_text(
        run_dir / "environment.json", _canonical_json(environment)
    )


def _source_loss_components(
    outputs: dict[str, torch.Tensor],
    temperature: torch.Tensor,
    alpha: torch.Tensor,
    material_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    components = joint_loss_components(
        outputs, temperature, alpha, material_mask
    )
    error = torch.abs(outputs["temperature"] - temperature)
    components["temperature_l4"] = torch.mean(
        torch.mean(error**4, dim=(1, 2)) ** 0.25
    )
    return components


def _weighted_loss(
    components: dict[str, torch.Tensor],
    config: CausalSourceTrainConfig,
) -> torch.Tensor:
    return (
        config.temperature_weight * components["temperature"]
        + config.temperature_l4_weight * components["temperature_l4"]
        + config.alpha_weight * components["alpha"]
        + config.gradient_weight * components["gradient"]
    )


def _evaluate_objective(
    model: torch.nn.Module,
    prepared: PreparedSource1D,
    split: str,
    config: CausalSourceTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        prepared.dataset(split),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    totals = {
        "temperature": 0.0,
        "temperature_l4": 0.0,
        "alpha": 0.0,
        "gradient": 0.0,
        "weighted": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, temperature, alpha, _, _ in loader:
            inputs = inputs.to(device)
            temperature = temperature.to(device)
            alpha = alpha.to(device)
            components = _source_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            weighted = _weighted_loss(components, config)
            batch_count = len(inputs)
            for name, value in components.items():
                totals[name] += float(value) * batch_count
            totals["weighted"] += float(weighted) * batch_count
            count += batch_count
    if count == 0:
        raise ValueError(f"Split {split!r} is empty.")
    return {name: value / count for name, value in totals.items()}


def _decode_temperature(
    values: np.ndarray, normalization: dict[str, Any]
) -> np.ndarray:
    metadata = normalization["field_temperature"]
    return values * (metadata["maximum"] - metadata["minimum"]) + metadata[
        "minimum"
    ]


def _evaluate_split_metrics(
    model: torch.nn.Module,
    prepared: PreparedSource1D,
    split: str,
    config: CausalSourceTrainConfig,
    device: torch.device,
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
        for inputs, temperature, alpha, case_ids, family_ids in loader:
            outputs = model(inputs.to(device))
            predicted_temperature = _decode_temperature(
                outputs["temperature"].cpu().numpy(), prepared.normalization
            )
            target_temperature = _decode_temperature(
                temperature.numpy(), prepared.normalization
            )
            predicted_alpha = outputs["alpha"].cpu().numpy()
            target_alpha = alpha.numpy()
            masks = inputs[..., 4].numpy() > 0.5
            for index in range(len(inputs)):
                temperature_error = (
                    predicted_temperature[index] - target_temperature[index]
                )
                mask = masks[index]
                alpha_prediction_masked = predicted_alpha[index][mask]
                alpha_target_masked = target_alpha[index][mask]
                alpha_error = alpha_prediction_masked - alpha_target_masked
                rows.append(
                    {
                        "split": split,
                        "case_id": int(case_ids[index]),
                        "family_id": int(family_ids[index]),
                        "temperature_relative_l2": float(
                            np.linalg.norm(temperature_error)
                            / max(
                                np.linalg.norm(target_temperature[index]),
                                np.finfo(np.float64).eps,
                            )
                        ),
                        "temperature_mae_K": float(
                            np.mean(np.abs(temperature_error))
                        ),
                        "temperature_linf_K": float(
                            np.max(np.abs(temperature_error))
                        ),
                        "alpha_relative_l2": float(
                            np.linalg.norm(alpha_error)
                            / max(
                                np.linalg.norm(alpha_target_masked),
                                np.finfo(np.float64).eps,
                            )
                        ),
                        "alpha_mae": float(np.mean(np.abs(alpha_error))),
                        "alpha_bound_violation_count": int(
                            np.count_nonzero(
                                (predicted_alpha[index] < -1.0e-7)
                                | (predicted_alpha[index] > 1.0 + 1.0e-7)
                            )
                        ),
                        "alpha_monotonicity_violation_count": int(
                            np.count_nonzero(
                                np.diff(predicted_alpha[index], axis=0)
                                < -1.0e-7
                            )
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Split {split!r} is empty.")
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


def _causality_cutoffs(
    time_count: int, fractions: Sequence[float]
) -> list[int]:
    if time_count < 3:
        raise ValueError("Future-perturbation checks require at least three times.")
    cutoffs = {
        min(max(int(round((time_count - 1) * fraction)), 0), time_count - 2)
        for fraction in fractions
    }
    return sorted(cutoffs)


def check_future_invariance(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    case_ids: Sequence[int] | torch.Tensor,
    *,
    tolerance: float = 1.0e-7,
    cutoff_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Perturb only future dynamic inputs and audit every predicted prefix."""

    if inputs.ndim != 4 or inputs.shape[-1] != len(INPUT_CHANNELS):
        raise ValueError("Expected source inputs shaped [B,Nt,Nz,14].")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    ids = [int(value) for value in case_ids]
    if len(ids) != len(inputs):
        raise ValueError("case_ids must align with inputs.")
    active_device = torch.device(device)
    rows: list[dict[str, float | int | str | bool]] = []
    dynamic_channels = (0, 1, 13)
    cutoffs = _causality_cutoffs(inputs.shape[1], cutoff_fractions)
    model.eval()
    with torch.no_grad():
        for index, case_id in enumerate(ids):
            original = inputs[index : index + 1].to(active_device)
            reference = model(original)
            for cutoff in cutoffs:
                counterfactual = original.clone()
                future = counterfactual[:, cutoff + 1 :, :, dynamic_channels]
                perturbed = torch.flip(future, dims=(1,))
                offsets = torch.tensor(
                    [0.173, -0.127, 0.061],
                    dtype=perturbed.dtype,
                    device=perturbed.device,
                )
                counterfactual[:, cutoff + 1 :, :, dynamic_channels] = (
                    perturbed + offsets
                )
                changed = model(counterfactual)
                for output_name in (
                    "temperature",
                    "temperature_residual",
                    "alpha",
                    "cure_rate",
                ):
                    baseline_prefix = reference[output_name][:, : cutoff + 1]
                    difference = (
                        changed[output_name][:, : cutoff + 1]
                        - baseline_prefix
                    )
                    denominator = torch.linalg.vector_norm(
                        baseline_prefix.double()
                    ).clamp_min(torch.finfo(torch.float64).eps)
                    score = (
                        torch.linalg.vector_norm(difference.double())
                        / denominator
                    )
                    maximum = float(torch.max(torch.abs(difference)))
                    rows.append(
                        {
                            "case_id": case_id,
                            "cutoff_index": cutoff,
                            "future_start_index": cutoff + 1,
                            "output": output_name,
                            "maximum_prefix_abs_difference": maximum,
                            "prefix_relative_l2": float(score),
                            "passed": maximum <= tolerance,
                        }
                    )
    passed = bool(rows) and all(bool(row["passed"]) for row in rows)
    return {
        "schema_version": 1,
        "test": "counterfactual_future_perturbation",
        "architecture_is_causal": True,
        "dynamic_input_channels": [INPUT_CHANNELS[index] for index in dynamic_channels],
        "case_ids": ids,
        "cutoff_indices": cutoffs,
        "tolerance": tolerance,
        "maximum_prefix_abs_difference": max(
            float(row["maximum_prefix_abs_difference"]) for row in rows
        ),
        "maximum_prefix_relative_l2": max(
            float(row["prefix_relative_l2"]) for row in rows
        ),
        "rows": rows,
        "passed": passed,
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    best_validation: float,
    best_epoch: int,
    best_model_state: Mapping[str, torch.Tensor],
    best_model_sha256: str,
    bad_epochs: int,
    history_rows: Sequence[Mapping[str, Any]],
    generator: torch.Generator,
    config: CausalSourceTrainConfig,
    checksums: dict[str, Any],
    normalization: dict[str, Any],
    git_sha: str,
    implementation_runtime_binding: dict[str, Any],
    parameter_count: int,
    optimizer_updates_per_epoch: int,
    device_type: str,
    resume_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = model.state_dict()
    compute_timing = causal_source_compute_timing_from_history(
        history_rows,
        expected_parameter_count=parameter_count,
        expected_optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device_type,
        expected_resume_evidence=resume_evidence,
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "last",
        "artifact_generation": epoch,
        "phase": "P3",
        "experiment": EXPERIMENT,
        "model_family": MODEL_FAMILY,
        "model_spec": _model_spec(config),
        "model": model_state,
        "model_sha256": _model_state_sha256(model_state),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "validation_objective": float(
            history_rows[-1]["validation_weighted_objective"]
        ),
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "best_snapshot": {
            "epoch": best_epoch,
            "validation_objective": best_validation,
            "model": dict(best_model_state),
            "model_sha256": best_model_sha256,
        },
        "bad_epochs": bad_epochs,
        "authoritative_history": [dict(row) for row in history_rows],
        "compute_timing": compute_timing,
        "compute_accounting": _compute_accounting(compute_timing),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda"
            else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "scientific_config": _scientific_config(config),
        "input_checksums": checksums,
        "normalization": normalization,
        "channel_names": tuple(INPUT_CHANNELS),
        "selection_split": "validation",
        "held_out_labels_used_for_selection": False,
        "git_sha": git_sha,
        "implementation_runtime_binding": implementation_runtime_binding,
    }


def _best_checkpoint_payload(
    last_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = last_checkpoint["best_snapshot"]
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "best",
        "artifact_generation": int(last_checkpoint["artifact_generation"]),
        "phase": "P3",
        "experiment": EXPERIMENT,
        "model_family": MODEL_FAMILY,
        "model_spec": last_checkpoint["model_spec"],
        "model": snapshot["model"],
        "model_sha256": snapshot["model_sha256"],
        "epoch": int(snapshot["epoch"]),
        "validation_objective": float(snapshot["validation_objective"]),
        "best_validation": float(snapshot["validation_objective"]),
        "scientific_config": last_checkpoint["scientific_config"],
        "input_checksums": last_checkpoint["input_checksums"],
        "normalization": last_checkpoint["normalization"],
        "channel_names": last_checkpoint["channel_names"],
        "selection_split": "validation",
        "held_out_labels_used_for_selection": False,
        "git_sha": last_checkpoint["git_sha"],
        "implementation_runtime_binding": last_checkpoint[
            "implementation_runtime_binding"
        ],
        "compute_timing": last_checkpoint["compute_timing"],
        "compute_accounting": last_checkpoint["compute_accounting"],
    }


def _validate_checkpoint_common(
    checkpoint: Any,
    *,
    role: str,
    model: torch.nn.Module,
    config: CausalSourceTrainConfig,
    checksums: dict[str, Any],
    normalization: dict[str, Any],
    git_sha: str,
    implementation_runtime_binding: dict[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{role}.pt must contain a mapping.")
    declarations = {
        "schema_version": (
            checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        ),
        "checkpoint_role": checkpoint.get("checkpoint_role") == role,
        "phase": checkpoint.get("phase") == "P3",
        "experiment": checkpoint.get("experiment") == EXPERIMENT,
        "model_family": checkpoint.get("model_family") == MODEL_FAMILY,
        "model_spec": checkpoint.get("model_spec") == _model_spec(config),
        "scientific_config": (
            checkpoint.get("scientific_config") == _scientific_config(config)
        ),
        "input_checksums": checkpoint.get("input_checksums") == checksums,
        "normalization": checkpoint.get("normalization") == normalization,
        "channel_names": (
            tuple(checkpoint.get("channel_names", ())) == tuple(INPUT_CHANNELS)
        ),
        "selection_split": checkpoint.get("selection_split") == "validation",
        "held_out_labels_used_for_selection": (
            checkpoint.get("held_out_labels_used_for_selection") is False
        ),
        "git_sha": checkpoint.get("git_sha") == git_sha,
    }
    if not declarations["git_sha"]:
        raise ValueError(
            f"{role}.pt Git SHA differs from the current implementation."
        )
    invalid = [name for name, valid in declarations.items() if not valid]
    if invalid:
        raise ValueError(
            f"{role}.pt metadata differs from this run: {invalid}."
        )
    stored_binding = _validate_implementation_runtime_binding(
        checkpoint.get("implementation_runtime_binding"),
        label=f"{role}.pt",
    )
    if stored_binding != implementation_runtime_binding:
        raise ValueError(
            f"{role}.pt implementation/runtime binding differs from "
            "the current implementation."
        )
    model_state = _validate_model_state(
        model, checkpoint.get("model"), label=f"{role}.pt"
    )
    declared_model_sha = checkpoint.get("model_sha256")
    if (
        not isinstance(declared_model_sha, str)
        or declared_model_sha != _model_state_sha256(model_state)
    ):
        raise ValueError(f"{role}.pt model-state hash is invalid.")
    return checkpoint


def _validate_last_checkpoint(
    checkpoint: Any,
    *,
    model: torch.nn.Module,
    config: CausalSourceTrainConfig,
    checksums: dict[str, Any],
    normalization: dict[str, Any],
    git_sha: str,
    implementation_runtime_binding: dict[str, Any],
    parameter_count: int,
    optimizer_updates_per_epoch: int,
    device_type: str,
    resume_evidence: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    validated = _validate_checkpoint_common(
        checkpoint,
        role="last",
        model=model,
        config=config,
        checksums=checksums,
        normalization=normalization,
        git_sha=git_sha,
        implementation_runtime_binding=implementation_runtime_binding,
    )
    try:
        epoch = int(validated["epoch"])
        generation = int(validated["artifact_generation"])
        bad_epochs = int(validated["bad_epochs"])
        best_epoch_recorded = int(validated["best_epoch"])
        best_validation_recorded = float(validated["best_validation"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("last.pt epoch/selection metadata is invalid.") from error
    if epoch < 1 or epoch > config.epochs or generation != epoch:
        raise ValueError("last.pt epoch or artifact generation is invalid.")
    history_raw = validated.get("authoritative_history")
    if not isinstance(history_raw, list) or len(history_raw) != epoch:
        raise ValueError("last.pt authoritative history length is invalid.")
    history = [dict(row) for row in history_raw]
    _validate_checkpoint_compute_binding(
        validated,
        history,
        label="last.pt",
        expected_parameter_count=parameter_count,
        expected_optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device_type,
        expected_resume_evidence=resume_evidence,
    )
    best_epoch, best_validation, expected_bad_epochs = _history_selection(
        history
    )
    if float(validated.get("validation_objective", float("nan"))) != float(
        history[-1]["validation_weighted_objective"]
    ):
        raise ValueError("last.pt validation objective differs from its history.")
    if (
        best_epoch_recorded != best_epoch
        or best_validation_recorded != best_validation
        or bad_epochs != expected_bad_epochs
    ):
        raise ValueError("last.pt selection state differs from its history.")
    snapshot = validated.get("best_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("last.pt has no valid best snapshot.")
    try:
        snapshot_epoch = int(snapshot["epoch"])
        snapshot_objective = float(snapshot["validation_objective"])
        snapshot_hash = str(snapshot["model_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("last.pt best snapshot metadata is invalid.") from error
    snapshot_state = _validate_model_state(
        model,
        snapshot.get("model"),
        label="last.pt best snapshot",
    )
    actual_snapshot_hash = _model_state_sha256(snapshot_state)
    final_row = history[-1]
    if (
        snapshot_epoch != best_epoch
        or snapshot_objective != best_validation
        or snapshot_hash != actual_snapshot_hash
        or final_row["selected_best_model_sha256"] != snapshot_hash
    ):
        raise ValueError("last.pt best snapshot differs from its history.")
    for required in (
        "optimizer",
        "scheduler",
        "generator_state",
        "torch_rng_state",
        "python_rng_state",
        "numpy_rng_state",
    ):
        if required not in validated:
            raise ValueError(f"last.pt is missing resume state {required!r}.")
    if not isinstance(validated["optimizer"], Mapping) or not isinstance(
        validated["scheduler"], Mapping
    ):
        raise ValueError("last.pt optimizer or scheduler state is invalid.")
    if int(validated["scheduler"].get("last_epoch", -1)) != epoch:
        raise ValueError("last.pt scheduler epoch differs from checkpoint epoch.")
    if not isinstance(validated["generator_state"], torch.Tensor) or not isinstance(
        validated["torch_rng_state"], torch.Tensor
    ):
        raise ValueError("last.pt RNG tensor state is invalid.")
    return history, snapshot


def _validate_best_checkpoint(
    checkpoint: Any,
    *,
    model: torch.nn.Module,
    config: CausalSourceTrainConfig,
    checksums: dict[str, Any],
    normalization: dict[str, Any],
    git_sha: str,
    implementation_runtime_binding: dict[str, Any],
    authoritative_history: Sequence[Mapping[str, Any]],
    parameter_count: int,
    optimizer_updates_per_epoch: int,
    device_type: str,
    resume_evidence: Mapping[str, Any] | None,
) -> int:
    validated = _validate_checkpoint_common(
        checkpoint,
        role="best",
        model=model,
        config=config,
        checksums=checksums,
        normalization=normalization,
        git_sha=git_sha,
        implementation_runtime_binding=implementation_runtime_binding,
    )
    _validate_checkpoint_compute_binding(
        validated,
        authoritative_history,
        label="best.pt",
        expected_parameter_count=parameter_count,
        expected_optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device_type,
        expected_resume_evidence=resume_evidence,
    )
    try:
        generation = int(validated["artifact_generation"])
        selected_epoch = int(validated["epoch"])
        objective = float(validated["validation_objective"])
        declared_hash = str(validated["model_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("best.pt selection metadata is invalid.") from error
    if generation < 1 or generation > len(authoritative_history):
        raise ValueError("best.pt artifact generation is invalid.")
    generation_row = authoritative_history[generation - 1]
    expected_epoch = int(generation_row["selected_best_epoch"])
    expected_objective = float(
        generation_row["selected_best_validation_objective"]
    )
    expected_hash = str(generation_row["selected_best_model_sha256"])
    actual_hash = _model_state_sha256(validated["model"])
    if (
        selected_epoch != expected_epoch
        or objective != expected_objective
        or float(validated.get("best_validation", float("nan")))
        != expected_objective
        or declared_hash != expected_hash
        or actual_hash != expected_hash
    ):
        raise ValueError("best.pt differs from authoritative last.pt history.")
    return generation


def _reconcile_best_artifact(
    path: Path,
    last_checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    config: CausalSourceTrainConfig,
    checksums: dict[str, Any],
    normalization: dict[str, Any],
    git_sha: str,
    implementation_runtime_binding: dict[str, Any],
    authoritative_history: Sequence[Mapping[str, Any]],
    parameter_count: int,
    optimizer_updates_per_epoch: int,
    device_type: str,
    resume_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_generation = int(last_checkpoint["artifact_generation"])
    repair = not path.is_file()
    if path.is_file():
        candidate = torch.load(path, map_location="cpu", weights_only=False)
        observed_generation = _validate_best_checkpoint(
            candidate,
            model=model,
            config=config,
            checksums=checksums,
            normalization=normalization,
            git_sha=git_sha,
            implementation_runtime_binding=implementation_runtime_binding,
            authoritative_history=authoritative_history,
            parameter_count=parameter_count,
            optimizer_updates_per_epoch=optimizer_updates_per_epoch,
            device_type=device_type,
            resume_evidence=resume_evidence,
        )
        if observed_generation > expected_generation:
            raise ValueError("best.pt is ahead of last.pt.")
        repair = observed_generation < expected_generation
    expected = _best_checkpoint_payload(last_checkpoint)
    if repair:
        _atomic_torch_save(expected, path)
    persisted = torch.load(path, map_location="cpu", weights_only=False)
    observed_generation = _validate_best_checkpoint(
        persisted,
        model=model,
        config=config,
        checksums=checksums,
        normalization=normalization,
        git_sha=git_sha,
        implementation_runtime_binding=implementation_runtime_binding,
        authoritative_history=authoritative_history,
        parameter_count=parameter_count,
        optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device_type,
        resume_evidence=resume_evidence,
    )
    if observed_generation != expected_generation:
        raise ValueError("best.pt could not be reconciled to last.pt.")
    final_snapshot = last_checkpoint["best_snapshot"]
    if (
        persisted["model_sha256"] != final_snapshot["model_sha256"]
        or int(persisted["epoch"]) != int(final_snapshot["epoch"])
        or float(persisted["validation_objective"])
        != float(final_snapshot["validation_objective"])
    ):
        raise ValueError("best.pt does not match the final best snapshot.")
    return persisted


def _restore_rng(checkpoint: dict[str, Any]) -> None:
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the checkpoint RNG-state count."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _acceptance_summary(
    split_summaries: dict[str, dict[str, float | int]],
    held_out_names: Sequence[str],
) -> tuple[dict[str, float | int], dict[str, bool]]:
    held_out_case_count = sum(
        int(split_summaries[name]["case_count"]) for name in held_out_names
    )
    weighted_temperature = sum(
        float(split_summaries[name]["temperature_relative_l2_mean"])
        * int(split_summaries[name]["case_count"])
        for name in held_out_names
    ) / held_out_case_count
    weighted_alpha = sum(
        float(split_summaries[name]["alpha_relative_l2_mean"])
        * int(split_summaries[name]["case_count"])
        for name in held_out_names
    ) / held_out_case_count
    held_out = {
        "case_count": held_out_case_count,
        "temperature_relative_l2_mean": weighted_temperature,
        "temperature_linf_K_max": max(
            float(split_summaries[name]["temperature_linf_K_max"])
            for name in held_out_names
        ),
        "alpha_relative_l2_mean": weighted_alpha,
        "alpha_bound_violation_count": sum(
            int(split_summaries[name]["alpha_bound_violation_count"])
            for name in held_out_names
        ),
        "alpha_monotonicity_violation_count": sum(
            int(split_summaries[name]["alpha_monotonicity_violation_count"])
            for name in held_out_names
        ),
    }
    checks = {
        "in_family_temperature": (
            float(
                split_summaries["in_family_test"][
                    "temperature_relative_l2_mean"
                ]
            )
            <= ACCEPTANCE["in_family_temperature_relative_l2_mean_max"]
        ),
        "held_out_temperature": (
            float(held_out["temperature_relative_l2_mean"])
            <= ACCEPTANCE["held_out_temperature_relative_l2_mean_max"]
        ),
        "held_out_each_family_temperature": all(
            float(split_summaries[name]["temperature_relative_l2_mean"])
            <= ACCEPTANCE[
                "held_out_each_family_temperature_relative_l2_mean_max"
            ]
            for name in held_out_names
        ),
        "held_out_alpha": (
            float(held_out["alpha_relative_l2_mean"])
            <= ACCEPTANCE["held_out_alpha_relative_l2_mean_max"]
        ),
        "held_out_linf": (
            float(held_out["temperature_linf_K_max"])
            <= ACCEPTANCE["held_out_temperature_linf_K_max"]
        ),
        "alpha_bounds": (
            int(held_out["alpha_bound_violation_count"])
            <= ACCEPTANCE["alpha_bound_violation_count_max"]
        ),
        "alpha_monotonicity": (
            int(held_out["alpha_monotonicity_violation_count"])
            <= ACCEPTANCE["alpha_monotonicity_violation_count_max"]
        ),
    }
    return held_out, checks


def make_causal_source_run_id(
    config: CausalSourceTrainConfig, git_sha: str
) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    return (
        f"{timestamp}__P3__source-causal__p3-source-1d-v3__"
        f"seed{config.seed}__{git_sha[:7]}"
    )


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _train_impl(
    config: CausalSourceTrainConfig,
    *,
    session_epoch_limit: int | None,
) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_causal_source_run_id(config, git_sha)
    config = replace(config, run_id=run_id)
    run_dir = config.output_root.resolve() / run_id
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
    if config.resume and (run_dir / "DONE").is_file():
        raise RuntimeError(f"Completed run cannot be resumed: {run_dir}")
    previous_status = _read_previous_status(run_dir) if config.resume else None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

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
    started_path = run_dir / "STARTED_AT"
    if config.resume and started_path.is_file():
        started_at = started_path.read_text(encoding="utf-8").strip()
    else:
        started_at = session_started_at
        _atomic_write_text(started_path, f"{started_at}\n")
    sessions_path = run_dir / "run_sessions.jsonl"
    _append_session_record(
        sessions_path,
        {
            "event": "session_started",
            "timestamp": session_started_at,
            "resume": config.resume,
            "previous_status": previous_status,
            "unobserved_failed_attempt_overhead_possible": (
                bool(config.resume and previous_status != "paused")
            ),
        },
    )
    resume_evidence = _resume_evidence_from_sessions(sessions_path)
    _atomic_write_text(
        run_dir / "STATUS.json",
        _canonical_json({"status": "running", "timestamp": session_started_at}),
    )

    wall_started = time.perf_counter()
    tracemalloc.start()
    _configure_determinism(config.seed)
    device = _resolve_device(config.device)
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    checksums = _input_checksums(config)
    implementation_runtime_binding = _implementation_runtime_binding(
        config, device
    )
    _validate_implementation_runtime_binding(
        implementation_runtime_binding,
        label="current",
    )
    if not config.resume:
        _write_provenance(
            run_dir,
            config,
            device,
            git_sha,
            checksums,
            implementation_runtime_binding,
        )
    else:
        frozen = json.loads(
            (run_dir / "config_resolved.json").read_text(encoding="utf-8")
        )
        if frozen["scientific_config"] != _scientific_config(config):
            raise ValueError("Resume configuration differs from the frozen run.")
        frozen_checksums = json.loads(
            (run_dir / "data_checksums.json").read_text(encoding="utf-8")
        )
        if frozen_checksums != checksums:
            raise ValueError("Resume input checksums differ from the frozen run.")
        binding_path = run_dir / "implementation_runtime.json"
        if not binding_path.is_file():
            raise FileNotFoundError(
                "Resume implementation_runtime.json is missing."
            )
        frozen_binding = _validate_implementation_runtime_binding(
            json.loads(binding_path.read_text(encoding="utf-8")),
            label="frozen run",
        )
        if frozen_binding != implementation_runtime_binding:
            raise ValueError(
                "Resume implementation/runtime binding differs from "
                "the frozen run."
            )

    prepared = prepare_source_1d(
        config.data_path,
        config.split_manifest,
        time_stride=config.time_stride,
    )
    if tuple(prepared.channel_names) != tuple(INPUT_CHANNELS):
        raise ValueError("Prepared source channel order violates the frozen contract.")
    receptive_field = causal_receptive_field(config.depth)
    if receptive_field < prepared.inputs.shape[1]:
        raise ValueError(
            f"Temporal receptive field {receptive_field} is shorter than "
            f"Nt={prepared.inputs.shape[1]}."
        )
    model = CausalFactorizedOperator(
        input_channels=config.input_channels,
        width=config.width,
        depth=config.depth,
        modes_space=config.modes_space,
    ).to(device)
    actual_parameter_count = parameter_count(model)
    if (
        config.expected_parameter_count is not None
        and actual_parameter_count != config.expected_parameter_count
    ):
        raise ValueError(
            f"Parameter count {actual_parameter_count} differs from pinned "
            f"{config.expected_parameter_count}."
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
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    optimizer_updates_per_epoch = len(train_loader)
    if optimizer_updates_per_epoch < 1:
        raise ValueError("Source training loader contains no optimizer updates.")

    history_rows: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = -1
    best_model_state: dict[str, torch.Tensor] | None = None
    best_model_sha256: str | None = None
    bad_epochs = 0
    terminal_checkpoint_reentry = False
    if config.resume:
        if not last_path.is_file():
            raise FileNotFoundError("Resume requires last.pt.")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        history_rows, best_snapshot = _validate_last_checkpoint(
            checkpoint,
            model=model,
            config=config,
            checksums=checksums,
            normalization=prepared.normalization,
            git_sha=git_sha,
            implementation_runtime_binding=implementation_runtime_binding,
            parameter_count=actual_parameter_count,
            optimizer_updates_per_epoch=optimizer_updates_per_epoch,
            device_type=device.type,
            resume_evidence=None,
        )
        history_path = run_dir / "history.parquet"
        history_rows = _reconcile_history_artifact(
            history_path, history_rows
        )
        _reconcile_best_artifact(
            best_path,
            checkpoint,
            model=model,
            config=config,
            checksums=checksums,
            normalization=prepared.normalization,
            git_sha=git_sha,
            implementation_runtime_binding=implementation_runtime_binding,
            authoritative_history=history_rows,
            parameter_count=actual_parameter_count,
            optimizer_updates_per_epoch=optimizer_updates_per_epoch,
            device_type=device.type,
            resume_evidence=None,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["generator_state"])
        _restore_rng(checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(best_snapshot["epoch"])
        best_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in best_snapshot["model"].items()
        }
        best_model_sha256 = str(best_snapshot["model_sha256"])
        bad_epochs = int(checkpoint["bad_epochs"])
        terminal_checkpoint_reentry = (
            int(checkpoint["epoch"]) >= config.epochs
            or (
                int(checkpoint["epoch"]) >= config.minimum_epochs
                and bad_epochs >= config.early_stopping_patience
            )
        )
        if terminal_checkpoint_reentry:
            final_row = history_rows[-1]
            final_row.update(
                {
                    "resume_invocation_count": resume_evidence[
                        "resume_invocation_count"
                    ],
                    "controlled_resume_invocation_count": resume_evidence[
                        "controlled_resume_invocation_count"
                    ],
                    "unobserved_interruption_resume_count": resume_evidence[
                        "unobserved_interruption_resume_count"
                    ],
                    "replayed_optimizer_update_count": 0,
                    "replayed_optimizer_duration_seconds": 0.0,
                    "timing_complete": resume_evidence["timing_complete"],
                }
            )
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch=int(checkpoint["epoch"]),
                best_validation=best_validation,
                best_epoch=best_epoch,
                best_model_state=best_model_state,
                best_model_sha256=best_model_sha256,
                bad_epochs=bad_epochs,
                history_rows=history_rows,
                generator=generator,
                config=config,
                checksums=checksums,
                normalization=prepared.normalization,
                git_sha=git_sha,
                implementation_runtime_binding=(
                    implementation_runtime_binding
                ),
                parameter_count=actual_parameter_count,
                optimizer_updates_per_epoch=optimizer_updates_per_epoch,
                device_type=device.type,
                resume_evidence=resume_evidence,
            )
            _atomic_torch_save(checkpoint, last_path)
            _atomic_torch_save(
                _best_checkpoint_payload(checkpoint),
                best_path,
            )
            _atomic_parquet(
                pd.DataFrame(history_rows),
                run_dir / "history.parquet",
            )

    executed_this_session = 0
    stopped_early = bool(
        terminal_checkpoint_reentry
        and int(history_rows[-1]["epoch"]) < config.epochs
    )
    epoch_range = (
        ()
        if terminal_checkpoint_reentry
        else range(start_epoch, config.epochs + 1)
    )
    for epoch in epoch_range:
        epoch_started = time.perf_counter()
        model.train()
        totals = {
            "temperature": 0.0,
            "temperature_l4": 0.0,
            "alpha": 0.0,
            "gradient": 0.0,
            "weighted": 0.0,
        }
        train_count = 0
        last_gradient_norm = 0.0
        optimizer_update_durations: list[float] = []
        for inputs, temperature, alpha, _, _ in train_loader:
            _synchronize_for_timing(device)
            optimizer_update_started = time.perf_counter()
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            temperature = temperature.to(
                device, non_blocking=device.type == "cuda"
            )
            alpha = alpha.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            components = _source_loss_components(
                model(inputs), temperature, alpha, inputs[..., 4]
            )
            loss = _weighted_loss(components, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}."
                )
            loss.backward()
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
            batch_count = len(inputs)
            for name, value in components.items():
                totals[name] += float(value.detach()) * batch_count
            totals["weighted"] += float(loss.detach()) * batch_count
            train_count += batch_count
            _synchronize_for_timing(device)
            optimizer_update_durations.append(
                _strict_nonnegative_number(
                    time.perf_counter() - optimizer_update_started,
                    label=f"Epoch {epoch} optimizer update duration",
                )
            )
        scheduler.step()
        _synchronize_for_timing(device)
        candidate_started = time.perf_counter()
        validation = _evaluate_objective(
            model, prepared, "validation", config, device
        )
        _synchronize_for_timing(device)
        candidate_duration = _strict_nonnegative_number(
            time.perf_counter() - candidate_started,
            label=f"Epoch {epoch} candidate-validation duration",
        )
        if not all(np.isfinite(value) for value in validation.values()):
            raise FloatingPointError(
                f"Non-finite validation metric at epoch {epoch}."
            )
        improved = validation["weighted"] < best_validation
        if improved:
            best_validation = validation["weighted"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if improved:
            best_model_state = _clone_model_state_cpu(model)
            best_model_sha256 = _model_state_sha256(best_model_state)
        if best_model_state is None or best_model_sha256 is None:
            raise RuntimeError("Training has no authoritative best model state.")
        row = {
            "epoch": epoch,
            "train_temperature_relative_l2": totals["temperature"] / train_count,
            "train_temperature_normalized_l4": (
                totals["temperature_l4"] / train_count
            ),
            "train_alpha_relative_l2": totals["alpha"] / train_count,
            "train_spatial_gradient_relative_l2": (
                totals["gradient"] / train_count
            ),
            "train_weighted_objective": totals["weighted"] / train_count,
            "validation_temperature_relative_l2": validation["temperature"],
            "validation_temperature_normalized_l4": validation[
                "temperature_l4"
            ],
            "validation_alpha_relative_l2": validation["alpha"],
            "validation_spatial_gradient_relative_l2": validation["gradient"],
            "validation_weighted_objective": validation["weighted"],
            "gradient_norm_before_clip_last_batch": last_gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.perf_counter() - epoch_started,
            "duration_seconds_semantics": _EPOCH_WALL_TIME_SEMANTICS,
            "improved": improved,
            "selected_best_epoch": best_epoch,
            "selected_best_validation_objective": best_validation,
            "selected_best_model_sha256": best_model_sha256,
            "history_timing_schema_version": HISTORY_TIMING_SCHEMA_VERSION,
            "parameter_count": actual_parameter_count,
            "optimizer_update_count": len(optimizer_update_durations),
            "device_synchronized_training_seconds": float(
                math.fsum(optimizer_update_durations)
            ),
            "candidate_count": 1,
            "candidate_validation_seconds": candidate_duration,
            "optimizer_timing_clock_source": (
                OPTIMIZER_TIMING_CLOCK_SOURCE
            ),
            "optimizer_device_synchronization": (
                _timing_device_synchronization(device.type)
            ),
            "candidate_timing_clock_source": (
                CANDIDATE_TIMING_CLOCK_SOURCE
            ),
            "candidate_device_synchronization": (
                _timing_device_synchronization(device.type)
            ),
            "candidate_validation_time_in_optimizer_total": False,
            "resume_invocation_count": resume_evidence[
                "resume_invocation_count"
            ],
            "controlled_resume_invocation_count": resume_evidence[
                "controlled_resume_invocation_count"
            ],
            "unobserved_interruption_resume_count": resume_evidence[
                "unobserved_interruption_resume_count"
            ],
            "replayed_optimizer_update_count": 0,
            "replayed_optimizer_duration_seconds": 0.0,
            "timing_complete": resume_evidence["timing_complete"],
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
            best_model_sha256=best_model_sha256,
            bad_epochs=bad_epochs,
            history_rows=history_rows,
            generator=generator,
            config=config,
            checksums=checksums,
            normalization=prepared.normalization,
            git_sha=git_sha,
            implementation_runtime_binding=implementation_runtime_binding,
            parameter_count=actual_parameter_count,
            optimizer_updates_per_epoch=optimizer_updates_per_epoch,
            device_type=device.type,
            resume_evidence=resume_evidence,
        )
        _atomic_torch_save(checkpoint, last_path)
        _atomic_torch_save(_best_checkpoint_payload(checkpoint), best_path)
        _atomic_parquet(
            pd.DataFrame(history_rows), run_dir / "history.parquet"
        )
        executed_this_session += 1
        if epoch == start_epoch or epoch % 10 == 0 or epoch == config.epochs:
            logger.info(
                "epoch=%d/%d train=%.6f validation=%.6f grad_norm=%.4f seconds=%.3f",
                epoch,
                config.epochs,
                row["train_weighted_objective"],
                row["validation_weighted_objective"],
                last_gradient_norm,
                row["duration_seconds"],
            )
        if (
            epoch >= config.minimum_epochs
            and bad_epochs >= config.early_stopping_patience
        ):
            stopped_early = True
            logger.info("early_stop epoch=%d bad_epochs=%d", epoch, bad_epochs)
            break
        if (
            session_epoch_limit is not None
            and executed_this_session >= session_epoch_limit
            and epoch < config.epochs
        ):
            paused_at = datetime.now().astimezone().isoformat()
            _append_session_record(
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
            tracemalloc.stop()
            _close_logger(logger)
            return {
                "status": "paused",
                "run_id": run_id,
                "last_epoch": epoch,
                "held_out_labels_evaluated": False,
            }

    if not best_path.is_file():
        raise RuntimeError("Training completed without a selected best checkpoint.")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    selected_generation = _validate_best_checkpoint(
        selected,
        model=model,
        config=config,
        checksums=checksums,
        normalization=prepared.normalization,
        git_sha=git_sha,
        implementation_runtime_binding=implementation_runtime_binding,
        authoritative_history=history_rows,
        parameter_count=actual_parameter_count,
        optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device.type,
        resume_evidence=resume_evidence,
    )
    if selected_generation != int(history_rows[-1]["epoch"]):
        raise RuntimeError("Selected checkpoint is not at the final generation.")
    model.load_state_dict(selected["model"], strict=True)
    selected_epoch = int(selected["epoch"])

    split_summaries: dict[str, dict[str, float | int]] = {}
    frames: list[pd.DataFrame] = []
    held_out_names = sorted(prepared.held_out_families)
    for split in ("in_family_test", *held_out_names):
        summary, frame = _evaluate_split_metrics(
            model, prepared, split, config, device
        )
        split_summaries[split] = summary
        frames.append(frame)
    per_case = pd.concat(frames, ignore_index=True)
    _atomic_parquet(per_case, run_dir / "metrics_per_case.parquet")
    held_out, checks = _acceptance_summary(split_summaries, held_out_names)

    validation_ids = prepared.splits["validation"][: config.causality_case_count]
    validation_indices = torch.tensor(validation_ids, dtype=torch.long)
    causality = check_future_invariance(
        model,
        prepared.inputs[validation_indices],
        prepared.case_ids[validation_indices],
        tolerance=config.causality_tolerance,
        cutoff_fractions=config.causality_cutoff_fractions,
        device=device,
    )
    _atomic_write_text(
        run_dir / "causality.json", _canonical_json(causality)
    )

    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    completed_at = datetime.now().astimezone().isoformat()
    final_wall_seconds = time.perf_counter() - wall_started
    _append_session_record(
        sessions_path,
        {
            "event": "session_completed",
            "timestamp": completed_at,
            "wall_seconds": final_wall_seconds,
        },
    )
    session_records = [
        json.loads(line)
        for line in sessions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    final_resume_evidence = _resume_evidence_from_sessions(sessions_path)
    if final_resume_evidence != resume_evidence:
        raise RuntimeError(
            "Run-session evidence changed during the training session."
        )
    compute_timing = causal_source_compute_timing_from_history(
        history_rows,
        expected_parameter_count=actual_parameter_count,
        expected_optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device.type,
        expected_resume_evidence=final_resume_evidence,
    )
    compute_accounting = _compute_accounting(compute_timing)
    history_path = run_dir / "history.parquet"
    history_artifact = _artifact_reference(
        history_path,
        project_root,
        allow_external=True,
    )
    sessions_artifact = _artifact_reference(
        sessions_path,
        project_root,
        allow_external=True,
    )
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "phase": "P3",
        "experiment": EXPERIMENT,
        "model": {
            **_model_spec(config),
            "parameter_count": actual_parameter_count,
        },
        "seed": config.seed,
        "selection": {
            "split": "validation",
            "objective": "weighted_source_v4_objective",
            "selected_epoch": selected_epoch,
            "best_validation_objective": float(selected["best_validation"]),
            "held_out_labels_used": False,
        },
        "training": {
            "configured_epochs": config.epochs,
            "executed_epochs": len(history_rows),
            "stopped_early": stopped_early,
            "batch_size": config.batch_size,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "device": str(device),
            "summed_epoch_seconds": float(
                sum(float(row["duration_seconds"]) for row in history_rows)
            ),
            "summed_epoch_seconds_semantics": (
                _EPOCH_WALL_TIME_SEMANTICS
            ),
            "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
            "train_case_count": len(prepared.splits["train"]),
            "validation_case_count": len(prepared.splits["validation"]),
            "final_session_wall_seconds": final_wall_seconds,
        },
        "compute_timing": compute_timing,
        "compute_accounting": compute_accounting,
        "training_history_artifact": history_artifact,
        "run_sessions_artifact": sessions_artifact,
        "loss_weights": {
            "temperature_relative_l2": config.temperature_weight,
            "temperature_normalized_l4": config.temperature_l4_weight,
            "alpha_relative_l2": config.alpha_weight,
            "spatial_gradient_relative_l2": config.gradient_weight,
        },
        "normalization": prepared.normalization,
        "input_checksums": checksums,
        "implementation_runtime_binding": implementation_runtime_binding,
        "checkpoints": {
            "best": {
                "path": _portable_path(best_path, project_root),
                "sha256": sha256_file(best_path),
                "selected_epoch": selected_epoch,
            },
            "last": {
                "path": _portable_path(last_path, project_root),
                "sha256": sha256_file(last_path),
                "epoch": int(
                    torch.load(
                        last_path, map_location="cpu", weights_only=False
                    )["epoch"]
                ),
            },
        },
        "split_metrics": split_summaries,
        "held_out_combined": held_out,
        "acceptance_thresholds_prespecified": ACCEPTANCE,
        "acceptance_checks": checks,
        "field_acceptance_passed": bool(all(checks.values())),
        "causality": {
            key: value for key, value in causality.items() if key != "rows"
        },
        "causality_passed": bool(causality["passed"]),
        "passed": bool(all(checks.values()) and causality["passed"]),
        "recorded_session_count": sum(
            record["event"] == "session_started" for record in session_records
        ),
        "resume_session_count": sum(
            record["event"] == "session_started" and record["resume"]
            for record in session_records
        ),
        "peak_python_tracemalloc_bytes": tracemalloc_peak,
        "peak_accelerator_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "started_at": started_at,
        "completed_at": completed_at,
        "git_sha": git_sha,
    }
    _atomic_write_text(run_dir / "metrics.json", _canonical_json(metrics))
    _atomic_write_text(
        run_dir / "STATUS.json",
        _canonical_json({"status": "completed", "timestamp": completed_at}),
    )
    receipt_unsigned = {
        "schema_version": 1,
        "phase": "P3",
        "artifact_role": "causal_source_terminal_compute_receipt",
        "status": "completed",
        "run_id": run_id,
        "seed": config.seed,
        "completed_at": completed_at,
        "compute_timing": compute_timing,
        "compute_accounting": compute_accounting,
        "artifacts": {
            "metrics.json": _artifact_reference(
                run_dir / "metrics.json",
                project_root,
                allow_external=True,
            ),
            "history.parquet": history_artifact,
            "run_sessions.jsonl": sessions_artifact,
            "checkpoints/best.pt": _artifact_reference(
                best_path,
                project_root,
                allow_external=True,
            ),
            "checkpoints/last.pt": _artifact_reference(
                last_path,
                project_root,
                allow_external=True,
            ),
        },
        "held_out_labels_used_for_selection": False,
        "legacy_epoch_duration_used_for_compute_accounting": False,
    }
    receipt = {
        **receipt_unsigned,
        "receipt_payload_sha256": _canonical_sha256(receipt_unsigned),
    }
    _atomic_write_bytes(run_dir / "DONE", _canonical_json_bytes(receipt))
    failed_marker = run_dir / "FAILED"
    if failed_marker.is_file():
        os.replace(failed_marker, run_dir / "FAILED_RECOVERED")
    logger.info(
        "completed run_id=%s selected_epoch=%d heldout_T=%.6f causal=%s passed=%s",
        run_id,
        selected_epoch,
        held_out["temperature_relative_l2_mean"],
        causality["passed"],
        metrics["passed"],
    )
    _close_logger(logger)
    return metrics


def validate_causal_source_compute_evidence(
    run_dir: Path,
    *,
    project_root: Path,
    expected_run_id: str | None = None,
    expected_seed: int | None = None,
    allow_external_run: bool = False,
) -> dict[str, Any]:
    """Validate a completed exact-timing source run without inference."""

    root = project_root.resolve(strict=True)
    directory = run_dir.resolve(strict=True)
    if (
        run_dir.is_symlink()
        or not directory.is_dir()
        or (
            not allow_external_run
            and not directory.is_relative_to(root)
        )
    ):
        raise ValueError("Causal-source run directory is unsafe.")
    paths = {
        "DONE": directory / "DONE",
        "metrics.json": directory / "metrics.json",
        "history.parquet": directory / "history.parquet",
        "run_sessions.jsonl": directory / "run_sessions.jsonl",
        "checkpoints/best.pt": directory / "checkpoints" / "best.pt",
        "checkpoints/last.pt": directory / "checkpoints" / "last.pt",
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "Causal-source exact compute evidence is incomplete."
        )
    raw_receipt = paths["DONE"].read_bytes()
    try:
        receipt = json.loads(raw_receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Causal-source DONE is not a JSON terminal receipt."
        ) from error
    expected_receipt_keys = {
        "schema_version",
        "phase",
        "artifact_role",
        "status",
        "run_id",
        "seed",
        "completed_at",
        "compute_timing",
        "compute_accounting",
        "artifacts",
        "held_out_labels_used_for_selection",
        "legacy_epoch_duration_used_for_compute_accounting",
        "receipt_payload_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or raw_receipt != _canonical_json_bytes(receipt)
    ):
        raise ValueError(
            "Causal-source terminal receipt shape/canonical form differs."
        )
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    run_id = receipt.get("run_id")
    seed = receipt.get("seed")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("phase") != "P3"
        or receipt.get("artifact_role")
        != "causal_source_terminal_compute_receipt"
        or receipt.get("status") != "completed"
        or type(run_id) is not str
        or not run_id
        or type(seed) is not int
        or seed < 0
        or receipt.get("held_out_labels_used_for_selection") is not False
        or receipt.get(
            "legacy_epoch_duration_used_for_compute_accounting"
        )
        is not False
        or receipt.get("receipt_payload_sha256")
        != _canonical_sha256(unsigned)
        or (expected_run_id is not None and run_id != expected_run_id)
        or (expected_seed is not None and seed != expected_seed)
    ):
        raise ValueError("Causal-source terminal receipt identity differs.")
    expected_artifacts = {
        name: _artifact_reference(
            path,
            root,
            allow_external=allow_external_run,
        )
        for name, path in paths.items()
        if name != "DONE"
    }
    if receipt.get("artifacts") != expected_artifacts:
        raise ValueError(
            "Causal-source terminal receipt artifact binding differs."
        )
    try:
        metrics = json.loads(
            paths["metrics.json"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Causal-source metrics are invalid.") from error
    if (
        not isinstance(metrics, dict)
        or metrics.get("status") != "completed"
        or metrics.get("run_id") != run_id
        or metrics.get("seed") != seed
        or metrics.get("completed_at") != receipt.get("completed_at")
        or metrics.get("training_history_artifact")
        != expected_artifacts["history.parquet"]
        or metrics.get("run_sessions_artifact")
        != expected_artifacts["run_sessions.jsonl"]
    ):
        raise ValueError("Causal-source metrics identity/binding differs.")
    implementation_runtime_binding = (
        _validate_implementation_runtime_binding(
            metrics.get("implementation_runtime_binding"),
            label="terminal metrics",
        )
    )
    if (
        implementation_runtime_binding["manifest"].get(
            "implementation_files"
        )
        != _implementation_file_manifest(root)
    ):
        raise ValueError(
            "Causal-source terminal implementation differs from live "
            "scientific source files."
        )
    try:
        parameter_count = metrics["model"]["parameter_count"]
        optimizer_updates_per_epoch = metrics["training"][
            "optimizer_updates_per_epoch"
        ]
        device_value = metrics["training"]["device"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Causal-source metrics lack exact timing dimensions."
        ) from error
    if type(device_value) is not str:
        raise ValueError("Causal-source metrics device is invalid.")
    device_type = torch.device(device_value).type
    try:
        history = pd.read_parquet(paths["history.parquet"])
    except (OSError, ValueError, ImportError) as error:
        raise ValueError(
            "Causal-source timing history cannot be read."
        ) from error
    resume_evidence = _resume_evidence_from_sessions(
        paths["run_sessions.jsonl"]
    )
    compute_timing = causal_source_compute_timing_from_history(
        history,
        expected_parameter_count=parameter_count,
        expected_optimizer_updates_per_epoch=optimizer_updates_per_epoch,
        device_type=device_type,
        expected_resume_evidence=resume_evidence,
    )
    compute_accounting = _compute_accounting(compute_timing)
    if (
        metrics.get("compute_timing") != compute_timing
        or metrics.get("compute_accounting") != compute_accounting
        or receipt.get("compute_timing") != compute_timing
        or receipt.get("compute_accounting") != compute_accounting
    ):
        raise ValueError(
            "Causal-source metrics/receipt compute repeats differ from history."
        )
    last = torch.load(
        paths["checkpoints/last.pt"],
        map_location="cpu",
        weights_only=False,
    )
    best = torch.load(
        paths["checkpoints/best.pt"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(last, Mapping) or not isinstance(best, Mapping):
        raise ValueError("Causal-source checkpoints must be mappings.")
    authoritative = last.get("authoritative_history")
    if (
        not isinstance(authoritative, list)
        or _plain_history_rows(authoritative) != _plain_history_rows(history)
        or last.get("compute_timing") != compute_timing
        or last.get("compute_accounting") != compute_accounting
        or best.get("compute_timing") != compute_timing
        or best.get("compute_accounting") != compute_accounting
        or last.get("implementation_runtime_binding")
        != implementation_runtime_binding
        or best.get("implementation_runtime_binding")
        != implementation_runtime_binding
        or last.get("artifact_generation") != len(history)
        or best.get("artifact_generation") != len(history)
    ):
        raise ValueError(
            "Causal-source checkpoint/history compute binding differs."
        )
    checkpoint_metrics = metrics.get("checkpoints")
    if (
        not isinstance(checkpoint_metrics, Mapping)
        or checkpoint_metrics.get("best", {}).get("sha256")
        != expected_artifacts["checkpoints/best.pt"]["sha256"]
        or checkpoint_metrics.get("last", {}).get("sha256")
        != expected_artifacts["checkpoints/last.pt"]["sha256"]
    ):
        raise ValueError(
            "Causal-source metrics checkpoint hashes differ."
        )
    return {
        "run_id": run_id,
        "seed": seed,
        "receipt": receipt,
        "compute_timing": compute_timing,
        "compute_accounting": compute_accounting,
        "artifacts": expected_artifacts,
        "passed": True,
    }


def train_causal_source(
    config: CausalSourceTrainConfig,
    *,
    session_epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Train or resume a causal source run without touching legacy P3 outputs."""

    config = config.validated()
    if session_epoch_limit is not None and session_epoch_limit < 1:
        raise ValueError("session_epoch_limit must be positive.")
    project_root = config.project_root.resolve()
    git_sha = _git(project_root, "rev-parse", "HEAD")
    run_id = config.run_id or make_causal_source_run_id(config, git_sha)
    resolved = replace(config, run_id=run_id)
    # P6 orchestration is intentionally absent from the public P0--P5 release.
    # Fail closed for its two reserved source-run IDs while leaving ordinary
    # P3 behavior unchanged and free of a private P6 module dependency.
    reserved_p6_run_ids = {
        "p3-source-causal-v1-seed1-fix1",
        "p3-source-causal-v1-seed2-fix1",
    }
    if run_id in reserved_p6_run_ids:
        raise ValueError(
            "This P6 supplemental run ID is not part of public release v0.0.1."
        )
    run_dir = resolved.output_root.resolve() / run_id
    if run_dir.exists() and not resolved.resume:
        raise FileExistsError(f"Run already exists; use --resume: {run_dir}")
    try:
        return _train_impl(
            resolved, session_epoch_limit=session_epoch_limit
        )
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        failed_at = datetime.now().astimezone().isoformat()
        _append_session_record(
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
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_causal_source_config(
    path: Path,
    *,
    project_root: Path,
    device: str | None = None,
    run_id: str | None = None,
    output_root: Path | None = None,
    epochs: int | None = None,
    resume: bool = False,
) -> CausalSourceTrainConfig:
    """Load and strictly validate the canonical YAML configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Causal source config must be a YAML mapping.")
    if payload.get("schema_version") != 1 or payload.get("phase") != "P3":
        raise ValueError("Unsupported causal source config schema or phase.")
    if payload.get("experiment") != EXPERIMENT:
        raise ValueError("Unexpected causal source experiment name.")
    if payload.get("acceptance") != ACCEPTANCE:
        raise ValueError("Config acceptance thresholds differ from P3 source v4.")
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    causality = payload["causality"]
    outputs = payload["outputs"]
    expected_dilations = [
        2 ** (index % 8) for index in range(int(model["depth"]))
    ]
    declarations = {
        "model.family": model.get("family") == MODEL_FAMILY,
        "model.temporal_kernel_size": model.get("temporal_kernel_size") == 3,
        "model.temporal_dilations": (
            model.get("temporal_dilations") == expected_dilations
        ),
        "model.temporal_receptive_field": (
            model.get("temporal_receptive_field")
            == causal_receptive_field(int(model["depth"]))
        ),
        "model.structurally_causal": model.get("structurally_causal") is True,
        "training.mixed_precision": training.get("mixed_precision") is False,
        "training.checkpoint_selection": (
            training.get("checkpoint_selection")
            == "validation_weighted_source_v4_objective"
        ),
        "training.held_out_labels_used_for_selection": (
            training.get("held_out_labels_used_for_selection") is False
        ),
        "training.compute_timing": (
            training.get("compute_timing", COMPUTE_TIMING_CONFIG)
            == COMPUTE_TIMING_CONFIG
        ),
        "causality.perturb_dynamic_channels_only": (
            causality.get("perturb_dynamic_channels_only") is True
        ),
    }
    invalid_declarations = [
        name for name, valid in declarations.items() if not valid
    ]
    if invalid_declarations:
        raise ValueError(
            "Causal source config violates frozen declarations: "
            f"{invalid_declarations}."
        )
    configured_output = (
        _resolve_repo_path(project_root, output_root)
        if output_root is not None
        else _resolve_repo_path(project_root, outputs["root"])
    )
    config = CausalSourceTrainConfig(
        data_path=_resolve_repo_path(project_root, data["path"]),
        split_manifest=_resolve_repo_path(
            project_root, data["split_manifest"]
        ),
        output_root=configured_output,
        project_root=project_root.resolve(),
        run_id=run_id or outputs.get("run_id"),
        config_file=path.resolve(),
        expected_data_sha256=data["expected_data_sha256"],
        expected_split_sha256=data["expected_split_sha256"],
        seed=int(payload["seed"]),
        time_stride=int(data["time_stride"]),
        input_channels=int(model["input_channels"]),
        width=int(model["width"]),
        depth=int(model["depth"]),
        modes_space=int(model["modes_space"]),
        expected_parameter_count=int(model["expected_parameter_count"]),
        epochs=int(epochs if epochs is not None else training["epochs"]),
        minimum_epochs=int(training["minimum_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        temperature_weight=float(
            training["loss_weights"]["temperature_relative_l2"]
        ),
        temperature_l4_weight=float(
            training["loss_weights"]["temperature_normalized_l4"]
        ),
        alpha_weight=float(training["loss_weights"]["alpha_relative_l2"]),
        gradient_weight=float(
            training["loss_weights"]["spatial_gradient_relative_l2"]
        ),
        gradient_clip=float(training["gradient_clip"]),
        device=device or str(training["device"]),
        num_threads=int(training["num_threads"]),
        causality_tolerance=float(causality["tolerance"]),
        causality_case_count=int(causality["validation_case_count"]),
        causality_cutoff_fractions=tuple(
            float(value) for value in causality["cutoff_fractions"]
        ),
        compute_timing_schema_version=int(
            training.get("compute_timing", COMPUTE_TIMING_CONFIG)[
                "schema_version"
            ]
        ),
        resume=resume,
    )
    if epochs is not None and config.minimum_epochs > epochs:
        config = replace(config, minimum_epochs=epochs)
    return config.validated()


def causal_source_dry_run(
    config: CausalSourceTrainConfig,
) -> dict[str, Any]:
    """Validate paths, hashes, architecture, and device without creating a run."""

    config = config.validated()
    from cdcureno.training.p6_v2_supplemental_source import (
        validate_reserved_source_training_authorization,
    )

    validate_reserved_source_training_authorization(config)
    data_exists = config.data_path.is_file()
    split_exists = config.split_manifest.is_file()
    checksums: dict[str, Any] | None = None
    checksum_error: str | None = None
    time_count: int | None = None
    split_counts: dict[str, Any] | None = None
    if data_exists and split_exists:
        try:
            checksums = _input_checksums(config)
            with np.load(config.data_path) as arrays:
                raw_time_count = int(arrays["air_temperature_K"].shape[1])
            indices = list(range(0, raw_time_count, config.time_stride))
            if indices[-1] != raw_time_count - 1:
                indices.append(raw_time_count - 1)
            time_count = len(indices)
            manifest = json.loads(
                config.split_manifest.read_text(encoding="utf-8")
            )
            split_counts = {
                "train": len(manifest["splits"]["train"]),
                "validation": len(manifest["splits"]["validation"]),
                "in_family_test": len(manifest["splits"]["in_family_test"]),
                "held_out_family_test": {
                    name: len(ids)
                    for name, ids in manifest["splits"][
                        "held_out_family_test"
                    ].items()
                },
            }
        except (KeyError, OSError, ValueError) as error:
            checksum_error = f"{type(error).__name__}: {error}"
    model = CausalFactorizedOperator(
        input_channels=config.input_channels,
        width=config.width,
        depth=config.depth,
        modes_space=config.modes_space,
    )
    actual_parameters = parameter_count(model)
    receptive_field = causal_receptive_field(config.depth)
    requested_device_available = not (
        torch.device(config.device if config.device != "auto" else "cpu").type
        == "cuda"
        and not torch.cuda.is_available()
    )
    checks = {
        "data_exists": data_exists,
        "split_exists": split_exists,
        "input_hashes_match": checksums is not None and checksum_error is None,
        "parameter_count_matches": (
            config.expected_parameter_count is None
            or actual_parameters == config.expected_parameter_count
        ),
        "full_time_receptive_field": (
            time_count is not None and receptive_field >= time_count
        ),
        "requested_device_available": requested_device_available,
    }
    return {
        "schema_version": 1,
        "dry_run": True,
        "experiment": EXPERIMENT,
        "run_id": config.run_id,
        "config": _scientific_config(config),
        "compute_timing_contract": COMPUTE_TIMING_CONFIG,
        "model": {
            **_model_spec(config),
            "parameter_count": actual_parameters,
        },
        "time_count": time_count,
        "split_counts": split_counts,
        "checksums": checksums,
        "checksum_error": checksum_error,
        "device_probe": {
            "requested": config.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
