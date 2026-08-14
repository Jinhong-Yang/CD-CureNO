"""Auditable causal target training for the P6 matrix.

This module is intentionally separate from the frozen noncausal P5 pilot.
Only the P4 nested training prefixes and validation labels are exposed here;
ID-test and OOD loaders are not imported.  Test/OOD evaluation is released by
a separate, checkpoint-hash-bound path after all validation-selected models
have been frozen.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cdcureno.data.source_1d import INPUT_CHANNELS, prepare_source_1d
from cdcureno.data.target_2d import (
    TARGET_INPUT_CHANNELS,
    PreparedTarget2DTraining,
    prepare_target_2d_training,
)
from cdcureno.models.p6_initialization import (
    P6_INITIALIZATION_METHODS,
    P6InitializationMethod,
    initialize_p6_causal_target,
    state_dict_sha256,
)
from cdcureno.models.p6_ablations import (
    MatchedTwoSidedTemporalConv1d,
    P6_ABLATION_IDS,
    P6_ABLATIONS,
    P6_TRAINED_ABLATION_IDS,
    P6AblationId,
    apply_p6_ablation,
)
from cdcureno.models.joint_operators import CausalTemporalConv1d
from cdcureno.models.target_operators import CausalAxisFactorized2DOperator
from cdcureno.physics import CureKinetics, PublicCase1Material
from cdcureno.physics.target_2d_residuals import (
    Target2DPhysicsContext,
    target_2d_physics_residuals,
)
from cdcureno.training import target_2d as p5
from cdcureno.training.p6_restriction import (
    FrozenCausalSource,
    SourceTeacherOutputs,
    audit_selected_model_restriction,
    load_frozen_causal_source,
    precompute_source_teacher,
    source_output_restriction_loss,
)
from cdcureno.training.p6_roster import p6_development_run_id
from cdcureno.training.process_liveness import windows_process_is_live


EXPERIMENT = "p6_causal_target_matrix_v1"
_IS_WINDOWS = os.name == "nt"
CHECKPOINT_SCHEMA_VERSION = 1
RESOURCE_PROFILE_SCHEMA_VERSION = 1
P6_BUDGETS = (8, 16, 32, 64, 128, 256)
P6_MAIN_SEEDS = (0, 1, 2, 3, 4)
P6_DEVELOPMENT_SEED = 99
P6_MAIN_METHODS: tuple[P6InitializationMethod, ...] = (
    "generic_causal_transfer",
    "cdcureno_full",
)
P6_SECONDARY_METHODS: tuple[P6InitializationMethod, ...] = (
    "scratch_causal",
    "ordinary_causal_transfer",
)
P6_METHODS = (*P6_INITIALIZATION_METHODS, *P6_ABLATION_IDS)
P6_ABLATION_SEEDS = (0, 1, 2)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_PARAMETER_GROUP_RATIOS = {
    "adapter": 1.0,
    "lift_and_heads": 0.5,
    "shared_core": 0.1,
}
SCRATCH_PARAMETER_GROUP_RATIOS = {
    "adapter": 1.0,
    "lift_and_heads": 1.0,
    "shared_core": 1.0,
}


class ResumeAuthorizationError(RuntimeError):
    """The existing run is not in a protocol-authorized resumable state."""


def _ablation_spec(method: str):
    return P6_ABLATIONS.get(method)  # type: ignore[arg-type]


def _initializer_method(method: str) -> P6InitializationMethod:
    spec = _ablation_spec(method)
    if spec is None:
        if method not in P6_INITIALIZATION_METHODS:
            raise ValueError(f"Unknown P6 method: {method}")
        return method  # type: ignore[return-value]
    if method == "A7":
        raise ValueError(
            "A7 is an alias of cdcureno_full at budget 32 and must reuse "
            "that checkpoint; a new A7 training run is forbidden."
        )
    return spec.initializer  # type: ignore[return-value]


def _restriction_enabled(method: str) -> bool:
    spec = _ablation_spec(method)
    return (
        method == "cdcureno_full"
        if spec is None
        else bool(spec.restriction)
    )


def _physics_enabled(method: str) -> bool:
    spec = _ablation_spec(method)
    return True if spec is None else bool(spec.physics)


def _expected_causal(method: str) -> bool:
    spec = _ablation_spec(method)
    return True if spec is None else bool(spec.causal_time)


def _effective_physics_multiplier(config: "P6CausalTrainConfig") -> float:
    return float(config.physics_weight) if _physics_enabled(config.method) else 0.0


def _effective_restriction_multiplier(
    config: "P6CausalTrainConfig",
) -> float:
    return (
        float(config.source_restriction_multiplier)
        if _restriction_enabled(config.method)
        else 0.0
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


def _strict_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def _sha256_json(payload: Any) -> str:
    value = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(
            json.dumps(
                dict(payload), allow_nan=False, sort_keys=True
            ) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _known_session_wall_summary(
    path: Path,
    *,
    current_completed_wall_seconds: float = 0.0,
) -> dict[str, Any]:
    terminal_events = {
        "session_paused",
        "session_completed",
        "session_interrupted",
        "session_failed",
    }
    known: list[float] = []
    unknown_count = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "session_interruption_detected":
                unknown_count += 1
            if event.get("event") in terminal_events:
                value = event.get("wall_seconds")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    known.append(float(value))
                else:
                    unknown_count += 1
    if current_completed_wall_seconds > 0.0:
        known.append(float(current_completed_wall_seconds))
    return {
        "known_session_wall_seconds": float(sum(known)),
        "known_terminal_session_count": len(known),
        "unknown_interrupted_session_count": unknown_count,
        "wall_time_complete": unknown_count == 0,
    }


def _portable(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Run artifact is missing: {path}")
    return {
        "path": _portable(path, root),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


IMPLEMENTATION_FILES = (
    "scripts/run_p6_roster.py",
    "scripts/train_p6_causal_target.py",
    "src/cdcureno/training/causal_target_2d.py",
    "src/cdcureno/training/target_2d.py",
    "src/cdcureno/training/p6_restriction.py",
    "src/cdcureno/training/p6_roster.py",
    "src/cdcureno/data/target_2d.py",
    "src/cdcureno/data/source_1d.py",
    "src/cdcureno/data/normalization.py",
    "src/cdcureno/models/p6_initialization.py",
    "src/cdcureno/models/p6_ablations.py",
    "src/cdcureno/models/causal_checkpoint_inflation.py",
    "src/cdcureno/models/target_operators.py",
    "src/cdcureno/models/joint_operators.py",
    "src/cdcureno/physics/target_2d_residuals.py",
    "src/cdcureno/physics/as4_8552.py",
)


def _implementation_state(root: Path) -> dict[str, Any]:
    # Hash the complete local Python package in addition to the entry point.
    # This deliberately over-approximates the runtime import closure so a
    # transitive solver, input-builder, sampler, or fallback change cannot
    # evade the P6 implementation lock.
    package_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "cdcureno").rglob("*.py"))
        if path.is_file()
    )
    implementation_files = tuple(
        sorted(set((*IMPLEMENTATION_FILES, *package_files)))
    )
    hashes: dict[str, str] = {}
    for relative in implementation_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"P6 implementation source is missing: {path}"
            )
        hashes[relative] = _sha256_file(path)
    return {
        "head_sha": _git(root, "rev-parse", "HEAD"),
        "status_short": _git(root, "status", "--short"),
        "source_file_sha256": hashes,
        "source_file_count": len(hashes),
        "closure_policy": (
            "entrypoint_plus_all_repository_src_cdcureno_python_files"
        ),
        "implementation_sha256": _sha256_json(hashes),
    }


def _configure_determinism(seed: int) -> None:
    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "P6 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA "
            f"initialization; received {configured_workspace!r}."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        index = (
            torch.cuda.current_device()
            if device.index is None
            else int(device.index)
        )
        torch.cuda.get_device_properties(index)
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ValueError("P6 device must be cpu, cuda, cuda:N, or auto.")
    return device


def _device_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resolved": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload["active_device"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [
                int(properties.major),
                int(properties.minor),
            ],
        }
    return payload


@dataclass(frozen=True)
class P6CausalTrainConfig:
    """Resolved P6 development or confirmatory training contract."""

    project_root: Path
    config_file: Path
    protocol_role: str
    target_split_manifest: Path
    source_checkpoint: Path
    inflated_checkpoint: Path
    target_model_config: Path
    source_data_path: Path
    source_split_manifest: Path
    p4_plan_path: Path
    material_config_path: Path
    output_root: Path
    resource_profile_path: Path
    method: P6InitializationMethod | P6AblationId
    label_budget: int
    seed: int
    physics_weight: float
    expected_target_split_sha256: str
    expected_source_checkpoint_sha256: str
    expected_inflated_checkpoint_sha256: str
    expected_target_model_config_sha256: str
    expected_source_data_sha256: str
    expected_source_split_sha256: str
    expected_p4_plan_sha256: str
    expected_material_config_sha256: str
    expected_parameter_count: int = 215_698
    epochs: int = 120
    effective_batch_size: int = 4
    candidate_micro_batch_sizes: tuple[int, ...] = (2, 1)
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    temperature_weight: float = 1.0
    alpha_weight: float = 0.5
    gradient_x_weight: float = 0.05
    gradient_z_weight: float = 0.05
    energy_component_weight: float = 1.0
    kinetics_component_weight: float = 0.25
    initial_condition_component_weight: float = 0.25
    source_temperature_restriction_weight: float = 1.0
    source_alpha_restriction_weight: float = 1.0
    source_lateral_invariance_weight: float = 1.0
    source_restriction_multiplier: float = 0.1
    restriction_validation_case_count: int = 8
    restriction_rvs_max: float = 0.25
    restriction_lis_max: float = 1.0e-6
    device: str = "cuda"
    num_threads: int = 8
    verify_array_checksums: bool = True
    require_resource_profile: bool = True
    run_id: str | None = None
    resume: bool = False

    def validated(
        self, *, require_resolved_resources: bool = False
    ) -> "P6CausalTrainConfig":
        if type(self.resume) is not bool:
            raise ValueError("resume must be boolean.")
        if self.verify_array_checksums is not True:
            raise ValueError("P6 array checksum verification is mandatory.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string.")
        integer_fields = {
            "label_budget": self.label_budget,
            "seed": self.seed,
            "expected_parameter_count": self.expected_parameter_count,
            "epochs": self.epochs,
            "effective_batch_size": self.effective_batch_size,
            "restriction_validation_case_count": (
                self.restriction_validation_case_count
            ),
            "num_threads": self.num_threads,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise ValueError(
                "P6 integer settings cannot be booleans or coerced scalars."
            )
        for name, value in (
            ("micro_batch_size", self.micro_batch_size),
            (
                "gradient_accumulation_steps",
                self.gradient_accumulation_steps,
            ),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive.")
        if self.run_id is not None and (
            not isinstance(self.run_id, str)
            or RUN_ID_PATTERN.fullmatch(self.run_id) is None
        ):
            raise ValueError(
                "run_id must be a single safe ASCII path component."
            )
        if self.protocol_role not in {"development", "confirmatory"}:
            raise ValueError("protocol_role must be development or confirmatory.")
        if self.method not in P6_METHODS:
            raise ValueError(f"Unknown P6 method: {self.method}")
        if self.method == "A7":
            raise ValueError(
                "A7 is an alias of cdcureno_full at budget 32 and must "
                "reuse the corresponding full-model checkpoint."
            )
        if self.label_budget not in P6_BUDGETS:
            raise ValueError("P6 label budget is not a frozen nested prefix.")
        if self.protocol_role == "development":
            if (
                self.method not in P6_MAIN_METHODS
                or self.seed != P6_DEVELOPMENT_SEED
                or self.label_budget != 8
                or self.epochs != 40
                or self.effective_batch_size != 4
            ):
                raise ValueError(
                    "Development is restricted to B7/P, budget 8, seed 99, "
                    "40 epochs, and effective batch 4."
                )
        elif self.epochs != 120 or self.effective_batch_size != 4:
            raise ValueError(
                "Confirmatory runs require exactly 120 epochs and effective "
                "batch size 4."
            )
        elif self.method in P6_MAIN_METHODS:
            if self.seed not in P6_MAIN_SEEDS:
                raise ValueError("Confirmatory main seed must be 0--4.")
        elif self.method == "scratch_causal":
            if self.label_budget != 32 or self.seed not in P6_MAIN_SEEDS:
                raise ValueError(
                    "Confirmatory scratch is restricted to budget 32, "
                    "seeds 0--4."
                )
        elif self.method == "ordinary_causal_transfer" and (
            self.label_budget != 32 or self.seed not in (0, 1, 2)
        ):
            raise ValueError(
                "Confirmatory ordinary transfer is restricted to budget 32, "
                "seeds 0--2."
            )
        elif self.method in P6_TRAINED_ABLATION_IDS and (
            self.label_budget != 32 or self.seed not in P6_ABLATION_SEEDS
        ):
            raise ValueError(
                "Confirmatory A0--A6 ablations are restricted to budget 32, "
                "seeds 0--2."
            )
        canonical_run_id = (
            f"p6-{self.method}-budget{self.label_budget}-seed{self.seed}"
        )
        if self.protocol_role == "development":
            canonical_run_id = p6_development_run_id(
                self.method, self.physics_weight
            )
        if self.run_id is not None and self.run_id != canonical_run_id:
            raise ValueError(
                "run_id must equal the canonical pre-registered P6 run ID."
            )
        if self.epochs < 1 or self.effective_batch_size < 1:
            raise ValueError("epochs and effective_batch_size must be positive.")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.candidate_micro_batch_sizes
        ):
            raise ValueError(
                "Microbatch candidates must be integer values."
            )
        candidates = tuple(self.candidate_micro_batch_sizes)
        if (
            not candidates
            or tuple(sorted(set(candidates), reverse=True)) != candidates
            or any(v < 1 or self.effective_batch_size % v for v in candidates)
        ):
            raise ValueError(
                "Microbatch candidates must be unique descending divisors."
            )
        if (
            isinstance(self.physics_weight, bool)
            or not isinstance(self.physics_weight, (int, float))
            or
            not np.isfinite(self.physics_weight)
            or self.physics_weight < 0.0
        ):
            raise ValueError("physics_weight must be finite and nonnegative.")
        positive = {
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "temperature_weight": self.temperature_weight,
            "alpha_weight": self.alpha_weight,
            "restriction_rvs_max": self.restriction_rvs_max,
            "restriction_lis_max": self.restriction_lis_max,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
            for value in positive.values()
        ):
            raise ValueError("Required optimizer/loss values must be positive.")
        nonnegative = (
            self.weight_decay,
            self.gradient_x_weight,
            self.gradient_z_weight,
            self.energy_component_weight,
            self.kinetics_component_weight,
            self.initial_condition_component_weight,
            self.source_temperature_restriction_weight,
            self.source_alpha_restriction_weight,
            self.source_lateral_invariance_weight,
            self.source_restriction_multiplier,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
            for value in nonnegative
        ):
            raise ValueError("Loss weights must be finite and nonnegative.")
        if self.expected_parameter_count != 215_698:
            raise ValueError("P6 parameter count must remain 215,698.")
        fixed_scalars = {
            "learning_rate": (self.learning_rate, 1.0e-3),
            "weight_decay": (self.weight_decay, 1.0e-4),
            "gradient_clip": (self.gradient_clip, 1.0),
            "temperature_weight": (self.temperature_weight, 1.0),
            "alpha_weight": (self.alpha_weight, 0.5),
            "gradient_x_weight": (self.gradient_x_weight, 0.05),
            "gradient_z_weight": (self.gradient_z_weight, 0.05),
            "energy_component_weight": (
                self.energy_component_weight,
                1.0,
            ),
            "kinetics_component_weight": (
                self.kinetics_component_weight,
                0.25,
            ),
            "initial_condition_component_weight": (
                self.initial_condition_component_weight,
                0.25,
            ),
            "source_temperature_restriction_weight": (
                self.source_temperature_restriction_weight,
                1.0,
            ),
            "source_alpha_restriction_weight": (
                self.source_alpha_restriction_weight,
                1.0,
            ),
            "source_lateral_invariance_weight": (
                self.source_lateral_invariance_weight,
                1.0,
            ),
            "source_restriction_multiplier": (
                self.source_restriction_multiplier,
                0.1,
            ),
            "restriction_rvs_max": (self.restriction_rvs_max, 0.25),
            "restriction_lis_max": (self.restriction_lis_max, 1.0e-6),
        }
        changed = {
            name: {"expected": expected, "actual": actual}
            for name, (actual, expected) in fixed_scalars.items()
            if actual != expected
        }
        if changed or self.restriction_validation_case_count != 8:
            raise ValueError(
                "Frozen P6 optimizer/loss/restriction settings changed: "
                f"{changed}"
            )
        for name in (
            "expected_target_split_sha256",
            "expected_source_checkpoint_sha256",
            "expected_inflated_checkpoint_sha256",
            "expected_target_model_config_sha256",
            "expected_source_data_sha256",
            "expected_source_split_sha256",
            "expected_p4_plan_sha256",
            "expected_material_config_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in value
            ):
                raise ValueError(f"{name} must be a SHA-256 digest.")
        if self.restriction_validation_case_count < 1:
            raise ValueError("restriction_validation_case_count must be positive.")
        if (
            self.micro_batch_size is None
            or self.gradient_accumulation_steps is None
        ):
            if require_resolved_resources:
                raise ValueError("Training resources have not been resolved.")
        elif (
            self.micro_batch_size * self.gradient_accumulation_steps
            != self.effective_batch_size
        ):
            raise ValueError("Resolved resources do not form effective batch 4.")
        elif self.micro_batch_size not in candidates:
            raise ValueError(
                "Resolved micro_batch_size was not a preflight candidate."
            )
        return replace(self, candidate_micro_batch_sizes=candidates)


def _config_payload(config: P6CausalTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for name in (
        "project_root",
        "config_file",
        "target_split_manifest",
        "source_checkpoint",
        "inflated_checkpoint",
        "target_model_config",
        "source_data_path",
        "source_split_manifest",
        "p4_plan_path",
        "material_config_path",
        "output_root",
        "resource_profile_path",
    ):
        payload[name] = getattr(config, name).resolve().as_posix()
    payload["candidate_micro_batch_sizes"] = list(
        config.candidate_micro_batch_sizes
    )
    return payload


def _scientific_config(config: P6CausalTrainConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    payload.pop("resume", None)
    return payload


def _preparation_launch_identity(
    *,
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    package_snapshot: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "run_id": make_p6_run_id(config),
        "scientific_config": _scientific_config(config),
        "input_files": dict(checksums),
        "git_sha": implementation["head_sha"],
        "implementation_sha256": implementation["implementation_sha256"],
        "package_snapshot_sha256": package_snapshot["snapshot_sha256"],
        "runtime_fingerprint": dict(runtime),
        "resource_profile": resource_profile,
    }


def _write_or_validate_preparation_launch_identity(
    path: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if not path.is_file():
            raise ResumeAuthorizationError(
                "Resume lacks its immutable preparation launch identity."
            )
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(identity):
            raise ResumeAuthorizationError(
                "Resume preparation launch identity differs."
            )
        return
    if path.exists():
        raise FileExistsError(f"Launch identity already exists: {path}")
    _atomic_text(path, _canonical_json(dict(identity)))


def _checked_file(
    path: Path, expected: str, *, label: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    digest = _sha256_file(path)
    if digest.lower() != expected.lower():
        raise ValueError(f"{label} SHA-256 differs from the P6 lock.")
    return {
        "path": path.resolve().as_posix(),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _validate_material_config(path: Path) -> dict[str, Any]:
    """Cross-check every material scalar used by P6 against readable YAML."""

    try:
        payload = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(
            f"Cannot load AS4/8552 material YAML at {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("AS4/8552 material YAML must be a mapping.")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "material_system",
            "temperature_unit",
            "time_unit",
            "kinetics",
            "public_case1_constituents",
            "p4_anisotropic_conductivity",
            "public_case1_geometry_and_boundaries",
            "public_case1_validation_acceptance",
        },
        label="AS4/8552 material configuration",
    )
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("material_system") != "AS4/8552"
        or payload.get("temperature_unit") != "K"
        or payload.get("time_unit") != "s"
    ):
        raise ValueError("AS4/8552 material identity or units changed.")
    kinetics = payload.get("kinetics")
    constituents = payload.get("public_case1_constituents")
    anisotropic = payload.get("p4_anisotropic_conductivity")
    geometry = payload.get("public_case1_geometry_and_boundaries")
    if not all(
        isinstance(section, dict)
        for section in (kinetics, constituents, anisotropic, geometry)
    ):
        raise ValueError("AS4/8552 executable sections must be mappings.")
    fibre = constituents.get("fibre")
    resin = constituents.get("resin")
    tool = constituents.get("invar_tool")
    longitudinal = anisotropic.get("longitudinal_composite_model")
    transverse = anisotropic.get("through_thickness_composite_model")
    if not all(
        isinstance(section, dict)
        for section in (fibre, resin, tool, longitudinal, transverse)
    ):
        raise ValueError("AS4/8552 material subsections must be mappings.")
    cure = CureKinetics()
    material = PublicCase1Material()
    if (
        kinetics.get("model_id")
        != "hubert_johnston_modified_autocatalytic"
        or kinetics.get("equation")
        != (
            "dalpha_dt = A*exp(-delta_E/(R*T))*alpha^M*(1-alpha)^N/"
            "(1+exp(C*(alpha-C_T*T-C_0)))"
        )
    ):
        raise ValueError("AS4/8552 cure-law declaration changed.")
    if cure.denominator_offset != 1.0:
        raise ValueError(
            "Executable AS4/8552 cure law lost the leading denominator 1."
        )
    values = {
        "kinetics.A_per_s": (kinetics.get("A_per_s"), cure.A_per_s),
        "kinetics.delta_E_J_per_mol": (
            kinetics.get("delta_E_J_per_mol"),
            cure.delta_E_J_per_mol,
        ),
        "kinetics.M": (kinetics.get("M"), cure.M),
        "kinetics.N": (kinetics.get("N"), cure.N),
        "kinetics.C": (kinetics.get("C"), cure.C),
        "kinetics.C_0": (kinetics.get("C_0"), cure.C_0),
        "kinetics.C_T_per_K": (
            kinetics.get("C_T_per_K"),
            cure.C_T_per_K,
        ),
        "kinetics.R_J_per_mol_K": (
            kinetics.get("R_J_per_mol_K"),
            cure.R_J_per_mol_K,
        ),
        "kinetics.initial_alpha": (
            kinetics.get("initial_alpha"),
            material.initial_alpha,
        ),
        "kinetics.heat_of_reaction_J_per_kg_resin": (
            kinetics.get("heat_of_reaction_J_per_kg_resin"),
            material.heat_of_reaction_J_kg_resin,
        ),
        "fibre.volume_fraction": (
            fibre.get("volume_fraction"),
            material.fibre_volume_fraction,
        ),
        "fibre.density_kg_per_m3": (
            fibre.get("density_kg_per_m3"),
            material.fibre_density_kg_m3,
        ),
        "fibre.specific_heat_J_per_kg_K": (
            fibre.get("specific_heat_J_per_kg_K"),
            material.fibre_cp_J_kg_K,
        ),
        "fibre.transverse_conductivity_W_per_m_K": (
            fibre.get("transverse_conductivity_W_per_m_K"),
            material.fibre_transverse_k_W_m_K,
        ),
        "resin.volume_fraction": (
            resin.get("volume_fraction"),
            material.resin_volume_fraction,
        ),
        "resin.density_kg_per_m3": (
            resin.get("density_kg_per_m3"),
            material.resin_density_kg_m3,
        ),
        "resin.specific_heat_J_per_kg_K": (
            resin.get("specific_heat_J_per_kg_K"),
            material.resin_cp_J_kg_K,
        ),
        "resin.conductivity_W_per_m_K": (
            resin.get("conductivity_W_per_m_K"),
            material.resin_k_W_m_K,
        ),
        "tool.density_kg_per_m3": (
            tool.get("density_kg_per_m3"),
            material.tool_density_kg_m3,
        ),
        "tool.specific_heat_J_per_kg_K": (
            tool.get("specific_heat_J_per_kg_K"),
            material.tool_cp_J_kg_K,
        ),
        "tool.conductivity_W_per_m_K": (
            tool.get("conductivity_W_per_m_K"),
            material.tool_k_W_m_K,
        ),
        "anisotropic.reference_temperature_K": (
            anisotropic.get("reference_temperature_K"),
            material.conductivity_reference_temperature_K,
        ),
        "longitudinal.intercept_W_per_m_K": (
            longitudinal.get("intercept_W_per_m_K"),
            material.longitudinal_composite_k_intercept_W_m_K,
        ),
        "longitudinal.slope_W_per_m_K_per_C": (
            longitudinal.get("slope_W_per_m_K_per_C"),
            material.longitudinal_composite_k_slope_W_m_K_per_C,
        ),
        "longitudinal.reference_value_W_per_m_K": (
            longitudinal.get("reference_value_W_per_m_K"),
            material.composite_longitudinal_k_W_m_K,
        ),
        "transverse.reference_value_W_per_m_K": (
            transverse.get("reference_value_W_per_m_K"),
            material.composite_k_W_m_K,
        ),
        "geometry.tool_thickness_m": (
            geometry.get("tool_thickness_m"),
            material.tool_thickness_m,
        ),
        "geometry.composite_thickness_m": (
            geometry.get("composite_thickness_m"),
            material.composite_thickness_m,
        ),
        "geometry.lower_tool_convection_W_per_m2_K": (
            geometry.get("lower_tool_convection_W_per_m2_K"),
            material.lower_h_W_m2_K,
        ),
        "geometry.upper_composite_convection_W_per_m2_K": (
            geometry.get("upper_composite_convection_W_per_m2_K"),
            material.upper_h_W_m2_K,
        ),
        "geometry.initial_temperature_K": (
            geometry.get("initial_temperature_K"),
            material.initial_temperature_K,
        ),
    }
    mismatches: dict[str, Any] = {}
    for name, (declared, executed) in values.items():
        parsed = declared
        if isinstance(declared, str) and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            declared,
        ):
            parsed = float(declared)
        if (
            isinstance(parsed, bool)
            or not isinstance(parsed, (int, float))
            or not math.isfinite(float(parsed))
            or not math.isclose(
                float(parsed),
                float(executed),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            mismatches[name] = {
                "yaml": declared,
                "python": float(executed),
            }
    if mismatches:
        raise ValueError(
            "AS4/8552 YAML differs from executable Python defaults: "
            f"{mismatches}"
        )
    return {
        "passed": True,
        "checked_scalar_count": len(values),
        "python_material": "PublicCase1Material",
        "python_kinetics": "CureKinetics",
        "denominator_offset": cure.denominator_offset,
        "denominator_leading_one_checked": True,
    }


def _input_checksums(config: P6CausalTrainConfig) -> dict[str, Any]:
    material = _checked_file(
        config.material_config_path,
        config.expected_material_config_sha256,
        label="AS4/8552 material config",
    )
    return {
        "experiment_config": {
            "path": config.config_file.resolve().as_posix(),
            "sha256": _sha256_file(config.config_file),
            "bytes": config.config_file.stat().st_size,
        },
        "target_id_manifest": _checked_file(
            config.target_split_manifest,
            config.expected_target_split_sha256,
            label="target ID manifest",
        ),
        "source_checkpoint": _checked_file(
            config.source_checkpoint,
            config.expected_source_checkpoint_sha256,
            label="causal source checkpoint",
        ),
        "inflated_checkpoint": _checked_file(
            config.inflated_checkpoint,
            config.expected_inflated_checkpoint_sha256,
            label="causal inflated checkpoint",
        ),
        "target_model_config": _checked_file(
            config.target_model_config,
            config.expected_target_model_config_sha256,
            label="causal target model config",
        ),
        "source_virtual_data": _checked_file(
            config.source_data_path,
            config.expected_source_data_sha256,
            label="source virtual-input data",
        ),
        "source_virtual_split": _checked_file(
            config.source_split_manifest,
            config.expected_source_split_sha256,
            label="source virtual-input split",
        ),
        "p4_pre_label_plan": _checked_file(
            config.p4_plan_path,
            config.expected_p4_plan_sha256,
            label="P4 pre-label plan",
        ),
        "material_config": material,
        "material_execution_crosscheck": _validate_material_config(
            config.material_config_path
        ),
    }


def build_parameter_groups(
    model: CausalAxisFactorized2DOperator,
    config: P6CausalTrainConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build exhaustive method-declared P6 optimizer groups."""

    ratios = (
        SCRATCH_PARAMETER_GROUP_RATIOS
        if config.method == "scratch_causal"
        else DEFAULT_PARAMETER_GROUP_RATIOS
    )
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in ratios
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("P6 confirmatory paths require T2 trainability.")
        if name.startswith("lift.geometry.") or ".lateral." in name:
            group = "adapter"
        elif name.startswith("lift.") or name.startswith("head."):
            group = "lift_and_heads"
        else:
            group = "shared_core"
        grouped[group].append((name, parameter))
    actual = [name for values in grouped.values() for name, _ in values]
    expected = [name for name, _ in model.named_parameters()]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise AssertionError("P6 parameter groups are not exhaustive/disjoint.")
    optimizer_groups: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for group_name, ratio in ratios.items():
        values = grouped[group_name]
        optimizer_groups.append(
            {
                "params": [parameter for _, parameter in values],
                "lr": config.learning_rate * ratio,
                "weight_decay": config.weight_decay,
                "group_name": group_name,
                "lr_ratio": ratio,
            }
        )
        report[group_name] = {
            "lr_ratio": ratio,
            "initial_learning_rate": config.learning_rate * ratio,
            "parameter_count": sum(p.numel() for _, p in values),
            "parameter_tensor_count": len(values),
            "parameter_names": [name for name, _ in values],
        }
    report["total_parameter_count"] = sum(
        report[name]["parameter_count"] for name in ratios
    )
    report["scratch_flat_lr_exception"] = (
        config.method == "scratch_causal"
    )
    report["ablation_uses_full_transfer_lr_ratios"] = (
        config.method in P6_TRAINED_ABLATION_IDS
    )
    return optimizer_groups, report


def _resolve_repo_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a repository-relative POSIX path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay within the repository.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository.")
    return path


def load_p6_causal_config(
    path: Path,
    *,
    project_root: Path,
    method: str | None = None,
    label_budget: int | None = None,
    seed: int | None = None,
    physics_weight: float | None = None,
    device: str | None = None,
    run_id: str | None = None,
    resume: bool = False,
) -> P6CausalTrainConfig:
    """Load and strictly validate a development or locked-main P6 YAML."""

    root = project_root.resolve()
    config_path = path.resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("P6 configuration must stay inside the repository.")
    try:
        payload = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(
            f"Cannot load strict P6 YAML at {config_path}: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("phase") != "P6"
        or payload.get("experiment") != EXPERIMENT
    ):
        raise ValueError("Unsupported P6 causal experiment configuration.")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "phase",
            "experiment",
            "protocol_role",
            "main_lock_status",
            "comparison",
            "data",
            "model",
            "training",
            "loss",
            "resource_preflight",
            "outputs",
        },
        label="P6 top-level configuration",
    )
    if not isinstance(payload.get("protocol_role"), str):
        raise ValueError("P6 protocol_role must be a string.")
    role = payload["protocol_role"]
    comparison = payload.get("comparison")
    data = payload.get("data")
    model = payload.get("model")
    training = payload.get("training")
    loss = payload.get("loss")
    resource = payload.get("resource_preflight")
    outputs = payload.get("outputs")
    if not all(
        isinstance(value, dict)
        for value in (
            comparison,
            data,
            model,
            training,
            loss,
            resource,
            outputs,
        )
    ):
        raise ValueError("P6 configuration sections must be mappings.")
    _require_exact_keys(
        data,
        {
            "target_split_manifest",
            "expected_target_split_sha256",
            "source_checkpoint",
            "expected_source_checkpoint_sha256",
            "inflated_checkpoint",
            "expected_inflated_checkpoint_sha256",
            "source_virtual_input_data",
            "expected_source_virtual_input_data_sha256",
            "source_virtual_input_split",
            "expected_source_virtual_input_split_sha256",
            "p4_pre_label_plan",
            "expected_p4_pre_label_plan_sha256",
            "material_config",
            "expected_material_config_sha256",
            "verify_array_checksums",
        },
        label="P6 data",
    )
    _require_exact_keys(
        model,
        {
            "config",
            "expected_config_sha256",
            "family",
            "structurally_causal",
            "expected_parameter_count",
            "transfer_stage",
        },
        label="P6 model",
    )
    _require_exact_keys(
        training,
        {
            "device",
            "epochs",
            "early_stopping",
            "effective_batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "num_threads",
            "mixed_precision",
            "checkpoint_selection",
        },
        label="P6 training",
    )
    _require_exact_keys(
        resource,
        {"profile_path", "candidate_micro_batch_sizes"},
        label="P6 resource_preflight",
    )
    _require_exact_keys(outputs, {"root"}, label="P6 outputs")
    if not isinstance(loss.get("data_weights"), dict):
        raise ValueError("P6 loss.data_weights must be a mapping.")
    if not isinstance(loss.get("physics_components"), dict):
        raise ValueError("P6 loss.physics_components must be a mapping.")
    if not isinstance(loss.get("restriction_components"), dict):
        raise ValueError("P6 loss.restriction_components must be a mapping.")
    if not isinstance(loss.get("restriction_validation"), dict):
        raise ValueError("P6 loss.restriction_validation must be a mapping.")
    _require_exact_keys(
        loss["data_weights"],
        {
            "temperature_relative_l2",
            "alpha_relative_l2",
            "gradient_x_relative_l2",
            "gradient_z_relative_l2",
        },
        label="P6 loss.data_weights",
    )
    _require_exact_keys(
        loss["physics_components"],
        {"energy", "kinetics", "initial_condition"},
        label="P6 loss.physics_components",
    )
    _require_exact_keys(
        loss["restriction_components"],
        {
            "source_temperature_rvs",
            "source_alpha_rvs",
            "lateral_invariance",
        },
        label="P6 loss.restriction_components",
    )
    _require_exact_keys(
        loss["restriction_validation"],
        {"case_count", "rvs_max", "lis_max"},
        label="P6 loss.restriction_validation",
    )
    if data.get("verify_array_checksums") is not True:
        raise ValueError("P6 array checksum verification must remain enabled.")
    if not isinstance(training.get("device"), str):
        raise ValueError("P6 training.device must be a string.")
    selected_device = training["device"] if device is None else device
    if not isinstance(selected_device, str) or not selected_device:
        raise ValueError("P6 selected device must be a nonempty string.")

    raw_method = (
        method if method is not None else comparison.get("default_method")
    )
    if not isinstance(raw_method, str):
        raise ValueError("P6 method must be a string.")
    selected_method = raw_method
    selected_budget = _strict_int(
        label_budget
        if label_budget is not None
        else comparison.get("default_budget"),
        label="P6 selected label budget",
    )
    selected_seed = _strict_int(
        seed if seed is not None else comparison.get("default_seed"),
        label="P6 selected seed",
    )
    if role == "development":
        _require_exact_keys(
            comparison,
            {
                "methods",
                "budgets",
                "seeds",
                "default_method",
                "default_budget",
                "default_seed",
                "test_or_ood_labels_used_for_selection",
            },
            label="P6 development comparison",
        )
        _require_exact_keys(
            loss,
            {
                "temperature_region",
                "alpha_region",
                "physics_snapshot_interval_s",
                "source_teacher_uses_2d_labels",
                "physics_weight_candidates",
                "default_physics_weight",
                "data_weights",
                "physics_components",
                "source_restriction_multiplier",
                "restriction_components",
                "restriction_validation",
            },
            label="P6 development loss",
        )
        expected_methods = list(P6_MAIN_METHODS)
        if (
            payload.get("main_lock_status")
            != "pending_validation_only_physics_selection"
            or
            comparison.get("methods") != expected_methods
            or comparison.get("budgets") != [8]
            or comparison.get("seeds") != [P6_DEVELOPMENT_SEED]
            or int(training.get("epochs", -1)) != 40
            or comparison.get("test_or_ood_labels_used_for_selection")
            is not False
        ):
            raise ValueError("P6 development roster differs from protocol.")
        candidates = tuple(
            _strict_float(
                value, label="P6 development physics candidate"
            )
            for value in loss.get("physics_weight_candidates", ())
        )
        if candidates != (0.0, 1.0e-4, 1.0e-3, 1.0e-2):
            raise ValueError("P6 development physics grid changed.")
        selected_physics = _strict_float(
            physics_weight
            if physics_weight is not None
            else loss["default_physics_weight"],
            label="P6 selected physics weight",
        )
        if selected_physics not in candidates:
            raise ValueError(
                "Development physics weight is not pre-registered."
            )
    elif role == "confirmatory":
        _require_exact_keys(
            comparison,
            {
                "main_methods",
                "budgets",
                "main_seeds",
                "secondary_methods",
                "secondary_budget",
                "scratch_seeds",
                "ordinary_transfer_seeds",
                "trained_ablation_methods",
                "ablation_budget",
                "ablation_seeds",
                "ablation_aliases",
                "default_method",
                "default_budget",
                "default_seed",
                "test_or_ood_labels_used_for_selection",
            },
            label="P6 confirmatory comparison",
        )
        _require_exact_keys(
            loss,
            {
                "temperature_region",
                "alpha_region",
                "physics_snapshot_interval_s",
                "source_teacher_uses_2d_labels",
                "selected_physics_weight",
                "data_weights",
                "physics_components",
                "source_restriction_multiplier",
                "restriction_components",
                "restriction_validation",
            },
            label="P6 confirmatory loss",
        )
        expected_comparison = {
            "main_methods": list(P6_MAIN_METHODS),
            "budgets": list(P6_BUDGETS),
            "main_seeds": list(P6_MAIN_SEEDS),
            "secondary_methods": list(P6_SECONDARY_METHODS),
            "secondary_budget": 32,
            "scratch_seeds": list(P6_MAIN_SEEDS),
            "ordinary_transfer_seeds": list(P6_ABLATION_SEEDS),
            "trained_ablation_methods": list(P6_TRAINED_ABLATION_IDS),
            "ablation_budget": 32,
            "ablation_seeds": list(P6_ABLATION_SEEDS),
            "ablation_aliases": {
                "A7": {
                    "reuses_method": "cdcureno_full",
                    "budget": 32,
                    "seeds": list(P6_ABLATION_SEEDS),
                }
            },
            "default_method": "generic_causal_transfer",
            "default_budget": 32,
            "default_seed": 0,
            "test_or_ood_labels_used_for_selection": False,
        }
        if (
            payload.get("main_lock_status") != "frozen_after_development"
            or comparison != expected_comparison
            or int(training.get("epochs", -1)) != 120
        ):
            raise ValueError("P6 confirmatory roster is not frozen.")
        selected_physics = _strict_float(
            loss["selected_physics_weight"],
            label="P6 selected confirmatory physics weight",
        )
        if selected_physics not in (1.0e-4, 1.0e-3, 1.0e-2):
            raise ValueError(
                "Confirmatory physics weight must be a selected positive "
                "pre-registered candidate."
            )
        if (
            physics_weight is not None
            and _strict_float(
                physics_weight,
                label="P6 physics override",
            )
            != selected_physics
        ):
            raise ValueError(
                "Confirmatory physics weight cannot be overridden."
            )
    else:
        raise ValueError("Unknown P6 protocol_role.")
    declared_parameter_count = _strict_int(
        model.get("expected_parameter_count"),
        label="P6 declared parameter count",
    )
    declared_effective_batch = _strict_int(
        training.get("effective_batch_size"),
        label="P6 declared effective batch size",
    )
    declared_epochs = _strict_int(
        training.get("epochs"), label="P6 declared epochs"
    )
    declared_snapshot = _strict_float(
        loss.get("physics_snapshot_interval_s"),
        label="P6 physics snapshot interval",
    )
    if training.get("early_stopping") is not False:
        raise ValueError("P6 early stopping must remain disabled.")
    if training.get("mixed_precision") is not False:
        raise ValueError("P6 mixed precision must remain disabled.")
    if model.get("structurally_causal") is not True:
        raise ValueError("P6 target must remain structurally causal.")
    resource_candidates = resource.get("candidate_micro_batch_sizes")
    if (
        not isinstance(resource_candidates, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in resource_candidates
        )
        or resource_candidates != [2, 1]
    ):
        raise ValueError(
            "P6 resource candidates must remain exactly [2, 1]."
        )
    if (
        model.get("family") != "causal_axis_factorized_2d"
        or declared_parameter_count != 215_698
        or model.get("transfer_stage") != "T2"
        or declared_effective_batch != 4
        or declared_epochs != (40 if role == "development" else 120)
        or training.get("checkpoint_selection")
        != "validation_data_objective_only"
        or loss.get("temperature_region") != "composite_only"
        or loss.get("alpha_region") != "composite_only"
        or declared_snapshot != 120.0
        or loss.get("source_teacher_uses_2d_labels") is not False
    ):
        raise ValueError("P6 model/training/loss declaration changed.")
    config = P6CausalTrainConfig(
        project_root=project_root.resolve(),
        config_file=config_path,
        protocol_role=role,
        target_split_manifest=_resolve_repo_path(
            project_root,
            data["target_split_manifest"],
            label="target_split_manifest",
        ),
        source_checkpoint=_resolve_repo_path(
            project_root,
            data["source_checkpoint"],
            label="source_checkpoint",
        ),
        inflated_checkpoint=_resolve_repo_path(
            project_root,
            data["inflated_checkpoint"],
            label="inflated_checkpoint",
        ),
        target_model_config=_resolve_repo_path(
            project_root,
            model["config"],
            label="target model config",
        ),
        source_data_path=_resolve_repo_path(
            project_root,
            data["source_virtual_input_data"],
            label="source virtual-input data",
        ),
        source_split_manifest=_resolve_repo_path(
            project_root,
            data["source_virtual_input_split"],
            label="source virtual-input split",
        ),
        p4_plan_path=_resolve_repo_path(
            project_root,
            data["p4_pre_label_plan"],
            label="P4 pre-label plan",
        ),
        material_config_path=_resolve_repo_path(
            project_root,
            data["material_config"],
            label="AS4/8552 material config",
        ),
        output_root=_resolve_repo_path(
            project_root, outputs["root"], label="output root"
        ),
        resource_profile_path=_resolve_repo_path(
            project_root, resource["profile_path"], label="resource profile"
        ),
        method=selected_method,  # type: ignore[arg-type]
        label_budget=selected_budget,
        seed=selected_seed,
        physics_weight=selected_physics,
        expected_target_split_sha256=str(
            data["expected_target_split_sha256"]
        ),
        expected_source_checkpoint_sha256=str(
            data["expected_source_checkpoint_sha256"]
        ),
        expected_inflated_checkpoint_sha256=str(
            data["expected_inflated_checkpoint_sha256"]
        ),
        expected_target_model_config_sha256=str(
            model["expected_config_sha256"]
        ),
        expected_source_data_sha256=str(
            data["expected_source_virtual_input_data_sha256"]
        ),
        expected_source_split_sha256=str(
            data["expected_source_virtual_input_split_sha256"]
        ),
        expected_p4_plan_sha256=str(
            data["expected_p4_pre_label_plan_sha256"]
        ),
        expected_material_config_sha256=str(
            data["expected_material_config_sha256"]
        ),
        expected_parameter_count=_strict_int(
            model["expected_parameter_count"],
            label="P6 expected parameter count",
        ),
        epochs=_strict_int(training["epochs"], label="P6 epochs"),
        effective_batch_size=_strict_int(
            training["effective_batch_size"],
            label="P6 effective batch size",
        ),
        candidate_micro_batch_sizes=tuple(
            _strict_int(value, label="P6 microbatch candidate")
            for value in resource["candidate_micro_batch_sizes"]
        ),
        micro_batch_size=(
            1 if role == "development" else None
        ),
        gradient_accumulation_steps=(
            4 if role == "development" else None
        ),
        learning_rate=_strict_float(
            training["learning_rate"], label="P6 learning rate"
        ),
        weight_decay=_strict_float(
            training["weight_decay"], label="P6 weight decay"
        ),
        gradient_clip=_strict_float(
            training["gradient_clip"], label="P6 gradient clip"
        ),
        temperature_weight=_strict_float(
            loss["data_weights"]["temperature_relative_l2"],
            label="P6 temperature weight",
        ),
        alpha_weight=_strict_float(
            loss["data_weights"]["alpha_relative_l2"],
            label="P6 alpha weight",
        ),
        gradient_x_weight=_strict_float(
            loss["data_weights"]["gradient_x_relative_l2"],
            label="P6 gradient-x weight",
        ),
        gradient_z_weight=_strict_float(
            loss["data_weights"]["gradient_z_relative_l2"],
            label="P6 gradient-z weight",
        ),
        energy_component_weight=_strict_float(
            loss["physics_components"]["energy"],
            label="P6 energy weight",
        ),
        kinetics_component_weight=_strict_float(
            loss["physics_components"]["kinetics"],
            label="P6 kinetics weight",
        ),
        initial_condition_component_weight=_strict_float(
            loss["physics_components"]["initial_condition"],
            label="P6 initial-condition weight",
        ),
        source_temperature_restriction_weight=_strict_float(
            loss["restriction_components"]["source_temperature_rvs"],
            label="P6 source-temperature restriction weight",
        ),
        source_alpha_restriction_weight=_strict_float(
            loss["restriction_components"]["source_alpha_rvs"],
            label="P6 source-alpha restriction weight",
        ),
        source_lateral_invariance_weight=_strict_float(
            loss["restriction_components"]["lateral_invariance"],
            label="P6 lateral-invariance weight",
        ),
        source_restriction_multiplier=_strict_float(
            loss["source_restriction_multiplier"],
            label="P6 restriction multiplier",
        ),
        restriction_validation_case_count=_strict_int(
            loss["restriction_validation"]["case_count"],
            label="P6 restriction validation case count",
        ),
        restriction_rvs_max=_strict_float(
            loss["restriction_validation"]["rvs_max"],
            label="P6 restriction RVS maximum",
        ),
        restriction_lis_max=_strict_float(
            loss["restriction_validation"]["lis_max"],
            label="P6 restriction LIS maximum",
        ),
        device=selected_device,
        num_threads=_strict_int(
            training["num_threads"], label="P6 num_threads"
        ),
        verify_array_checksums=data["verify_array_checksums"],
        require_resource_profile=(role == "confirmatory"),
        run_id=run_id,
        resume=resume,
    )
    return config.validated()


def _initialize_p6_run_model(
    config: P6CausalTrainConfig,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    CausalAxisFactorized2DOperator, dict[str, Any], dict[str, Any]
]:
    """Initialize the declared method and apply any state-preserving ablation."""

    initializer = _initializer_method(config.method)
    model, model_spec, initialization = initialize_p6_causal_target(
        method=initializer,
        seed=config.seed,
        target_config_path=config.target_model_config,
        inflated_checkpoint_path=config.inflated_checkpoint,
        expected_target_config_sha256=(
            config.expected_target_model_config_sha256
        ),
        expected_inflated_checkpoint_sha256=(
            config.expected_inflated_checkpoint_sha256
        ),
        device=device,
    )
    ablation = _ablation_spec(config.method)
    if ablation is None:
        return model, model_spec, initialization

    transformation = apply_p6_ablation(model, ablation.ablation_id)
    transformed_state_sha256 = state_dict_sha256(model.state_dict())
    if transformed_state_sha256 != initialization["initial_state_sha256"]:
        raise AssertionError(
            "P6 ablation changed one or more initialized state tensors."
        )
    transformation = {
        **transformation,
        "state_dict_sha256_before": initialization[
            "initial_state_sha256"
        ],
        "state_dict_sha256_after": transformed_state_sha256,
        "state_tensors_unchanged": True,
    }
    runtime_spec = {
        **model_spec,
        "temporal_family": model.temporal_family,
        "structurally_causal": bool(model.structurally_causal),
        "monotone_alpha": bool(ablation.monotone_alpha),
        "interface_channels": bool(ablation.interface_channels),
        "ablation_id": ablation.ablation_id,
        "ablation_contract": transformation["ablation"],
    }
    runtime_initialization = {
        **initialization,
        "method": config.method,
        "initializer_method": initializer,
        "model_spec": runtime_spec,
        "source_output_restriction_loss_enabled": bool(
            ablation.restriction
        ),
        "ablation_transformation": transformation,
    }
    return model, runtime_spec, runtime_initialization


@dataclass(frozen=True)
class FrozenPhysicsInputs:
    """Label-free P4 definitions and grids used by the physics objective."""

    definitions: Mapping[int, Mapping[str, Any]]
    time_s: np.ndarray
    z_m: np.ndarray
    x_m: np.ndarray
    composite_mask: np.ndarray
    array_root: Path


def load_frozen_physics_inputs(
    config: P6CausalTrainConfig,
) -> FrozenPhysicsInputs:
    """Load only pre-label definitions, coordinates, and the material mask."""

    if _sha256_file(config.p4_plan_path) != config.expected_p4_plan_sha256:
        raise ValueError("P4 pre-label plan changed.")
    plan = json.loads(config.p4_plan_path.read_text(encoding="utf-8"))
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 512:
        raise ValueError("P4 plan must contain 512 definitions.")
    definitions: dict[int, Mapping[str, Any]] = {}
    for row in cases:
        if not isinstance(row, dict) or not isinstance(
            row.get("definition"), dict
        ):
            raise ValueError("P4 plan case row is invalid.")
        definition = row["definition"]
        case_id = int(definition["case_id"])
        if case_id in definitions:
            raise ValueError("P4 plan contains a duplicate case ID.")
        definitions[case_id] = definition
    split = json.loads(
        config.target_split_manifest.read_text(encoding="utf-8")
    )
    source_path = _resolve_repo_path(
        config.project_root,
        split["source_manifest"],
        label="P4 source manifest",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    array_root = _resolve_repo_path(
        config.project_root,
        source["array_artifact_root"],
        label="P4 array root",
    )
    values = {
        name: np.load(
            array_root / f"{name}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in ("time_s", "z_m", "x_m", "composite_mask")
    }
    time_s = np.array(values["time_s"], dtype=np.float64, copy=True)
    z_m = np.array(values["z_m"], dtype=np.float64, copy=True)
    x_m = np.array(values["x_m"], dtype=np.float64, copy=True)
    mask = np.array(values["composite_mask"], dtype=np.bool_, copy=True)
    if (
        time_s.shape != (112,)
        or z_m.shape != (50,)
        or x_m.shape != (40,)
        or mask.shape != (50, 40)
        or not np.all(np.diff(time_s) > 0.0)
        or not np.all(np.diff(z_m) > 0.0)
        or not np.all(np.diff(x_m) > 0.0)
    ):
        raise ValueError("Frozen P4 physics grid has changed.")
    return FrozenPhysicsInputs(
        definitions=definitions,
        time_s=time_s,
        z_m=z_m,
        x_m=x_m,
        composite_mask=mask,
        array_root=array_root,
    )


def physics_context_for_cases(
    case_ids: Sequence[int],
    frozen: FrozenPhysicsInputs,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Target2DPhysicsContext:
    """Build differentiable-loss context from label-free case definitions."""

    ids = [int(value) for value in case_ids]
    if not ids or any(case_id not in frozen.definitions for case_id in ids):
        raise ValueError("Physics context references an unknown case ID.")
    material = PublicCase1Material()
    batch = len(ids)
    mask = frozen.composite_mask
    density = np.empty((batch, *mask.shape), dtype=np.float32)
    heat_capacity = np.empty_like(density)
    conductivity_x = np.empty_like(density)
    conductivity_z = np.empty_like(density)
    air = np.empty((batch, len(frozen.time_s)), dtype=np.float32)
    lower = np.empty((batch, len(frozen.x_m)), dtype=np.float32)
    upper = np.empty_like(lower)
    left = np.empty((batch, len(frozen.z_m)), dtype=np.float32)
    right = np.empty_like(left)
    reaction_scale = np.empty(batch, dtype=np.float32)
    for index, case_id in enumerate(ids):
        definition = frozen.definitions[case_id]
        density[index] = np.where(
            mask,
            material.composite_density_kg_m3,
            material.tool_density_kg_m3,
        )
        heat_capacity[index] = np.where(
            mask,
            material.composite_cp_J_kg_K,
            material.tool_cp_J_kg_K,
        )
        conductivity_x[index] = np.where(
            mask,
            float(definition["composite_conductivity_x_W_m_K"]),
            material.tool_k_W_m_K,
        )
        conductivity_z[index] = np.where(
            mask,
            float(definition["composite_conductivity_z_W_m_K"]),
            material.tool_k_W_m_K,
        )
        air[index] = np.asarray(
            definition["air_temperature_K"], dtype=np.float32
        )
        lower[index] = float(definition["bottom_h_W_m2_K"])
        upper[index] = np.asarray(
            definition["top_h_W_m2_K"], dtype=np.float32
        )
        left[index] = float(definition["left_h_W_m2_K"])
        right[index] = float(definition["right_h_W_m2_K"])
        reaction_scale[index] = float(
            definition["reaction_enthalpy_scale"]
        )

    def tensor(value: np.ndarray | float) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    return Target2DPhysicsContext(
        density_kg_m3=tensor(density),
        specific_heat_J_kg_K=tensor(heat_capacity),
        conductivity_x_W_m_K=tensor(conductivity_x),
        conductivity_z_W_m_K=tensor(conductivity_z),
        base_cure_source_J_m3_per_alpha=float(
            material.cure_source_J_m3_per_alpha
        ),
        air_temperature_K=tensor(air),
        lower_h_W_m2_K=tensor(lower),
        upper_h_W_m2_K=tensor(upper),
        left_h_W_m2_K=tensor(left),
        right_h_W_m2_K=tensor(right),
        reaction_enthalpy_scale=tensor(reaction_scale),
        initial_temperature_K=float(material.initial_temperature_K),
    )


def _decode_temperature_torch(
    values: torch.Tensor, normalization: Mapping[str, Any]
) -> torch.Tensor:
    metadata = normalization["field_temperature"]
    minimum = float(metadata["minimum"])
    maximum = float(metadata["maximum"])
    return values * (maximum - minimum) + minimum


def _data_components(
    model: CausalAxisFactorized2DOperator,
    inputs: torch.Tensor,
    temperature: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = model(inputs)
    components = p5.target_loss_components(
        outputs, temperature, alpha, inputs[..., 4]
    )
    return outputs, components


def _data_objective(
    components: Mapping[str, torch.Tensor],
    config: P6CausalTrainConfig,
) -> torch.Tensor:
    return (
        config.temperature_weight * components["temperature"]
        + config.alpha_weight * components["alpha"]
        + config.gradient_x_weight * components["gradient_x"]
        + config.gradient_z_weight * components["gradient_z"]
    )


def _physics_objective(
    outputs: Mapping[str, torch.Tensor],
    case_ids: Sequence[int],
    *,
    prepared: PreparedTarget2DTraining,
    frozen: FrozenPhysicsInputs,
    config: P6CausalTrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    temperature_K = _decode_temperature_torch(
        outputs["temperature"], prepared.normalization
    )
    context = physics_context_for_cases(
        case_ids,
        frozen,
        dtype=temperature_K.dtype,
        device=temperature_K.device,
    )
    residual = target_2d_physics_residuals(
        temperature_K,
        outputs["alpha"],
        frozen.time_s,
        frozen.z_m,
        frozen.x_m,
        frozen.composite_mask,
        context,
    )
    terms = {
        "energy": residual.energy_mean_square,
        "kinetics": residual.cure_kinetics_mean_square,
        "initial_condition": residual.temperature_initial_mean_square,
    }
    total = (
        config.energy_component_weight * terms["energy"]
        + config.kinetics_component_weight * terms["kinetics"]
        + config.initial_condition_component_weight
        * terms["initial_condition"]
    )
    return total, terms


@dataclass(frozen=True)
class VirtualRestrictionPool:
    inputs: torch.Tensor
    teacher: SourceTeacherOutputs
    source: FrozenCausalSource
    metadata: Mapping[str, Any]


def _prepare_training_virtual_pool(
    config: P6CausalTrainConfig,
    *,
    teacher_device: torch.device,
) -> tuple[VirtualRestrictionPool | None, Mapping[str, Any]]:
    if _restriction_enabled(config.method):
        pool = prepare_virtual_restriction_pool(
            config, teacher_device=teacher_device
        )
        return pool, pool.metadata
    return None, {
        "enabled": False,
        "reason": "source_output_restriction_disabled_for_method",
        "method": config.method,
        "source_checkpoint_loaded_for_teacher": False,
        "source_labels_loaded_by_preparation_but_not_used": False,
        "source_teacher_outputs_used": False,
        "target_2d_labels_used": False,
    }


def _training_physics_objective(
    outputs: Mapping[str, torch.Tensor],
    case_ids: Sequence[int],
    *,
    prepared: PreparedTarget2DTraining,
    frozen: FrozenPhysicsInputs,
    config: P6CausalTrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if _physics_enabled(config.method):
        return _physics_objective(
            outputs,
            case_ids,
            prepared=prepared,
            frozen=frozen,
            config=config,
        )
    reference = outputs["temperature"]
    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    return zero, {
        "energy": zero,
        "kinetics": zero,
        "initial_condition": zero,
    }


def prepare_virtual_restriction_pool(
    config: P6CausalTrainConfig,
    *,
    teacher_device: torch.device | str,
) -> VirtualRestrictionPool:
    """Cache frozen source outputs on source-train inputs only."""

    prepared = prepare_source_1d(
        config.source_data_path,
        config.source_split_manifest,
        time_stride=2,
    )
    if tuple(prepared.channel_names) != tuple(INPUT_CHANNELS):
        raise ValueError("Source virtual-input channel order changed.")
    checkpoint = torch.load(
        config.source_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("normalization") != prepared.normalization:
        raise ValueError(
            "Source virtual inputs differ from checkpoint normalization."
        )
    positions = tuple(int(value) for value in prepared.splits["train"])
    if len(positions) != 160 or len(set(positions)) != 160:
        raise ValueError(
            "Source restriction pool must contain 160 unique train cases."
        )
    indices = torch.tensor(positions, dtype=torch.long)
    inputs = prepared.inputs.index_select(0, indices).detach().cpu()
    if inputs.shape != (160, 112, 51, len(INPUT_CHANNELS)):
        raise ValueError(
            "Source restriction inputs must have shape [160,112,51,14]."
        )
    case_ids = tuple(int(prepared.case_ids[index]) for index in positions)
    source = load_frozen_causal_source(
        config.source_checkpoint,
        expected_sha256=config.expected_source_checkpoint_sha256,
        device=teacher_device,
    )
    teacher = precompute_source_teacher(
        source,
        inputs,
        case_ids=case_ids,
        batch_size=8,
        output_device="cpu",
    )
    # Teacher outputs are now immutable CPU tensors; the source network is no
    # longer needed on the accelerator during target optimization.
    source.model.to("cpu")
    return VirtualRestrictionPool(
        inputs=inputs,
        teacher=teacher,
        source=source,
        metadata={
            "split": "source_train_only",
            "position_count": len(positions),
            "case_ids": list(case_ids),
            "source_labels_loaded_by_preparation_but_not_used": True,
            "source_teacher_outputs_used": True,
            "target_2d_labels_used": False,
            "source_checkpoint_sha256": source.checkpoint_sha256,
        },
    )


def _resource_contract(
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "architecture": {
            "family": "causal_axis_factorized_2d",
            "parameter_count": config.expected_parameter_count,
            "full_resolution": [112, 50, 40, 20],
            "full_t2_trainability": True,
        },
        "effective_batch_size": config.effective_batch_size,
        "candidate_micro_batch_sizes": list(
            config.candidate_micro_batch_sizes
        ),
        "mixed_precision": False,
        "worst_case_method": "cdcureno_full",
        "optimizer": "AdamW",
        "parameter_group_lr_ratios": dict(
            DEFAULT_PARAMETER_GROUP_RATIOS
        ),
        "physics_weight": config.physics_weight,
        "physics_components": {
            "energy": config.energy_component_weight,
            "kinetics": config.kinetics_component_weight,
            "initial_condition": (
                config.initial_condition_component_weight
            ),
        },
        "restriction_multiplier": config.source_restriction_multiplier,
        "implementation_sha256": implementation["implementation_sha256"],
        "implementation_source_file_sha256": implementation[
            "source_file_sha256"
        ],
        "input_sha256": {
            name: value["sha256"]
            for name, value in checksums.items()
            if isinstance(value, Mapping) and "sha256" in value
        },
    }


def _validate_resource_profile_semantics(
    profile: Mapping[str, Any],
    config: P6CausalTrainConfig,
    contract: Mapping[str, Any],
) -> tuple[int, int]:
    if profile.get("phase") != "P6":
        raise ValueError("P6 resource profile phase changed.")
    if profile.get("profile_role") != (
        "worst_case_full_loss_full_resolution_cuda"
    ):
        raise ValueError("P6 resource profile role changed.")
    if profile.get("resource_contract") != contract:
        raise ValueError("Embedded P6 resource contract differs.")
    selected = _strict_int(
        profile.get("selected_micro_batch_size"),
        label="P6 selected resource microbatch",
    )
    accumulation = _strict_int(
        profile.get("gradient_accumulation_steps"),
        label="P6 resource accumulation",
    )
    effective = _strict_int(
        profile.get("effective_batch_size"),
        label="P6 resource effective batch",
    )
    if (
        selected not in config.candidate_micro_batch_sizes
        or effective != config.effective_batch_size
        or selected * accumulation != effective
    ):
        raise ValueError("P6 resource batch/accumulation contract changed.")
    attempts = profile.get("attempts")
    selected_index = config.candidate_micro_batch_sizes.index(selected)
    expected_candidates = list(
        config.candidate_micro_batch_sizes[: selected_index + 1]
    )
    if not isinstance(attempts, list) or len(attempts) != len(
        expected_candidates
    ):
        raise ValueError("P6 resource attempts are incomplete.")
    for index, (attempt, candidate) in enumerate(
        zip(attempts, expected_candidates, strict=True)
    ):
        if (
            not isinstance(attempt, dict)
            or _strict_int(
                attempt.get("micro_batch_size"),
                label="P6 attempted microbatch",
            )
            != candidate
        ):
            raise ValueError("P6 resource attempt order changed.")
        if index < selected_index:
            if attempt.get("status") != "cuda_out_of_memory":
                raise ValueError(
                    "A larger P6 resource candidate lacks OOM evidence."
                )
        elif (
            attempt.get("status") != "passed"
            or attempt.get("optimizer_step_completed") is not True
        ):
            raise ValueError(
                "Selected P6 resource candidate lacks a completed step."
            )
    label_access = profile.get("target_label_access")
    if (
        not isinstance(label_access, dict)
        or set(label_access) != {
            "train",
            "validation",
            "id_test",
            "ood",
        }
        or label_access["validation"] != []
        or label_access["id_test"] != []
        or label_access["ood"] != []
        or not isinstance(label_access["train"], list)
        or len(label_access["train"])
        > max(config.candidate_micro_batch_sizes)
    ):
        raise ValueError(
            "P6 resource profile contains invalid target-label access."
        )
    preflight_ids = profile.get("preflight_training_case_ids")
    if (
        not isinstance(preflight_ids, list)
        or len(preflight_ids) != selected
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in preflight_ids
        )
        or not set(preflight_ids).issubset(set(label_access["train"]))
    ):
        raise ValueError("P6 preflight training-case evidence changed.")
    return selected, accumulation


def _resolve_resource_profile(
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    implementation: Mapping[str, Any],
    device: torch.device,
) -> tuple[P6CausalTrainConfig, Mapping[str, Any] | None]:
    if not config.require_resource_profile:
        return config.validated(require_resolved_resources=True), None
    path = config.resource_profile_path
    if not path.is_file():
        raise FileNotFoundError(
            "P6 CUDA resource profile is missing; run --resource-preflight."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        profile.get("schema_version") != RESOURCE_PROFILE_SCHEMA_VERSION
        or profile.get("experiment") != EXPERIMENT
        or profile.get("passed") is not True
    ):
        raise ValueError("P6 resource profile schema/status changed.")
    contract = _resource_contract(config, checksums, implementation)
    if profile.get("resource_contract_sha256") != _sha256_json(contract):
        raise ValueError("P6 resource profile is not bound to this run.")
    if profile.get("device") != _device_payload(device):
        raise ValueError("Active device differs from P6 resource profile.")
    micro, accumulation = _validate_resource_profile_semantics(
        profile, config, contract
    )
    resolved = replace(
        config,
        micro_batch_size=micro,
        gradient_accumulation_steps=accumulation,
    ).validated(require_resolved_resources=True)
    return resolved, profile


def run_p6_resource_preflight(
    config: P6CausalTrainConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze a worst-case full-loss CUDA microbatch contract."""

    config = config.validated()
    if config.protocol_role != "confirmatory":
        raise ValueError("Resource preflight requires the locked main config.")
    device = _resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("Canonical P6 resource preflight requires CUDA.")
    output_path = config.resource_profile_path
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Resource profile exists: {output_path}")
    _configure_determinism(config.seed)
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    torch.cuda.set_device(device)
    checksums = _input_checksums(config)
    implementation = _implementation_state(config.project_root)
    prepared = prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=8,
        project_root=config.project_root,
        verify_array_checksums=config.verify_array_checksums,
    )
    # Exactly two training labels at most are materialized; no validation,
    # ID-test, or OOD labels are touched by preflight.
    samples = [
        prepared.dataset("train")[index]
        for index in range(max(config.candidate_micro_batch_sizes))
    ]
    frozen = load_frozen_physics_inputs(config)
    virtual = prepare_virtual_restriction_pool(
        config, teacher_device=device
    )
    worst_case_config = replace(
        config,
        method="cdcureno_full",
        label_budget=8,
    ).validated()
    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    selected_allocated: int | None = None
    selected_reserved: int | None = None
    for candidate in config.candidate_micro_batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model: CausalAxisFactorized2DOperator | None = None
        optimizer: torch.optim.Optimizer | None = None
        inputs: torch.Tensor | None = None
        temperature: torch.Tensor | None = None
        alpha: torch.Tensor | None = None
        outputs: Mapping[str, torch.Tensor] | None = None
        components: Mapping[str, torch.Tensor] | None = None
        data_loss: torch.Tensor | None = None
        physics: torch.Tensor | None = None
        physics_terms: Mapping[str, torch.Tensor] | None = None
        restriction: Any = None
        objective: torch.Tensor | None = None
        groups: list[dict[str, Any]] | None = None
        unused_spec: Mapping[str, Any] | None = None
        unused_initialization: Mapping[str, Any] | None = None
        try:
            _configure_determinism(config.seed)
            model, unused_spec, unused_initialization = (
                initialize_p6_causal_target(
                    method="cdcureno_full",
                    seed=config.seed,
                    target_config_path=config.target_model_config,
                    inflated_checkpoint_path=config.inflated_checkpoint,
                    expected_target_config_sha256=(
                        config.expected_target_model_config_sha256
                    ),
                    expected_inflated_checkpoint_sha256=(
                        config.expected_inflated_checkpoint_sha256
                    ),
                    device=device,
                )
            )
            groups, _ = build_parameter_groups(model, worst_case_config)
            optimizer = torch.optim.AdamW(groups)
            inputs = torch.stack(
                [sample[0] for sample in samples[:candidate]]
            ).to(device)
            temperature = torch.stack(
                [sample[1] for sample in samples[:candidate]]
            ).to(device)
            alpha = torch.stack(
                [sample[2] for sample in samples[:candidate]]
            ).to(device)
            case_ids = [int(sample[3]) for sample in samples[:candidate]]
            outputs, components = _data_components(
                model, inputs, temperature, alpha
            )
            data_loss = _data_objective(components, config)
            physics, physics_terms = _physics_objective(
                outputs,
                case_ids,
                prepared=prepared,
                frozen=frozen,
                config=config,
            )
            positions = tuple(range(candidate))
            restriction = source_output_restriction_loss(
                model,
                virtual.inputs[list(positions)].to(device),
                virtual.teacher.select(positions),
                nx=40,
                temperature_weight=(
                    config.source_temperature_restriction_weight
                ),
                alpha_weight=config.source_alpha_restriction_weight,
                lateral_invariance_weight=(
                    config.source_lateral_invariance_weight
                ),
            )
            objective = (
                data_loss
                + config.physics_weight * physics
                + config.source_restriction_multiplier * restriction.loss
            )
            if not bool(torch.isfinite(objective.detach()).item()):
                raise FloatingPointError(
                    "P6 resource preflight produced a non-finite objective."
                )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(
                    "P6 resource preflight produced non-finite gradients."
                )
            optimizer.step()
            torch.cuda.synchronize(device)
            selected = candidate
            selected_allocated = int(
                torch.cuda.max_memory_allocated(device)
            )
            selected_reserved = int(
                torch.cuda.max_memory_reserved(device)
            )
            attempts.append(
                {
                    "micro_batch_size": candidate,
                    "status": "passed",
                    "optimizer_step_completed": True,
                    "objective": float(objective.detach()),
                    "gradient_norm_before_clip": gradient_norm,
                    "peak_memory_allocated_bytes": selected_allocated,
                    "peak_memory_reserved_bytes": selected_reserved,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
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
            del (
                model,
                optimizer,
                inputs,
                temperature,
                alpha,
                outputs,
                components,
                data_loss,
                physics,
                physics_terms,
                restriction,
                objective,
                groups,
                unused_spec,
                unused_initialization,
            )
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("No P6 full-loss microbatch fits the GPU.")
    contract = _resource_contract(config, checksums, implementation)
    profile = {
        "schema_version": RESOURCE_PROFILE_SCHEMA_VERSION,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "passed": True,
        "profile_role": "worst_case_full_loss_full_resolution_cuda",
        "resource_contract": contract,
        "resource_contract_sha256": _sha256_json(contract),
        "device": _device_payload(device),
        "selected_micro_batch_size": selected,
        "gradient_accumulation_steps": (
            config.effective_batch_size // selected
        ),
        "effective_batch_size": config.effective_batch_size,
        "selected_peak_memory_allocated_bytes": selected_allocated,
        "selected_peak_memory_reserved_bytes": selected_reserved,
        "attempts": attempts,
        "preflight_training_case_ids": [
            int(sample[3]) for sample in samples[:selected]
        ],
        "target_label_access": {
            "train": list(prepared.accessed_case_ids["train"]),
            "validation": [],
            "id_test": [],
            "ood": [],
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    _atomic_text(output_path, _canonical_json(profile))
    return profile


def make_p6_run_id(config: P6CausalTrainConfig) -> str:
    if config.run_id:
        return config.run_id
    base = (
        f"p6-{config.method}-budget{config.label_budget}-seed{config.seed}"
    )
    if config.protocol_role == "development":
        return p6_development_run_id(config.method, config.physics_weight)
    return base


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in state.items()
    }


def _installed_package_snapshot() -> dict[str, Any]:
    packages: dict[str, dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        display_name = distribution.metadata.get("Name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("Installed distribution lacks a package name.")
        canonical = re.sub(r"[-_.]+", "-", display_name).lower()
        version = str(distribution.version)
        record = {
            "name": canonical,
            "display_name": display_name,
            "version": version,
        }
        if canonical in packages and packages[canonical] != record:
            raise ValueError(
                f"Conflicting installed distributions for {canonical!r}."
            )
        packages[canonical] = record
    rows = [packages[name] for name in sorted(packages)]
    snapshot = {
        "schema_version": 1,
        "format": "installed_python_distributions_name_version",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "package_count": len(rows),
        "packages": rows,
    }
    snapshot["snapshot_sha256"] = _sha256_json(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    return snapshot


def _nvidia_driver_version(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    versions = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    index = 0 if device.index is None else int(device.index)
    if index >= len(versions):
        raise RuntimeError("nvidia-smi omitted the active CUDA device.")
    return versions[index]


def _runtime_fingerprint(
    device: torch.device,
    *,
    package_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "platform_tag": sys.platform,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": _nvidia_driver_version(device),
        "device": _device_payload(device),
        "package_snapshot_sha256": package_snapshot["snapshot_sha256"],
        "package_count": package_snapshot["package_count"],
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _checkpoint_payload(
    *,
    model: CausalAxisFactorized2DOperator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int,
    best_validation: float,
    best_model: Mapping[str, torch.Tensor],
    history: Sequence[Mapping[str, Any]],
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    initialization: Mapping[str, Any],
    parameter_groups: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
    train_generator: torch.Generator,
    virtual_generator: torch.Generator,
    train_sampler: p5.StatefulShuffleSampler,
    virtual_sampler: p5.StatefulShuffleSampler,
) -> dict[str, Any]:
    frozen_best = _clone_state(best_model)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "model_family": "causal_axis_factorized_2d",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation": float(best_validation),
        "best_model": frozen_best,
        "best_model_sha256": state_dict_sha256(frozen_best),
        "history_rows": [dict(row) for row in history],
        "scientific_config": _scientific_config(config),
        "input_checksums": dict(checksums),
        "target_data_checksums": prepared.checksums,
        "normalization": prepared.normalization,
        "channel_names": tuple(TARGET_INPUT_CHANNELS),
        "train_case_ids": tuple(prepared.train_case_ids),
        "validation_case_ids": tuple(prepared.validation_case_ids),
        "selection_split": "validation",
        "selection_objective": "common_data_objective_only",
        "selection_uses_test_or_ood_labels": False,
        "implementation": dict(implementation),
        "runtime_fingerprint": dict(runtime),
        "initialization": dict(initialization),
        "parameter_groups": dict(parameter_groups),
        "resource_profile_sha256": (
            None
            if resource_profile is None
            else _sha256_json(resource_profile)
        ),
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
    }


def _best_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    best = checkpoint["best_model"]
    if state_dict_sha256(best) != checkpoint["best_model_sha256"]:
        raise ValueError("Authoritative best-model hash is invalid.")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": checkpoint["protocol_role"],
        "method": checkpoint["method"],
        "label_budget": checkpoint["label_budget"],
        "seed": checkpoint["seed"],
        "model_family": checkpoint["model_family"],
        "model": best,
        "model_sha256": checkpoint["best_model_sha256"],
        "epoch": checkpoint["best_epoch"],
        "best_validation": checkpoint["best_validation"],
        "scientific_config": checkpoint["scientific_config"],
        "input_checksums": checkpoint["input_checksums"],
        "target_data_checksums": checkpoint["target_data_checksums"],
        "normalization": checkpoint["normalization"],
        "channel_names": checkpoint["channel_names"],
        "selection_split": "validation",
        "selection_uses_test_or_ood_labels": False,
        "git_sha": checkpoint["implementation"]["head_sha"],
        "implementation_sha256": checkpoint["implementation"][
            "implementation_sha256"
        ],
        "runtime_fingerprint": checkpoint["runtime_fingerprint"],
        "initialization": checkpoint["initialization"],
        "parameter_groups": checkpoint["parameter_groups"],
        "model_config": checkpoint["initialization"]["model_spec"],
        "derived_from_last_epoch": checkpoint["epoch"],
    }


def _write_derived_epoch_artifacts(
    checkpoint: Mapping[str, Any],
    *,
    best_path: Path,
    history_path: Path,
) -> None:
    _atomic_torch(_best_payload(checkpoint), best_path)
    frame = pd.DataFrame(
        [dict(row) for row in checkpoint["history_rows"]]
    )
    if "learning_rates" in frame:
        frame["learning_rates"] = frame["learning_rates"].map(
            lambda value: json.dumps(value, sort_keys=True)
            if not isinstance(value, str)
            else value
        )
    _atomic_parquet(frame, history_path)


def _restore_rng(checkpoint: Mapping[str, Any]) -> None:
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if (
        checkpoint.get("cuda_rng_state_all") is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])


def _load_resume_checkpoint(path: Path) -> Any:
    """Load mixed model and CPU-only replay state without device remapping."""

    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_resume_checkpoint(
    checkpoint: Any,
    *,
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
) -> None:
    """Fail closed on incomplete or internally inconsistent epoch state."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("P6 resume checkpoint must be a mapping.")
    expected_metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "model_family": "causal_axis_factorized_2d",
        "scientific_config": _scientific_config(config),
        "input_checksums": checksums,
        "target_data_checksums": prepared.checksums,
        "normalization": prepared.normalization,
        "channel_names": tuple(TARGET_INPUT_CHANNELS),
        "train_case_ids": tuple(prepared.train_case_ids),
        "validation_case_ids": tuple(prepared.validation_case_ids),
        "selection_split": "validation",
        "selection_objective": "common_data_objective_only",
        "selection_uses_test_or_ood_labels": False,
        "runtime_fingerprint": runtime,
        "resource_profile_sha256": (
            None
            if resource_profile is None
            else _sha256_json(resource_profile)
        ),
    }
    mismatches = {
        name: {"expected": expected, "actual": checkpoint.get(name)}
        for name, expected in expected_metadata.items()
        if checkpoint.get(name) != expected
    }
    checkpoint_implementation = checkpoint.get("implementation")
    if (
        not isinstance(checkpoint_implementation, Mapping)
        or checkpoint_implementation.get("implementation_sha256")
        != implementation["implementation_sha256"]
    ):
        mismatches["implementation_sha256"] = {
            "expected": implementation["implementation_sha256"],
            "actual": (
                None
                if not isinstance(checkpoint_implementation, Mapping)
                else checkpoint_implementation.get("implementation_sha256")
            ),
        }
    if mismatches:
        raise ValueError(f"P6 resume checkpoint contract differs: {mismatches}")

    epoch = _strict_int(
        checkpoint.get("epoch"), label="P6 checkpoint epoch"
    )
    best_epoch = _strict_int(
        checkpoint.get("best_epoch"), label="P6 checkpoint best epoch"
    )
    best_validation = _strict_float(
        checkpoint.get("best_validation"),
        label="P6 checkpoint best validation",
    )
    history = checkpoint.get("history_rows")
    if (
        not 1 <= epoch <= config.epochs
        or not 1 <= best_epoch <= epoch
        or best_validation < 0.0
        or not isinstance(history, list)
        or len(history) != epoch
    ):
        raise ValueError("P6 checkpoint epoch/history bounds are invalid.")
    validation_values: list[float] = []
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("P6 checkpoint history row must be a mapping.")
        if _strict_int(
            row.get("epoch"), label="P6 checkpoint history epoch"
        ) != expected_epoch:
            raise ValueError("P6 checkpoint history epochs are not contiguous.")
        validation_values.append(
            _strict_float(
                row.get("validation_data_objective"),
                label="P6 checkpoint validation objective",
            )
        )
    minimum = min(validation_values)
    first_minimum_epoch = validation_values.index(minimum) + 1
    if (
        best_validation != minimum
        or best_epoch != first_minimum_epoch
    ):
        raise ValueError(
            "P6 checkpoint best selection is not the earliest data-only "
            "validation minimum."
        )
    scheduler_state = checkpoint.get("scheduler")
    if (
        not isinstance(scheduler_state, Mapping)
        or scheduler_state.get("last_epoch") != epoch
    ):
        raise ValueError("P6 checkpoint scheduler epoch is inconsistent.")
    train_sampler_state = checkpoint.get("train_sampler_state")
    if (
        not isinstance(train_sampler_state, Mapping)
        or train_sampler_state.get("size") != config.label_budget
        or train_sampler_state.get("position") != config.label_budget
        or train_sampler_state.get("epoch") != epoch
    ):
        raise ValueError(
            "P6 checkpoint train sampler is not at an epoch boundary."
        )
    if not isinstance(checkpoint.get("virtual_sampler_state"), Mapping):
        raise ValueError("P6 checkpoint virtual sampler state is missing.")
    best_model = checkpoint.get("best_model")
    best_hash = checkpoint.get("best_model_sha256")
    if (
        not isinstance(best_model, Mapping)
        or not isinstance(best_hash, str)
        or state_dict_sha256(best_model) != best_hash
    ):
        raise ValueError("P6 resume best-model hash is invalid.")


def _next_virtual(
    loader: DataLoader,
    iterator: Iterator,
) -> tuple[tuple[torch.Tensor, torch.Tensor], Iterator]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def _evaluate_selected_physics(
    model: CausalAxisFactorized2DOperator,
    prepared: PreparedTarget2DTraining,
    frozen: FrozenPhysicsInputs,
    config: P6CausalTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        prepared.dataset("validation"),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    totals = {
        "energy": 0.0,
        "kinetics": 0.0,
        "initial_condition": 0.0,
        "combined": 0.0,
    }
    truth_totals = {
        "energy": 0.0,
        "kinetics": 0.0,
        "initial_condition": 0.0,
        "combined": 0.0,
    }
    count = 0
    alpha_bound_violations = 0
    alpha_monotonicity_violations = 0
    finite = True
    model.eval()
    with torch.no_grad():
        for inputs, temperature_truth, alpha_truth, case_ids in loader:
            inputs = inputs.to(device)
            temperature_truth = temperature_truth.to(device)
            alpha_truth = alpha_truth.to(device)
            outputs = model(inputs)
            physics, terms = _physics_objective(
                outputs,
                [int(value) for value in case_ids],
                prepared=prepared,
                frozen=frozen,
                config=config,
            )
            truth_physics, truth_terms = _physics_objective(
                {
                    "temperature": temperature_truth,
                    "alpha": alpha_truth,
                },
                [int(value) for value in case_ids],
                prepared=prepared,
                frozen=frozen,
                config=config,
            )
            batch = len(inputs)
            for name, value in terms.items():
                totals[name] += float(value) * batch
            totals["combined"] += float(physics) * batch
            for name, value in truth_terms.items():
                truth_totals[name] += float(value) * batch
            truth_totals["combined"] += float(truth_physics) * batch
            alpha = outputs["alpha"]
            finite &= bool(
                torch.all(torch.isfinite(outputs["temperature"])).item()
                and torch.all(torch.isfinite(alpha)).item()
            )
            alpha_bound_violations += int(
                torch.count_nonzero((alpha < -1.0e-7) | (alpha > 1.0 + 1.0e-7))
            )
            alpha_monotonicity_violations += int(
                torch.count_nonzero(torch.diff(alpha, dim=1) < -1.0e-7)
            )
            count += batch
    if count != 32:
        raise ValueError("Selected physics audit did not cover validation.")
    return {
        "case_count": count,
        **{f"{name}_mean": value / count for name, value in totals.items()},
        "prediction_finite": finite,
        "alpha_bound_violation_count": alpha_bound_violations,
        "alpha_monotonicity_violation_count": (
            alpha_monotonicity_violations
        ),
        "truth_snapshot_discretization_floor": {
            "case_count": count,
            **{
                f"{name}_mean": value / count
                for name, value in truth_totals.items()
            },
            "uses_validation_truth": True,
            "used_for_checkpoint_selection": False,
            "test_or_ood_labels_used": False,
        },
        "test_or_ood_labels_used": False,
    }


def _causality_audit(
    model: CausalAxisFactorized2DOperator,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    expected_causal: bool = True,
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    if inputs.ndim != 4 or inputs.shape[-1] != 20:
        raise ValueError("Causality audit input must be [T,Z,X,20].")
    temporal_modules = [block.temporal for block in model.blocks]
    expected_type = (
        CausalTemporalConv1d
        if expected_causal
        else MatchedTwoSidedTemporalConv1d
    )
    structure_checks = {
        "model_structurally_causal_flag": (
            bool(model.structurally_causal) is expected_causal
        ),
        "all_temporal_blocks_match_expected_type": all(
            isinstance(module, expected_type)
            for module in temporal_modules
        ),
        "temporal_block_count_matches_depth": (
            len(temporal_modules) == model.depth
        ),
    }
    structure_passed = bool(all(structure_checks.values()))
    if not expected_causal:
        return {
            "expectation": "expected_noncausal",
            "structurally_causal": False,
            "future_invariance_required": False,
            "future_invariance_evaluated": False,
            "structure_checks": structure_checks,
            "passed": structure_passed,
        }

    original = inputs.unsqueeze(0).to(device)
    split = int(original.shape[1] // 2)
    perturbed = original.clone()
    generator = torch.Generator(device="cpu").manual_seed(20260725)
    noise = torch.randn(
        perturbed[:, split:].shape,
        generator=generator,
        dtype=perturbed.dtype,
    ).to(device)
    perturbed[:, split:] += 0.1 * noise
    model.eval()
    with torch.no_grad():
        baseline = model(original)
        changed = model(perturbed)
    fields: dict[str, Any] = {}
    for name in ("temperature", "alpha"):
        prefix_difference = torch.max(
            torch.abs(
                baseline[name][:, :split] - changed[name][:, :split]
            )
        )
        maximum = float(prefix_difference)
        fields[name] = {
            "prefix_maximum_absolute_difference": maximum,
            "passed": maximum <= tolerance,
        }
    return {
        "expectation": "expected_causal",
        "structurally_causal": True,
        "future_invariance_required": True,
        "future_invariance_evaluated": True,
        "perturbation_start_index": split,
        "tolerance": tolerance,
        "fields": fields,
        "structure_checks": structure_checks,
        "passed": bool(
            structure_passed
            and all(value["passed"] for value in fields.values())
        ),
    }


def _write_run_provenance(
    run_dir: Path,
    *,
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    package_snapshot: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
    virtual_metadata: Mapping[str, Any],
) -> None:
    _atomic_text(
        run_dir / "config_resolved.json",
        _canonical_json(
            {
                "schema_version": 1,
                "phase": "P6",
                "experiment": EXPERIMENT,
                "scientific_config": _scientific_config(config),
                "selection_split": "validation",
                "selection_objective": "common_data_objective_only",
                "target_test_or_ood_labels_used": False,
            }
        ),
    )
    _atomic_text(
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
    package_path = run_dir / "package_snapshot.json"
    _atomic_text(package_path, _canonical_json(dict(package_snapshot)))
    _atomic_text(
        run_dir / "environment.json",
        _canonical_json(
            {
                "schema_version": 1,
                "runtime": dict(runtime),
                "package_snapshot": _artifact(
                    package_path, config.project_root
                ),
                "command": sys.argv,
                "resource_profile": resource_profile,
            }
        ),
    )
    _atomic_text(
        run_dir / "restriction_virtual_input.json",
        _canonical_json(dict(virtual_metadata)),
    )
    _atomic_text(
        run_dir / "git_state.txt",
        (
            f"HEAD {implementation['head_sha']}\n"
            f"implementation_sha256 "
            f"{implementation['implementation_sha256']}\n\n"
            f"status --short\n{implementation['status_short']}\n"
        ),
    )


def _validate_existing_run_provenance(
    run_dir: Path,
    *,
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    package_snapshot: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
    virtual_metadata: Mapping[str, Any],
) -> None:
    """Fail closed before mutating the audit state of any resumed run."""

    required = (
        "config_resolved.json",
        "data_checksums.json",
        "environment.json",
        "package_snapshot.json",
        "restriction_virtual_input.json",
        "git_state.txt",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ResumeAuthorizationError(
            "Resume lacks immutable pre-epoch provenance: "
            + ", ".join(missing)
        )
    frozen_config = json.loads(
        (run_dir / "config_resolved.json").read_text(encoding="utf-8")
    )
    frozen_data = json.loads(
        (run_dir / "data_checksums.json").read_text(encoding="utf-8")
    )
    frozen_environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    frozen_package = json.loads(
        (run_dir / "package_snapshot.json").read_text(encoding="utf-8")
    )
    frozen_virtual = json.loads(
        (run_dir / "restriction_virtual_input.json").read_text(
            encoding="utf-8"
        )
    )
    expected_config = {
        "schema_version": 1,
        "phase": "P6",
        "experiment": EXPERIMENT,
        "scientific_config": _scientific_config(config),
        "selection_split": "validation",
        "selection_objective": "common_data_objective_only",
        "target_test_or_ood_labels_used": False,
    }
    expected_data = {
        "input_files": dict(checksums),
        "prepared_target_contract": prepared.checksums,
        "train_case_ids": list(prepared.train_case_ids),
        "validation_case_ids": list(prepared.validation_case_ids),
    }
    package_path = run_dir / "package_snapshot.json"
    expected_environment = {
        "schema_version": 1,
        "runtime": dict(runtime),
        "package_snapshot": _artifact(
            package_path, config.project_root
        ),
        "command": frozen_environment.get("command"),
        "resource_profile": resource_profile,
    }
    if (
        frozen_config != expected_config
        or frozen_data != expected_data
        or frozen_package != dict(package_snapshot)
        or frozen_virtual != dict(virtual_metadata)
        or frozen_environment != expected_environment
    ):
        raise ValueError(
            "Resume configuration, data, runtime, package, resource, or "
            "virtual-input provenance differs."
        )
    git_lines = (run_dir / "git_state.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if (
        not git_lines
        or git_lines[0] != f"HEAD {implementation['head_sha']}"
        or len(git_lines) < 2
        or git_lines[1]
        != (
            "implementation_sha256 "
            f"{implementation['implementation_sha256']}"
        )
    ):
        raise ValueError(
            "Resume Git or implementation identity differs."
        )


def _ensure_restart_before_epoch_provenance(
    run_dir: Path,
    *,
    config: P6CausalTrainConfig,
    checksums: Mapping[str, Any],
    prepared: PreparedTarget2DTraining,
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    package_snapshot: Mapping[str, Any],
    resource_profile: Mapping[str, Any] | None,
    virtual_metadata: Mapping[str, Any],
) -> None:
    """Complete only missing provenance; never overwrite prior bytes."""

    package_path = run_dir / "package_snapshot.json"
    expected_json: dict[Path, Mapping[str, Any]] = {
        run_dir / "config_resolved.json": {
            "schema_version": 1,
            "phase": "P6",
            "experiment": EXPERIMENT,
            "scientific_config": _scientific_config(config),
            "selection_split": "validation",
            "selection_objective": "common_data_objective_only",
            "target_test_or_ood_labels_used": False,
        },
        run_dir / "data_checksums.json": {
            "input_files": dict(checksums),
            "prepared_target_contract": prepared.checksums,
            "train_case_ids": list(prepared.train_case_ids),
            "validation_case_ids": list(prepared.validation_case_ids),
        },
        package_path: dict(package_snapshot),
        run_dir / "restriction_virtual_input.json": dict(virtual_metadata),
    }
    for path, expected in expected_json.items():
        if path.is_file():
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != dict(expected):
                raise ValueError(
                    f"Restart provenance differs: {path.name}."
                )
        else:
            _atomic_text(path, _canonical_json(dict(expected)))
    package_artifact = _artifact(package_path, config.project_root)
    environment_path = run_dir / "environment.json"
    if environment_path.is_file():
        environment = json.loads(
            environment_path.read_text(encoding="utf-8")
        )
        if (
            environment.get("schema_version") != 1
            or environment.get("runtime") != dict(runtime)
            or environment.get("package_snapshot") != package_artifact
            or environment.get("resource_profile") != resource_profile
            or not isinstance(environment.get("command"), list)
        ):
            raise ValueError(
                "Restart environment provenance differs."
            )
    else:
        _atomic_text(
            environment_path,
            _canonical_json(
                {
                    "schema_version": 1,
                    "runtime": dict(runtime),
                    "package_snapshot": package_artifact,
                    "command": sys.argv,
                    "resource_profile": resource_profile,
                }
            ),
        )
    git_path = run_dir / "git_state.txt"
    if git_path.is_file():
        lines = git_path.read_text(encoding="utf-8").splitlines()
        if (
            not lines
            or lines[0] != f"HEAD {implementation['head_sha']}"
            or len(lines) < 2
            or lines[1]
            != (
                "implementation_sha256 "
                f"{implementation['implementation_sha256']}"
            )
        ):
            raise ValueError(
                "Restart Git or implementation provenance differs."
            )
    else:
        _atomic_text(
            git_path,
            (
                f"HEAD {implementation['head_sha']}\n"
                f"implementation_sha256 "
                f"{implementation['implementation_sha256']}\n\n"
                f"status --short\n{implementation['status_short']}\n"
            ),
        )


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()


def _process_is_live(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    if _IS_WINDOWS:
        return windows_process_is_live(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _is_infrastructure_interruption(error: BaseException) -> bool:
    if isinstance(error, KeyboardInterrupt):
        return True
    if not isinstance(error, OSError):
        return False
    return error.errno in {
        errno.EIO,
        errno.ENOSPC,
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
    }


def _validate_managed_launch_authorization(
    config: P6CausalTrainConfig,
    *,
    run_id: str,
) -> None:
    """Require an exact active launcher record before any training write."""

    state_path = config.project_root.resolve() / "RUN_STATE.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeAuthorizationError(
            "P6 training requires a readable managed RUN_STATE.json; "
            "launch through scripts/run_p6_roster.py."
        ) from error
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ResumeAuthorizationError(
            "P6 managed RUN_STATE.json has an invalid schema."
        )
    if state.get("current_phase") != "P6":
        raise ResumeAuthorizationError(
            "P6 managed launch state is not in phase P6."
        )
    if state.get("active_run_ids") != [run_id]:
        raise ResumeAuthorizationError(
            f"P6 run {run_id} is not the sole managed active run."
        )
    records = state.get("p6_run_records")
    record = records.get(run_id) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        raise ResumeAuthorizationError(
            f"P6 run {run_id} lacks its managed launch record."
        )
    expected = {
        "status": "running",
        "protocol_role": config.protocol_role,
        "run_id": run_id,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "physics_weight": config.physics_weight,
    }
    drift = {
        name: {"expected": value, "actual": record.get(name)}
        for name, value in expected.items()
        if record.get(name) != value
    }
    if drift:
        raise ResumeAuthorizationError(
            f"P6 managed launch record differs from the request: {drift}"
        )
    identity_names = (
        "roster_sha256",
        "roster_file_sha256",
        "config_sha256",
        "git_sha",
        "implementation_sha256",
        "package_snapshot_sha256",
        "runtime_sha256",
    )
    if any(
        not isinstance(record.get(name), str) or not record[name]
        for name in identity_names
    ):
        raise ResumeAuthorizationError(
            "P6 managed launch record lacks immutable identity hashes."
        )
    if (
        not isinstance(record.get("attempt_id"), str)
        or not record["attempt_id"]
        or not isinstance(record.get("last_argv"), list)
    ):
        raise ResumeAuthorizationError(
            "P6 managed launch record lacks its attempt identity."
        )
    runtime_locks = state.get("p6_runtime_locks")
    role_lock = (
        runtime_locks.get(config.protocol_role)
        if isinstance(runtime_locks, dict)
        else None
    )
    expected_lock = {
        name: record[name]
        for name in (
            "git_sha",
            "implementation_sha256",
            "package_snapshot_sha256",
            "runtime_sha256",
        )
    }
    if role_lock != expected_lock:
        raise ResumeAuthorizationError(
            "P6 managed runtime lock differs from the active run."
        )
    for name in (
        "completed_run_ids",
        "failed_run_ids",
        "paused_run_ids",
        "interrupted_run_ids",
    ):
        values = state.get(name)
        if not isinstance(values, list) or run_id in values:
            raise ResumeAuthorizationError(
                f"P6 managed state list {name} conflicts with the active run."
            )


def _repair_pending_terminal_commit(
    run_dir: Path,
    *,
    config: P6CausalTrainConfig,
) -> dict[str, Any]:
    """Finish STATUS/DONE commit steps after a verified hard interruption."""

    if (run_dir / "DONE").exists() or (run_dir / "FAILED").exists():
        raise ResumeAuthorizationError(
            "Terminal repair requires no DONE or FAILED marker."
        )
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (run_dir / "STATUS.json").read_text(encoding="utf-8")
    )
    if not isinstance(metrics, dict) or not isinstance(status, dict):
        raise ResumeAuthorizationError(
            "Pending terminal metrics/STATUS must be mappings."
        )
    completed = metrics.get("completed_at")
    identity = {
        "schema_version": 1,
        "status": "completed",
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "run_id": make_p6_run_id(config),
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "physics_weight": config.physics_weight,
    }
    if (
        any(metrics.get(name) != value for name, value in identity.items())
        or not isinstance(completed, str)
    ):
        raise ResumeAuthorizationError(
            "Pending terminal metrics identity differs."
        )
    completed_status = status == {
        "status": "completed",
        "timestamp": completed,
    }
    interrupted_running_status = (
        status.get("status") == "running"
        and isinstance(status.get("timestamp"), str)
        and not _process_is_live(status.get("pid"))
    )
    if not completed_status and not interrupted_running_status:
        raise ResumeAuthorizationError(
            "Pending terminal STATUS is neither the exact completed marker "
            "nor a dead running process."
        )
    training = metrics.get("training")
    if (
        not isinstance(training, dict)
        or training.get("configured_epochs") != config.epochs
        or training.get("executed_epochs") != config.epochs
    ):
        raise ResumeAuthorizationError(
            "Pending terminal metrics do not record the complete epoch plan."
        )
    sessions = [
        json.loads(line)
        for line in (run_dir / "run_sessions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if (
        not sessions
        or sessions[-1].get("event") != "session_completed"
        or sessions[-1].get("timestamp") != completed
        or not isinstance(sessions[-1].get("wall_seconds"), (int, float))
        or not math.isfinite(float(sessions[-1]["wall_seconds"]))
        or float(sessions[-1]["wall_seconds"]) < 0.0
    ):
        raise ResumeAuthorizationError(
            "Pending terminal commit lacks its completed session event."
        )
    checkpoints = metrics.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        "best",
        "last",
    }:
        raise ResumeAuthorizationError(
            "Pending terminal commit lacks checkpoint records."
        )
    root = config.project_root.resolve()
    for name in ("best", "last"):
        item = checkpoints[name]
        if not isinstance(item, dict) or not isinstance(
            item.get("path"), str
        ):
            raise ResumeAuthorizationError(
                f"Pending terminal {name} checkpoint record is invalid."
            )
        path = (root / item["path"]).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or item.get("sha256") != _sha256_file(path)
        ):
            raise ResumeAuthorizationError(
                f"Pending terminal {name} checkpoint hash differs."
            )
    if checkpoints["last"].get("epoch") != config.epochs:
        raise ResumeAuthorizationError(
            "Pending terminal last checkpoint is not the final epoch."
        )
    history = metrics.get("auditable_artifacts")
    history = history.get("history") if isinstance(history, dict) else None
    history_path = (run_dir / "history.parquet").resolve()
    if (
        not isinstance(history, dict)
        or not isinstance(history.get("path"), str)
    ):
        raise ResumeAuthorizationError(
            "Pending terminal metrics lack the history artifact record."
        )
    recorded_history_path = (root / history["path"]).resolve()
    if (
        recorded_history_path != history_path
        or not history_path.is_relative_to(root)
        or not history_path.is_file()
        or history.get("sha256") != _sha256_file(history_path)
    ):
        raise ResumeAuthorizationError(
            "Pending terminal history artifact hash differs."
        )
    if interrupted_running_status:
        _atomic_text(
            run_dir / "STATUS.json",
            _canonical_json(
                {"status": "completed", "timestamp": completed}
            ),
        )
    _atomic_text(run_dir / "DONE", f"{completed}\n")
    return metrics


def _authorize_resume(
    run_dir: Path,
    *,
    last_path: Path,
    best_path: Path,
    history_path: Path,
) -> str:
    """Return the prior status only for an infrastructure-safe resume."""

    if (run_dir / "FAILED").exists():
        raise ResumeAuthorizationError(
            "A numerical/scientific FAILED run cannot be resumed."
        )
    status_path = run_dir / "STATUS.json"
    if not status_path.is_file():
        raise ResumeAuthorizationError(
            "Resume requires an existing STATUS.json audit record."
        )
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeAuthorizationError(
            "Resume STATUS.json is unreadable."
        ) from error
    if not isinstance(status, dict):
        raise ResumeAuthorizationError(
            "Resume STATUS.json must contain one mapping."
        )
    prior = status.get("status")
    if prior == "completed":
        if (
            not (run_dir / "DONE").exists()
            and (run_dir / "metrics.json").is_file()
            and last_path.is_file()
            and best_path.is_file()
            and history_path.is_file()
        ):
            return "terminal_commit_pending"
        raise ResumeAuthorizationError(
            "Completed status is resumable only for a missing DONE-last "
            "terminal commit."
        )
    if prior == "paused":
        if not last_path.is_file():
            raise ResumeAuthorizationError(
                "A planned pause requires an authoritative last.pt."
            )
        return prior
    if prior == "interrupted":
        if not (run_dir / "INTERRUPTED").is_file():
            raise ResumeAuthorizationError(
                "Interrupted resume lacks its audit marker."
            )
        if not last_path.is_file() and (
            best_path.exists() or history_path.exists()
        ):
            raise ResumeAuthorizationError(
                "Interrupted run has partial epoch artifacts without last.pt."
            )
        return prior
    if prior == "running":
        # A process killed by host preemption cannot rewrite STATUS.  It is
        # resumable only from a complete epoch checkpoint and only after the
        # recorded process is no longer alive.
        if _process_is_live(status.get("pid")):
            raise ResumeAuthorizationError(
                "The recorded P6 training process is still running."
            )
        if (
            not (run_dir / "DONE").exists()
            and (run_dir / "metrics.json").is_file()
            and (run_dir / "run_sessions.jsonl").is_file()
            and last_path.is_file()
            and best_path.is_file()
            and history_path.is_file()
        ):
            return "terminal_commit_pending"
        if last_path.is_file():
            return "unclean_host_interruption"
        if not best_path.exists() and not history_path.exists():
            return "unclean_host_interruption_restart_before_epoch1"
        raise ResumeAuthorizationError(
            "Unclean running state has partial epoch artifacts without "
            "authoritative last.pt."
        )
    raise ResumeAuthorizationError(
        f"Run status {prior!r} is not resumable under the P6 protocol."
    )


def _train_p6_impl(
    config: P6CausalTrainConfig,
    *,
    session_epoch_limit: int | None = None,
) -> dict[str, Any]:
    root = config.project_root.resolve()
    run_id = make_p6_run_id(config)
    config = replace(config, run_id=run_id)
    run_dir = config.output_root / run_id
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    history_path = run_dir / "history.parquet"
    if run_dir.exists() and not config.resume:
        raise FileExistsError(f"Run exists; use --resume: {run_dir}")
    if config.resume and (run_dir / "DONE").is_file():
        raise ResumeAuthorizationError(
            "A completed P6 run cannot be resumed."
        )
    prior_resume_status = None
    if config.resume:
        prior_resume_status = _authorize_resume(
            run_dir,
            last_path=last_path,
            best_path=best_path,
            history_path=history_path,
        )
    _validate_managed_launch_authorization(config, run_id=run_id)
    if prior_resume_status == "terminal_commit_pending":
        return _repair_pending_terminal_commit(
            run_dir,
            config=config,
        )
    resume_from_checkpoint = config.resume and last_path.is_file()
    restart_before_first_epoch = config.resume and not resume_from_checkpoint
    if restart_before_first_epoch and (
        best_path.exists() or history_path.exists()
    ):
        raise ResumeAuthorizationError(
            "Resume has partial epoch artifacts but no authoritative last.pt."
        )

    # Resolve and bind every launch identity before creating a new run
    # directory. A hard kill during this read-only preparation therefore
    # leaves no ambiguous new run. Resumes validate the prior identity before
    # mutating STATUS or session logs.
    wall_started = time.perf_counter()
    _configure_determinism(config.seed)
    device = _resolve_device(config.device)
    torch.set_num_threads(min(config.num_threads, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    implementation = _implementation_state(root)
    package_snapshot = _installed_package_snapshot()
    runtime = _runtime_fingerprint(
        device, package_snapshot=package_snapshot
    )
    checksums = _input_checksums(config)
    config, resource_profile = _resolve_resource_profile(
        config, checksums, implementation, device
    )
    launch_identity = _preparation_launch_identity(
        config=config,
        checksums=checksums,
        implementation=implementation,
        runtime=runtime,
        package_snapshot=package_snapshot,
        resource_profile=resource_profile,
    )
    session_started = datetime.now().astimezone().isoformat()
    session_event = {
        "event": "session_started",
        "timestamp": session_started,
        "resume": config.resume,
        "resume_from_checkpoint": resume_from_checkpoint,
        "restart_before_first_epoch": restart_before_first_epoch,
        "prior_resume_status": prior_resume_status,
    }
    running_status = {
        "status": "running",
        "timestamp": session_started,
        "pid": os.getpid(),
        "resume": config.resume,
        "prior_resume_status": prior_resume_status,
        "preparation_launch_identity_sha256": _sha256_json(
            launch_identity
        ),
    }

    if config.resume:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        _write_or_validate_preparation_launch_identity(
            run_dir / "preparation_launch_identity.json",
            launch_identity,
            resume=True,
        )
    else:
        launch_staging = config.output_root / (
            f".{run_id}.launching.{os.getpid()}"
        )
        if launch_staging.exists():
            raise FileExistsError(
                f"P6 launch staging path already exists: {launch_staging}"
            )
        (launch_staging / "checkpoints").mkdir(parents=True)
        _write_or_validate_preparation_launch_identity(
            launch_staging / "preparation_launch_identity.json",
            launch_identity,
            resume=False,
        )
        _atomic_text(
            launch_staging / "STARTED_AT", f"{session_started}\n"
        )
        _append_jsonl(
            launch_staging / "run_sessions.jsonl", session_event
        )
        _atomic_text(
            launch_staging / "STATUS.json",
            _canonical_json(running_status),
        )
        os.replace(launch_staging, run_dir)
    if resume_from_checkpoint:
        required_provenance = (
            "config_resolved.json",
            "data_checksums.json",
            "environment.json",
            "package_snapshot.json",
            "restriction_virtual_input.json",
            "git_state.txt",
        )
        missing = [
            name
            for name in required_provenance
            if not (run_dir / name).is_file()
        ]
        if missing:
            raise ResumeAuthorizationError(
                "Checkpoint resume lacks immutable provenance: "
                + ", ".join(missing)
            )

    logger = logging.getLogger(f"cdcureno.p6.{run_id}")
    _close_logger(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        run_dir / "stdout.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    started_path = run_dir / "STARTED_AT"
    if config.resume:
        (run_dir / "INTERRUPTED").unlink(missing_ok=True)
        if not started_path.is_file():
            raise ResumeAuthorizationError(
                "Resume lacks the original STARTED_AT timestamp."
            )
        started_at = started_path.read_text(encoding="utf-8").strip()
        _append_jsonl(run_dir / "run_sessions.jsonl", session_event)
        if isinstance(
            prior_resume_status, str
        ) and prior_resume_status.startswith("unclean_host_interruption"):
            _append_jsonl(
                run_dir / "run_sessions.jsonl",
                {
                    "event": "session_interruption_detected",
                    "timestamp": session_started,
                    "prior_resume_status": prior_resume_status,
                    "wall_seconds": None,
                },
            )
        _atomic_text(
            run_dir / "STATUS.json",
            _canonical_json(running_status),
        )
    else:
        started_at = session_started

    prepared = prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=config.label_budget,
        project_root=root,
        verify_array_checksums=config.verify_array_checksums,
    )
    if (
        prepared.label_budget != config.label_budget
        or tuple(prepared.channel_names) != tuple(TARGET_INPUT_CHANNELS)
    ):
        raise ValueError("Prepared target data violates the P6 contract.")
    frozen = load_frozen_physics_inputs(config)
    virtual, virtual_metadata = _prepare_training_virtual_pool(
        config, teacher_device=device
    )
    if not config.resume:
        _write_run_provenance(
            run_dir,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime=runtime,
            package_snapshot=package_snapshot,
            resource_profile=resource_profile,
            virtual_metadata=virtual_metadata,
        )
    elif restart_before_first_epoch:
        _ensure_restart_before_epoch_provenance(
            run_dir,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime=runtime,
            package_snapshot=package_snapshot,
            resource_profile=resource_profile,
            virtual_metadata=virtual_metadata,
        )
    else:
        _validate_existing_run_provenance(
            run_dir,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime=runtime,
            package_snapshot=package_snapshot,
            resource_profile=resource_profile,
            virtual_metadata=virtual_metadata,
        )

    model, model_spec, initialization = _initialize_p6_run_model(
        config,
        device=device,
    )
    groups, group_report = build_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    train_generator = torch.Generator().manual_seed(config.seed + 10_001)
    virtual_generator = torch.Generator().manual_seed(config.seed + 20_003)
    train_sampler = p5.StatefulShuffleSampler(
        len(prepared.train_dataset), train_generator
    )
    virtual_sampler = p5.StatefulShuffleSampler(
        len(virtual.inputs) if virtual is not None else 1,
        virtual_generator,
    )
    # DataLoader creates a base seed even with num_workers=0.  Isolate that
    # operational draw from the checkpointed global RNG so reconstruction of
    # the virtual iterator on resume cannot perturb scientific replay.
    train_loader_generator = torch.Generator().manual_seed(
        config.seed + 30_007
    )
    virtual_loader_generator = torch.Generator().manual_seed(
        config.seed + 40_009
    )
    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=config.micro_batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=train_loader_generator,
    )
    if virtual is not None:
        virtual_dataset = TensorDataset(
            virtual.inputs,
            torch.arange(len(virtual.inputs), dtype=torch.long),
        )
        virtual_loader: DataLoader | None = DataLoader(
            virtual_dataset,
            batch_size=config.micro_batch_size,
            sampler=virtual_sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=device.type == "cuda",
            generator=virtual_loader_generator,
        )
    else:
        virtual_loader = None
    if len(train_loader) % int(config.gradient_accumulation_steps) != 0:
        raise ValueError(
            "Frozen budgets must form complete effective batches."
        )

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation = float("inf")
    best_model: dict[str, torch.Tensor] | None = None
    if resume_from_checkpoint:
        checkpoint = _load_resume_checkpoint(last_path)
        _validate_resume_checkpoint(
            checkpoint,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime=runtime,
            resource_profile=resource_profile,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_model = _clone_state(checkpoint["best_model"])
        history = [dict(row) for row in checkpoint["history_rows"]]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation = float(checkpoint["best_validation"])
        train_generator.set_state(checkpoint["train_generator_state"])
        virtual_generator.set_state(checkpoint["virtual_generator_state"])
        train_sampler.load_state_dict(checkpoint["train_sampler_state"])
        virtual_sampler.load_state_dict(checkpoint["virtual_sampler_state"])
        _restore_rng(checkpoint)
        _write_derived_epoch_artifacts(
            checkpoint, best_path=best_path, history_path=history_path
        )

    virtual_iterator = (
        iter(virtual_loader) if virtual_loader is not None else None
    )
    executed_this_session = 0
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = {
            "temperature": 0.0,
            "alpha": 0.0,
            "gradient_x": 0.0,
            "gradient_z": 0.0,
            "data": 0.0,
            "physics": 0.0,
            "energy": 0.0,
            "kinetics": 0.0,
            "initial_condition": 0.0,
            "restriction": 0.0,
            "restriction_temperature": 0.0,
            "restriction_alpha": 0.0,
            "restriction_lis2": 0.0,
            "total": 0.0,
        }
        sample_count = 0
        optimizer_steps = 0
        last_gradient_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        for micro_index, (
            inputs,
            temperature,
            alpha,
            case_ids,
        ) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            temperature = temperature.to(
                device, non_blocking=device.type == "cuda"
            )
            alpha = alpha.to(device, non_blocking=device.type == "cuda")
            outputs, components = _data_components(
                model, inputs, temperature, alpha
            )
            data_loss = _data_objective(components, config)
            physics_loss, physics_terms = _training_physics_objective(
                outputs,
                [int(value) for value in case_ids],
                prepared=prepared,
                frozen=frozen,
                config=config,
            )
            if _restriction_enabled(config.method):
                if virtual is None or virtual_loader is None:
                    raise AssertionError(
                        "Restriction-enabled method lacks its virtual pool."
                    )
                (source_inputs, positions), virtual_iterator = _next_virtual(
                    virtual_loader, virtual_iterator
                )
                restriction = source_output_restriction_loss(
                    model,
                    source_inputs,
                    virtual.teacher.select(positions),
                    nx=int(inputs.shape[3]),
                    temperature_weight=(
                        config.source_temperature_restriction_weight
                    ),
                    alpha_weight=(
                        config.source_alpha_restriction_weight
                    ),
                    lateral_invariance_weight=(
                        config.source_lateral_invariance_weight
                    ),
                )
                restriction_loss = restriction.loss
                restriction_values = {
                    "temperature": restriction.temperature_relative_l2,
                    "alpha": restriction.alpha_relative_l2,
                    "lis2": (
                        restriction.temperature_lateral_invariance_squared
                        + restriction.alpha_lateral_invariance_squared
                    ),
                }
            else:
                restriction_loss = torch.zeros(
                    (), dtype=inputs.dtype, device=device
                )
                restriction_values = {
                    name: restriction_loss
                    for name in ("temperature", "alpha", "lis2")
                }
            objective = (
                data_loss
                + _effective_physics_multiplier(config) * physics_loss
                + _effective_restriction_multiplier(config)
                * restriction_loss
            )
            loss = objective / int(config.gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite P6 loss at epoch {epoch}."
                )
            loss.backward()
            if micro_index % int(config.gradient_accumulation_steps) == 0:
                last_gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                )
                if not np.isfinite(last_gradient_norm):
                    raise FloatingPointError("Non-finite P6 gradient norm.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            batch = len(inputs)
            for name, value in components.items():
                totals[name] += float(value.detach()) * batch
            totals["data"] += float(data_loss.detach()) * batch
            totals["physics"] += float(physics_loss.detach()) * batch
            for name, value in physics_terms.items():
                totals[name] += float(value.detach()) * batch
            totals["restriction"] += (
                float(restriction_loss.detach()) * batch
            )
            totals["restriction_temperature"] += (
                float(restriction_values["temperature"].detach()) * batch
            )
            totals["restriction_alpha"] += (
                float(restriction_values["alpha"].detach()) * batch
            )
            totals["restriction_lis2"] += (
                float(restriction_values["lis2"].detach()) * batch
            )
            totals["total"] += float(objective.detach()) * batch
            sample_count += batch
        scheduler.step()
        validation = p5._evaluate_objective(
            model, prepared, config, device
        )
        if not all(np.isfinite(value) for value in validation.values()):
            raise FloatingPointError("Non-finite P6 validation objective.")
        improved = float(validation["weighted"]) < best_validation
        if improved:
            best_validation = float(validation["weighted"])
            best_epoch = epoch
            best_model = _clone_state(model.state_dict())
        if best_model is None:
            raise AssertionError("First validation epoch must select a model.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "epoch": epoch,
            **{
                f"train_{name}": value / sample_count
                for name, value in totals.items()
            },
            "validation_temperature_relative_l2": validation[
                "temperature"
            ],
            "validation_alpha_relative_l2": validation["alpha"],
            "validation_gradient_x_relative_l2": validation["gradient_x"],
            "validation_gradient_z_relative_l2": validation["gradient_z"],
            "validation_data_objective": validation["weighted"],
            "optimizer_steps": optimizer_steps,
            "gradient_norm_before_clip_last_step": last_gradient_norm,
            "learning_rates": {
                str(group["group_name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "duration_seconds": time.perf_counter() - epoch_started,
            "improved": improved,
        }
        history.append(row)
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation=best_validation,
            best_model=best_model,
            history=history,
            config=config,
            checksums=checksums,
            prepared=prepared,
            implementation=implementation,
            runtime=runtime,
            initialization=initialization,
            parameter_groups=group_report,
            resource_profile=resource_profile,
            train_generator=train_generator,
            virtual_generator=virtual_generator,
            train_sampler=train_sampler,
            virtual_sampler=virtual_sampler,
        )
        _atomic_torch(checkpoint, last_path)
        _write_derived_epoch_artifacts(
            checkpoint, best_path=best_path, history_path=history_path
        )
        executed_this_session += 1
        logger.info(
            "epoch=%d/%d train=%.6f val=%.6f steps=%d seconds=%.3f",
            epoch,
            config.epochs,
            row["train_total"],
            row["validation_data_objective"],
            optimizer_steps,
            row["duration_seconds"],
        )
        if (
            session_epoch_limit is not None
            and executed_this_session >= session_epoch_limit
            and epoch < config.epochs
        ):
            paused = datetime.now().astimezone().isoformat()
            paused_wall_seconds = time.perf_counter() - wall_started
            _atomic_text(
                run_dir / "STATUS.json",
                _canonical_json(
                    {
                        "status": "paused",
                        "timestamp": paused,
                        "last_epoch": epoch,
                    }
                ),
            )
            _append_jsonl(
                run_dir / "run_sessions.jsonl",
                {
                    "event": "session_paused",
                    "timestamp": paused,
                    "last_epoch": epoch,
                    "wall_seconds": paused_wall_seconds,
                },
            )
            _close_logger(logger)
            return {
                "status": "paused",
                "run_id": run_id,
                "last_epoch": epoch,
                "current_session_wall_seconds": paused_wall_seconds,
                "target_test_or_ood_labels_evaluated": False,
            }

    authority = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    _write_derived_epoch_artifacts(
        authority, best_path=best_path, history_path=history_path
    )
    selected = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(selected["model"], strict=True)
    validation_summary, validation_cases, predictions = (
        p5._evaluate_validation_metrics(
            model, prepared, config, device
        )
    )
    validation_case_path = (
        run_dir / "metrics_per_case_validation.parquet"
    )
    validation_prediction_path = (
        run_dir / "predictions" / "validation_best.npz"
    )
    _atomic_parquet(validation_cases, validation_case_path)
    _atomic_npz(validation_prediction_path, **predictions)
    selected_physics = _evaluate_selected_physics(
        model, prepared, frozen, config, device
    )
    if not selected_physics["prediction_finite"] or any(
        not np.isfinite(value)
        for name, value in selected_physics.items()
        if name.endswith("_mean")
    ):
        raise FloatingPointError(
            "Selected P6 model has non-finite physics validation outputs."
        )
    restriction_gate_required = _restriction_enabled(config.method)
    if restriction_gate_required:
        if virtual is None:
            raise AssertionError(
                "Restriction-enabled selected model lacks its virtual pool."
            )
        audit_count = config.restriction_validation_case_count
        audit_inputs = virtual.inputs[:audit_count]
        audit_teacher = virtual.teacher.select(range(audit_count))
        restriction_audit = audit_selected_model_restriction(
            model,
            audit_inputs,
            audit_teacher,
            nx=40,
            rvs_max=config.restriction_rvs_max,
            lis_max=config.restriction_lis_max,
        )
    else:
        restriction_audit = {
            "schema_version": 1,
            "evaluated": False,
            "passed": None,
            "reason": "source_output_restriction_disabled_for_method",
            "source_checkpoint_loaded_for_teacher": False,
            "source_teacher_outputs_used": False,
            "target_2d_labels_used": False,
        }
    restriction_audit["gate_required_for_method"] = (
        restriction_gate_required
    )
    restriction_audit["gate_role"] = (
        "required" if restriction_gate_required else "diagnostic_only"
    )
    restriction_path = run_dir / "restriction_validation.json"
    _atomic_text(restriction_path, _canonical_json(restriction_audit))
    if restriction_gate_required and not restriction_audit["passed"]:
        raise RuntimeError(
            "Selected restriction-enabled model failed its source "
            "restriction gate."
        )
    causality = _causality_audit(
        model,
        prepared.validation_dataset[0][0],
        device=device,
        expected_causal=_expected_causal(config.method),
    )
    causality_path = run_dir / "causality_validation.json"
    _atomic_text(causality_path, _canonical_json(causality))
    if not causality["passed"]:
        raise RuntimeError(
            "Selected P6 model failed its registered temporal-structure "
            "and causality audit."
        )

    accessed = prepared.accessed_case_ids
    if set(accessed["train"]) != set(prepared.train_case_ids):
        raise PermissionError("P6 run did not access exactly its train budget.")
    if tuple(accessed["validation"]) != tuple(
        prepared.validation_case_ids
    ):
        raise PermissionError("P6 validation access is incomplete.")
    completed_at = datetime.now().astimezone().isoformat()
    wall_seconds = time.perf_counter() - wall_started
    wall_summary = _known_session_wall_summary(
        run_dir / "run_sessions.jsonl",
        current_completed_wall_seconds=wall_seconds,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "phase": "P6",
        "experiment": EXPERIMENT,
        "protocol_role": config.protocol_role,
        "run_id": run_id,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "physics_weight": config.physics_weight,
        "model": model_spec,
        "initialization": initialization,
        "parameter_groups": group_report,
        "selection": {
            "split": "validation",
            "objective": "common_data_objective_only",
            "selected_epoch": int(selected["epoch"]),
            "best_validation_objective": float(
                selected["best_validation"]
            ),
            "test_or_ood_labels_used": False,
        },
        "training": {
            "configured_epochs": config.epochs,
            "executed_epochs": len(history),
            "early_stopping": False,
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
                sum(float(row["duration_seconds"]) for row in history)
            ),
            "current_session_wall_seconds": wall_seconds,
            "known_session_wall_seconds_total": wall_summary[
                "known_session_wall_seconds"
            ],
            "known_terminal_session_count": wall_summary[
                "known_terminal_session_count"
            ],
            "unknown_interrupted_session_count": wall_summary[
                "unknown_interrupted_session_count"
            ],
            "session_wall_time_complete": wall_summary[
                "wall_time_complete"
            ],
            "peak_accelerator_memory_bytes": peak_memory,
            "optimizer_steps_total": int(
                sum(int(row["optimizer_steps"]) for row in history)
            ),
        },
        "loss": {
            "data_weights": {
                "temperature": config.temperature_weight,
                "alpha": config.alpha_weight,
                "gradient_x": config.gradient_x_weight,
                "gradient_z": config.gradient_z_weight,
            },
            "configured_physics_weight": config.physics_weight,
            "physics_multiplier": _effective_physics_multiplier(config),
            "physics_components": {
                "energy": config.energy_component_weight,
                "kinetics": config.kinetics_component_weight,
                "initial_condition": (
                    config.initial_condition_component_weight
                ),
            },
            "configured_source_restriction_multiplier": (
                config.source_restriction_multiplier
            ),
            "source_restriction_multiplier": (
                _effective_restriction_multiplier(config)
            ),
        },
        "validation_metrics": validation_summary,
        "selected_physics_validation": selected_physics,
        "restriction_validation": restriction_audit,
        "causality_validation": causality,
        "normalization": prepared.normalization,
        "input_checksums": checksums,
        "target_data_checksums": prepared.checksums,
        "runtime_fingerprint": runtime,
        "target_label_access_audit": {
            "train": list(accessed["train"]),
            "validation": list(accessed["validation"]),
            "id_test": [],
            "ood": [],
        },
        "auditable_artifacts": {
            "history": _artifact(history_path, root),
            "validation_metrics_per_case": _artifact(
                validation_case_path, root
            ),
            "validation_predictions": _artifact(
                validation_prediction_path, root
            ),
            "restriction_validation": _artifact(
                restriction_path, root
            ),
            "causality_validation": _artifact(causality_path, root),
            "environment": _artifact(
                run_dir / "environment.json", root
            ),
            "package_snapshot": _artifact(
                run_dir / "package_snapshot.json", root
            ),
            "preparation_launch_identity": _artifact(
                run_dir / "preparation_launch_identity.json", root
            ),
        },
        "checkpoints": {
            "best": {
                "path": _portable(best_path, root),
                "sha256": _sha256_file(best_path),
                "selected_epoch": int(selected["epoch"]),
            },
            "last": {
                "path": _portable(last_path, root),
                "sha256": _sha256_file(last_path),
                "epoch": int(authority["epoch"]),
            },
        },
        "resource_profile": resource_profile,
        "started_at": started_at,
        "completed_at": completed_at,
        "git_sha": implementation["head_sha"],
        "implementation_sha256": implementation["implementation_sha256"],
    }
    _atomic_text(run_dir / "metrics.json", _canonical_json(metrics))
    _append_jsonl(
        run_dir / "run_sessions.jsonl",
        {
            "event": "session_completed",
            "timestamp": completed_at,
            "wall_seconds": wall_seconds,
        },
    )
    _atomic_text(
        run_dir / "STATUS.json",
        _canonical_json(
            {"status": "completed", "timestamp": completed_at}
        ),
    )
    _atomic_text(run_dir / "DONE", f"{completed_at}\n")
    logger.info(
        "completed run_id=%s selected_epoch=%d validation_T=%.6f",
        run_id,
        int(selected["epoch"]),
        validation_summary[
            "temperature_relative_l2_K_composite_mean"
        ],
    )
    _close_logger(logger)
    return metrics


def train_p6_causal_target(
    config: P6CausalTrainConfig,
    *,
    session_epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Train or resume one leakage-safe P6 target run."""

    config = config.validated()
    if session_epoch_limit is not None and (
        isinstance(session_epoch_limit, bool)
        or not isinstance(session_epoch_limit, int)
        or session_epoch_limit < 1
    ):
        raise ValueError("session_epoch_limit must be a positive integer.")
    run_id = make_p6_run_id(config)
    run_dir = config.output_root / run_id
    existed_before_call = run_dir.exists()
    preserve_existing_state = existed_before_call and (
        not config.resume
        or not run_dir.is_dir()
        or (run_dir / "DONE").is_file()
    )
    call_wall_started = time.perf_counter()
    try:
        return _train_p6_impl(
            config, session_epoch_limit=session_epoch_limit
        )
    except ResumeAuthorizationError:
        _close_logger(logging.getLogger(f"cdcureno.p6.{run_id}"))
        raise
    except BaseException as error:
        _close_logger(logging.getLogger(f"cdcureno.p6.{run_id}"))
        if preserve_existing_state:
            raise
        failed_at = datetime.now().astimezone().isoformat()
        failed_wall_seconds = time.perf_counter() - call_wall_started
        infrastructure_interruption = _is_infrastructure_interruption(error)
        if (
            infrastructure_interruption
            and not existed_before_call
            and not (
                run_dir / "preparation_launch_identity.json"
            ).is_file()
        ):
            _append_jsonl(
                config.output_root
                / "p6_preparation_interruptions.jsonl",
                {
                    "event": "pre_identity_infrastructure_interruption",
                    "timestamp": failed_at,
                    "run_id": run_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "wall_seconds": failed_wall_seconds,
                    "registered_run_directory_created": False,
                    "resume_mode": "restart_locked_run_from_initialization",
                },
            )
            raise
        run_dir.mkdir(parents=True, exist_ok=True)
        last_path = run_dir / "checkpoints" / "last.pt"
        if infrastructure_interruption:
            _append_jsonl(
                run_dir / "failure_history.jsonl",
                {
                    "event": "infrastructure_interruption",
                    "timestamp": failed_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "authoritative_checkpoint_available": (
                        last_path.is_file()
                    ),
                    "wall_seconds": failed_wall_seconds,
                },
            )
            _append_jsonl(
                run_dir / "run_sessions.jsonl",
                {
                    "event": "session_interrupted",
                    "timestamp": failed_at,
                    "error_type": type(error).__name__,
                    "wall_seconds": failed_wall_seconds,
                },
            )
            _atomic_text(
                run_dir / "INTERRUPTED",
                f"{failed_at}\n{type(error).__name__}: {error}\n",
            )
            _atomic_text(
                run_dir / "STATUS.json",
                _canonical_json(
                    {
                        "status": "interrupted",
                        "timestamp": failed_at,
                        "classification": "infrastructure",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "authoritative_checkpoint_available": (
                            last_path.is_file()
                        ),
                    }
                ),
            )
            raise
        _append_jsonl(
            run_dir / "failure_history.jsonl",
            {
                "event": "failure",
                "timestamp": failed_at,
                "error_type": type(error).__name__,
                "error": str(error),
                "classification": "numerical_scientific_or_implementation",
                "resume_authorized": False,
                "wall_seconds": failed_wall_seconds,
            },
        )
        _append_jsonl(
            run_dir / "run_sessions.jsonl",
            {
                "event": "session_failed",
                "timestamp": failed_at,
                "error_type": type(error).__name__,
                "wall_seconds": failed_wall_seconds,
            },
        )
        _atomic_text(
            run_dir / "FAILED",
            f"{failed_at}\n{type(error).__name__}: {error}\n",
        )
        _atomic_text(
            run_dir / "STATUS.json",
            _canonical_json(
                {
                    "status": "failed",
                    "timestamp": failed_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "classification": (
                        "numerical_scientific_or_implementation"
                    ),
                    "resume_authorized": False,
                }
            ),
        )
        raise


def p6_causal_dry_run(
    config: P6CausalTrainConfig,
) -> dict[str, Any]:
    """Validate inputs, roster, architecture, and zero held-out access."""

    config = config.validated()
    checksums = _input_checksums(config)
    implementation = _implementation_state(config.project_root)
    prepared = prepare_target_2d_training(
        config.target_split_manifest,
        config.source_checkpoint,
        label_budget=config.label_budget,
        project_root=config.project_root,
        verify_array_checksums=config.verify_array_checksums,
    )
    model, spec, initialization = _initialize_p6_run_model(
        config,
    )
    _, groups = build_parameter_groups(model, config)
    frozen = load_frozen_physics_inputs(config)
    checks = {
        "target_train_budget_is_exact": (
            len(prepared.train_case_ids) == config.label_budget
        ),
        "validation_is_256_through_287": (
            prepared.validation_case_ids == tuple(range(256, 288))
        ),
        "target_labels_not_indexed": prepared.accessed_case_ids
        == {"train": (), "validation": ()},
        "test_or_ood_loader_not_constructed": True,
        "parameter_count_is_215698": (
            groups["total_parameter_count"] == 215_698
        ),
        "temporal_structure_matches_method": (
            bool(spec["structurally_causal"])
            is _expected_causal(config.method)
        ),
        "physics_multiplier_matches_method": (
            _effective_physics_multiplier(config)
            == (
                config.physics_weight
                if _physics_enabled(config.method)
                else 0.0
            )
        ),
        "restriction_multiplier_matches_method": (
            _effective_restriction_multiplier(config)
            == (
                config.source_restriction_multiplier
                if _restriction_enabled(config.method)
                else 0.0
            )
        ),
        "physics_grid_is_frozen": (
            frozen.time_s.shape == (112,)
            and frozen.z_m.shape == (50,)
            and frozen.x_m.shape == (40,)
        ),
    }
    resource_error: str | None = None
    resolved: P6CausalTrainConfig | None = None
    if config.require_resource_profile:
        try:
            device = _resolve_device(config.device)
            resolved, _ = _resolve_resource_profile(
                config, checksums, implementation, device
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            resource_error = f"{type(error).__name__}: {error}"
            checks["resource_profile_valid"] = False
        else:
            checks["resource_profile_valid"] = True
    return {
        "schema_version": 1,
        "phase": "P6",
        "dry_run": True,
        "protocol_role": config.protocol_role,
        "method": config.method,
        "label_budget": config.label_budget,
        "seed": config.seed,
        "physics_weight": config.physics_weight,
        "model": spec,
        "initialization": initialization,
        "parameter_groups": groups,
        "train_case_ids": list(prepared.train_case_ids),
        "validation_case_ids": list(prepared.validation_case_ids),
        "input_checksums": checksums,
        "target_data_checksums": prepared.checksums,
        "resolved_micro_batch_size": (
            None if resolved is None else resolved.micro_batch_size
        ),
        "resource_profile_error": resource_error,
        "checks": checks,
        "target_test_or_ood_labels_evaluated": False,
        "passed": bool(all(checks.values())),
    }
