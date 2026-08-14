"""Deterministic P3 1-D to P5 true-2-D checkpoint inflation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from cdcureno.models.joint_operators import FactorizedFNO
from cdcureno.models.target_operators import AxisFactorized2DOperator


SHARED_LIFT_MAPPING = {
    "lift.0.weight": "lift.shared.weight",
    "lift.0.bias": "lift.shared.bias",
    "lift.2.weight": "lift.projection.weight",
    "lift.2.bias": "lift.projection.bias",
}
VERIFICATION_ATOL = {
    "temperature": 1.0e-6,
    "temperature_residual": 1.0e-6,
    "alpha": 1.0e-6,
    "cure_rate": 1.0e-5,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _find_project_root(path: Path) -> Path | None:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _portable_path(path: Path, project_root: Path | None) -> str:
    if project_root is not None and path.is_relative_to(project_root):
        return path.relative_to(project_root).as_posix()
    return path.as_posix()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    return _sha256_bytes(values.view(torch.uint8).numpy().tobytes())


def load_target_config(path: Path) -> dict[str, Any]:
    """Load a YAML target contract and retain its immutable checksum."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Target config must be a YAML mapping.")
    payload = dict(payload)
    payload["_config_sha256"] = _sha256_file(path)
    return payload


def _source_spec(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Source checkpoint has no nonempty 'model' state mapping.")
    if "lift.0.weight" not in state:
        raise ValueError("Source checkpoint is not a supported P3 joint operator.")
    lift_weight = state["lift.0.weight"]
    if not isinstance(lift_weight, torch.Tensor) or lift_weight.ndim != 2:
        raise ValueError("Source lift.0.weight must be a rank-2 tensor.")
    block_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("blocks.") and key.count(".") >= 2
        }
    )
    if block_indices != list(range(len(block_indices))) or not block_indices:
        raise ValueError("Source block indices must be contiguous and nonempty.")
    if "blocks.0.temporal.weight" in state:
        temporal_family = "spectral_noncausal"
    elif "blocks.0.temporal.convolution.weight" in state:
        temporal_family = "causal_convolution"
    else:
        raise ValueError("Cannot infer the source temporal operator family.")
    if temporal_family != "spectral_noncausal":
        raise ValueError(
            "Exact inflation currently supports the P3 spectral FactorizedFNO "
            f"only, not {temporal_family!r}."
        )
    required = (
        "blocks.0.spatial.weight",
        "blocks.0.temporal.weight",
        "head.temperature_residual.2.weight",
        "head.cure_rate.2.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Source checkpoint is missing tensors: {missing}")
    channel_names_raw = checkpoint.get("channel_names")
    if not isinstance(channel_names_raw, Sequence) or isinstance(
        channel_names_raw, (str, bytes)
    ):
        raise ValueError("Source checkpoint must record ordered channel_names.")
    channel_names = tuple(str(name) for name in channel_names_raw)
    input_channels = int(lift_weight.shape[1])
    width = int(lift_weight.shape[0])
    if len(channel_names) != input_channels:
        raise ValueError("Source channel_names do not match the lift input width.")
    spec = {
        "family": "factorized_fno",
        "temporal_family": temporal_family,
        "causal": False,
        "input_channels": input_channels,
        "channel_names": channel_names,
        "width": width,
        "depth": len(block_indices),
        "modes_time": int(state["blocks.0.temporal.weight"].shape[-1]),
        "modes_z": int(state["blocks.0.spatial.weight"].shape[-1]),
    }
    source_model = FactorizedFNO(
        input_channels=input_channels,
        width=width,
        depth=spec["depth"],
        modes_time=spec["modes_time"],
        modes_space=spec["modes_z"],
    )
    try:
        source_model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Source state is not a strict FactorizedFNO checkpoint."
        ) from error
    return spec, state


def _validated_target_spec(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Target config schema_version must be 1.")
    if config.get("family") != "axis_factorized_2d":
        raise ValueError("Target family must be 'axis_factorized_2d'.")
    if config.get("source_family") != "factorized_fno":
        raise ValueError("source_family must be 'factorized_fno'.")
    expected_source_sha = config.get("expected_source_checkpoint_sha256")
    if expected_source_sha is not None:
        if (
            not isinstance(expected_source_sha, str)
            or len(expected_source_sha) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in expected_source_sha
            )
        ):
            raise ValueError(
                "expected_source_checkpoint_sha256 must be a SHA-256 digest."
            )
        if source_checkpoint_sha256 is None:
            raise ValueError(
                "This target contract pins a source checkpoint SHA-256; "
                "the caller must provide the computed source file digest."
            )
        if source_checkpoint_sha256.lower() != expected_source_sha.lower():
            raise ValueError(
                "Source checkpoint SHA-256 differs from the target contract."
            )
    if config.get("temporal_family") != source["temporal_family"]:
        raise ValueError(
            "Source and target temporal families differ; exact checkpoint "
            "inflation is impossible."
        )
    if bool(config.get("causal")):
        raise ValueError(
            "The current P3 source is noncausal. It cannot be advertised or "
            "inflated as an exactly equivalent causal target."
        )
    source_names_raw = config.get("source_channel_names")
    new_names_raw = config.get("new_channel_names")
    if not isinstance(source_names_raw, list) or not isinstance(
        new_names_raw, list
    ):
        raise ValueError(
            "source_channel_names and new_channel_names must be YAML lists."
        )
    source_names = tuple(str(name) for name in source_names_raw)
    new_names = tuple(str(name) for name in new_names_raw)
    if source_names != tuple(source["channel_names"]):
        raise ValueError(
            "Target shared-channel prefix must exactly equal the source "
            "checkpoint channel order."
        )
    if not new_names:
        raise ValueError("Target config must introduce at least one new channel.")
    inherited = {
        "width": int(config.get("width", -1)),
        "depth": int(config.get("depth", -1)),
        "modes_time": int(config.get("modes_time", -1)),
        "modes_z": int(config.get("modes_z", -1)),
    }
    for name, target_value in inherited.items():
        source_value = int(source[name])
        if target_value != source_value:
            raise ValueError(
                f"Target {name}={target_value} differs from source "
                f"{source_value}; exact tensor copying is impossible."
            )
    modes_x = int(config.get("modes_x", 0))
    lateral_rank = int(config.get("lateral_rank", 0))
    if min(modes_x, lateral_rank) < 1:
        raise ValueError("modes_x and lateral_rank must be positive.")
    if lateral_rank > inherited["width"]:
        raise ValueError("lateral_rank cannot exceed model width.")
    axis_order = config.get("axis_order")
    if axis_order != ["batch", "time", "z", "x", "channel"]:
        raise ValueError("Target axis_order must be [batch,time,z,x,channel].")
    return {
        **inherited,
        "modes_x": modes_x,
        "lateral_rank": lateral_rank,
        "source_channel_names": source_names,
        "new_channel_names": new_names,
        "channel_names": (*source_names, *new_names),
        "input_channels": len(source_names) + len(new_names),
        "family": "axis_factorized_2d",
        "source_family": "factorized_fno",
        "expected_source_checkpoint_sha256": expected_source_sha,
        "temporal_family": source["temporal_family"],
        "causal": False,
        "axis_order": axis_order,
    }


def _copy_source_state(
    target: AxisFactorized2DOperator,
    source_state: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_state = target.state_dict()
    copied: list[dict[str, Any]] = []
    for source_key, source_tensor in source_state.items():
        target_key = SHARED_LIFT_MAPPING.get(source_key, source_key)
        if target_key not in target_state:
            raise ValueError(
                f"No deterministic target mapping for source tensor {source_key!r}."
            )
        if target_state[target_key].shape != source_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {source_key!r} -> {target_key!r}: "
                f"{tuple(source_tensor.shape)} vs "
                f"{tuple(target_state[target_key].shape)}."
            )
        target_state[target_key].copy_(source_tensor)
        copied.append(
            {
                "source": source_key,
                "target": target_key,
                "shape": list(source_tensor.shape),
                "dtype": str(source_tensor.dtype),
                "sha256": _tensor_sha256(source_tensor),
            }
        )
    target.load_state_dict(target_state, strict=True)
    initialized = []
    copied_targets = {item["target"] for item in copied}
    for target_key, tensor in target.state_dict().items():
        if target_key in copied_targets:
            continue
        if target_key == "lift.geometry.weight":
            strategy = "zero_target_only_input_contribution"
            if torch.count_nonzero(tensor):
                raise AssertionError("Target-only lift must initialize to exact zero.")
        elif target_key.endswith(".lateral.input_factor"):
            strategy = "deterministic_nonzero_low_rank_input_factor"
            if not torch.count_nonzero(tensor):
                raise AssertionError("Lateral input factor must be nonzero.")
        elif target_key.endswith(".lateral.output_factor"):
            strategy = "zero_low_rank_output_factor"
            if torch.count_nonzero(tensor):
                raise AssertionError("Lateral output factor must be exact zero.")
        else:
            raise AssertionError(f"Unclassified initialized tensor: {target_key}")
        initialized.append(
            {
                "target": target_key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "strategy": strategy,
                "sha256": _tensor_sha256(tensor),
            }
        )
    for item in copied:
        if not torch.equal(
            source_state[item["source"]], target.state_dict()[item["target"]]
        ):
            raise AssertionError(f"Copied tensor changed: {item['source']}")
    return copied, initialized


def _verification_source_input(
    source_spec: Mapping[str, Any],
    *,
    seed: int,
    batch: int = 1,
    time_count: int = 29,
    z_count: int = 21,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(
        batch,
        time_count,
        z_count,
        int(source_spec["input_channels"]),
        generator=generator,
    )
    names = tuple(source_spec["channel_names"])
    time_index = names.index("time_normalized")
    z_index = names.index("through_thickness_position_normalized")
    mask_index = names.index("composite_mask")
    alpha_index = names.index("initial_degree_of_cure")
    values[..., time_index] = torch.linspace(0.0, 1.0, time_count)[
        None, :, None
    ]
    values[..., z_index] = torch.linspace(0.0, 1.0, z_count)[None, None, :]
    mask = torch.ones(batch, z_count)
    mask[:, : max(1, z_count // 4)] = 0.0
    values[..., mask_index] = mask[:, None, :]
    initial_alpha = 0.05 * torch.rand(
        batch, z_count, generator=generator
    ) * mask
    values[..., alpha_index] = initial_alpha[:, None, :]
    return values


def verify_restriction_preservation(
    source: FactorizedFNO,
    target: AxisFactorized2DOperator,
    *,
    seed: int,
    nx_values: Sequence[int] = (1, 2, 7, 40),
    tolerances: Mapping[str, float] = VERIFICATION_ATOL,
) -> dict[str, Any]:
    """Compare source and target on a label-free synthetic extrusion batch."""

    if not nx_values or any(int(value) < 1 for value in nx_values):
        raise ValueError("nx_values must contain positive grid sizes.")
    source_spec = {
        "input_channels": source.input_channels,
        "channel_names": target.source_channel_names,
    }
    inputs = _verification_source_input(
        source_spec, seed=seed + 100_003
    )
    source.eval()
    target.eval()
    rows: dict[str, Any] = {}
    passed = True
    with torch.no_grad():
        expected = source(inputs)
        for nx_raw in nx_values:
            nx = int(nx_raw)
            shared = inputs[:, :, :, None, :].expand(
                -1, -1, -1, nx, -1
            )
            generator = torch.Generator(device="cpu").manual_seed(
                seed + 200_003 + nx
            )
            geometry = torch.rand(
                *shared.shape[:-1],
                len(target.new_channel_names),
                generator=generator,
            )
            target_inputs = torch.cat((shared, geometry), dim=-1)
            actual = target(target_inputs)
            fields = {}
            nx_passed = True
            for field, tolerance in tolerances.items():
                reference = expected[field][:, :, :, None].expand(
                    -1, -1, -1, nx
                )
                difference = actual[field] - reference
                maximum = float(torch.max(torch.abs(difference)))
                relative = float(
                    torch.linalg.vector_norm(difference)
                    / torch.clamp(
                        torch.linalg.vector_norm(reference),
                        min=torch.finfo(reference.dtype).eps,
                    )
                )
                restricted = torch.mean(actual[field], dim=3)
                restriction_violation = float(
                    torch.linalg.vector_norm(restricted - expected[field])
                    / torch.clamp(
                        torch.linalg.vector_norm(expected[field]),
                        min=torch.finfo(reference.dtype).eps,
                    )
                )
                lateral_variance = torch.var(
                    actual[field], dim=3, unbiased=False
                )
                lateral_invariance = float(
                    torch.sqrt(torch.mean(lateral_variance))
                    / torch.clamp(
                        torch.std(actual[field], unbiased=False),
                        min=torch.finfo(reference.dtype).eps,
                    )
                )
                lateral_range = float(
                    torch.max(
                        torch.amax(actual[field], dim=3)
                        - torch.amin(actual[field], dim=3)
                    )
                )
                field_passed = maximum <= float(tolerance)
                nx_passed &= field_passed
                fields[field] = {
                    "maximum_absolute_mismatch": maximum,
                    "relative_l2_mismatch": relative,
                    "restriction_violation_score": restriction_violation,
                    "lateral_invariance_score": lateral_invariance,
                    "maximum_lateral_range": lateral_range,
                    "tolerance": float(tolerance),
                    "passed": field_passed,
                }
            bitwise = {
                field: bool(
                    torch.equal(actual[field].squeeze(3), expected[field])
                )
                for field in tolerances
            }
            if nx == 1:
                nx_passed &= all(bitwise.values())
            rows[str(nx)] = {
                "fields": fields,
                "source_target_bitwise_equal": bitwise,
                "passed": nx_passed,
            }
            passed &= nx_passed
    return {
        "uses_labels": False,
        "input_generation": "deterministic_synthetic_source_contract",
        "source_input_shape": list(inputs.shape),
        "tested_nx": [int(value) for value in nx_values],
        "per_nx": rows,
        "tolerances": {key: float(value) for key, value in tolerances.items()},
        "passed": passed,
    }


def inflate_checkpoint_payload(
    source_checkpoint: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    seed: int = 20260726,
    source_checkpoint_sha256: str | None = None,
) -> tuple[AxisFactorized2DOperator, dict[str, Any], dict[str, Any]]:
    """Inflate an in-memory checkpoint and return model, payload, and report."""

    source_spec, source_state = _source_spec(source_checkpoint)
    target_spec = _validated_target_spec(
        target_config,
        source_spec,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    pinned_seed = target_config.get("inflation_seed")
    if pinned_seed is not None and int(pinned_seed) != int(seed):
        raise ValueError(
            f"Inflation seed {seed} differs from the target contract "
            f"value {int(pinned_seed)}."
        )
    source_model = FactorizedFNO(
        input_channels=source_spec["input_channels"],
        width=source_spec["width"],
        depth=source_spec["depth"],
        modes_time=source_spec["modes_time"],
        modes_space=source_spec["modes_z"],
    )
    source_model.load_state_dict(source_state, strict=True)
    target_model = AxisFactorized2DOperator(
        source_channel_names=target_spec["source_channel_names"],
        new_channel_names=target_spec["new_channel_names"],
        width=target_spec["width"],
        depth=target_spec["depth"],
        modes_time=target_spec["modes_time"],
        modes_z=target_spec["modes_z"],
        modes_x=target_spec["modes_x"],
        lateral_rank=target_spec["lateral_rank"],
        adapter_seed=seed,
    )
    with torch.no_grad():
        target_model.lift.geometry.weight.zero_()
        for layer_index, block in enumerate(target_model.blocks):
            block.lateral.reset_zero_residual(seed + layer_index)
    copied, initialized = _copy_source_state(target_model, source_state)
    verification = verify_restriction_preservation(
        source_model, target_model, seed=seed
    )
    if not verification["passed"]:
        raise RuntimeError(
            "Inflated checkpoint failed source-target restriction verification."
        )
    shape_changes = [
        {
            "source": "lift.0.weight",
            "source_shape": list(source_state["lift.0.weight"].shape),
            "target_conceptual": "lift.combined_weight",
            "target_shape": list(target_model.lift.combined_weight().shape),
            "shared_columns_copied": len(target_spec["source_channel_names"]),
            "new_columns_zero": len(target_spec["new_channel_names"]),
        },
        *[
            {
                "target": f"blocks.{index}.lateral",
                "source_shape": None,
                "target_effective_kernel_shape": [
                    target_spec["width"],
                    target_spec["width"],
                    target_spec["modes_x"],
                ],
                "rank": target_spec["lateral_rank"],
                "zero_mode_anchored": True,
            }
            for index in range(target_spec["depth"])
        ],
    ]
    report = {
        "schema_version": 1,
        "phase": "P5",
        "operation": "restriction_preserving_1d_to_2d_checkpoint_inflation",
        "deterministic_seed": int(seed),
        "source": {
            **source_spec,
            "channel_names": list(source_spec["channel_names"]),
            "selected_epoch": source_checkpoint.get("epoch"),
            "validation_objective": source_checkpoint.get(
                "validation_objective"
            ),
        },
        "target": {
            **target_spec,
            "source_channel_names": list(target_spec["source_channel_names"]),
            "new_channel_names": list(target_spec["new_channel_names"]),
            "channel_names": list(target_spec["channel_names"]),
            "parameter_count": sum(
                parameter.numel() for parameter in target_model.parameters()
            ),
        },
        "tensor_mapping": {
            "copied_count": len(copied),
            "initialized_count": len(initialized),
            "copied": copied,
            "initialized": initialized,
            "shape_changes": shape_changes,
            "all_copied_tensors_bitwise_equal": True,
        },
        "verification": verification,
        "passed": True,
    }
    payload = {
        "schema_version": 1,
        "phase": "P5",
        "model": target_model.state_dict(),
        "model_config": {
            key: (
                list(value)
                if key
                in {
                    "source_channel_names",
                    "new_channel_names",
                    "channel_names",
                }
                else value
            )
            for key, value in target_spec.items()
        },
        "channel_names": target_spec["channel_names"],
        "source_channel_names": target_spec["source_channel_names"],
        "new_channel_names": target_spec["new_channel_names"],
        "source_checkpoint_epoch": source_checkpoint.get("epoch"),
        "source_validation_objective": source_checkpoint.get(
            "validation_objective"
        ),
        "normalization": source_checkpoint.get("normalization"),
        "inflation_report": report,
    }
    return target_model, payload, report


def verify_inflated_checkpoint_integrity(
    source_checkpoint: Mapping[str, Any],
    target_checkpoint: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
    target_config_sha256: str,
) -> dict[str, Any]:
    """Recreate and bitwise-audit a canonical untrained inflation payload."""

    if not isinstance(target_checkpoint, Mapping):
        raise ValueError("Target checkpoint must be a mapping.")
    report = target_checkpoint.get("inflation_report")
    if not isinstance(report, Mapping):
        raise ValueError("Target checkpoint has no embedded inflation report.")
    if "inflation_seed" not in target_config:
        raise ValueError("Target contract must pin inflation_seed.")
    seed = int(target_config["inflation_seed"])
    expected_model, expected_payload, expected_report = (
        inflate_checkpoint_payload(
            source_checkpoint,
            target_config,
            seed=seed,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
    )
    expected_report["source"]["checkpoint_sha256"] = (
        source_checkpoint_sha256
    )
    expected_report["target"]["config_sha256"] = target_config_sha256
    actual_state = target_checkpoint.get("model")
    if not isinstance(actual_state, Mapping):
        raise ValueError("Target checkpoint has no model state mapping.")
    expected_state = expected_model.state_dict()
    state_keys_match = set(actual_state) == set(expected_state)
    bitwise_state_match = state_keys_match and all(
        isinstance(actual_state[name], torch.Tensor)
        and torch.equal(actual_state[name], expected_tensor)
        for name, expected_tensor in expected_state.items()
    )
    initialized = report.get("tensor_mapping", {}).get("initialized", [])
    initialized_by_name = {
        str(item.get("target")): item
        for item in initialized
        if isinstance(item, Mapping)
    }
    strategy_state_matches = bool(bitwise_state_match)
    if state_keys_match:
        for name, tensor in actual_state.items():
            if name == "lift.geometry.weight":
                strategy_state_matches &= (
                    initialized_by_name.get(name, {}).get("strategy")
                    == "zero_target_only_input_contribution"
                    and not bool(torch.count_nonzero(tensor))
                )
            elif name.endswith(".lateral.input_factor"):
                strategy_state_matches &= (
                    initialized_by_name.get(name, {}).get("strategy")
                    == "deterministic_nonzero_low_rank_input_factor"
                    and bool(torch.count_nonzero(tensor))
                )
            elif name.endswith(".lateral.output_factor"):
                strategy_state_matches &= (
                    initialized_by_name.get(name, {}).get("strategy")
                    == "zero_low_rank_output_factor"
                    and not bool(torch.count_nonzero(tensor))
                )
    checks = {
        "checkpoint_schema": (
            target_checkpoint.get("schema_version") == 1
            and target_checkpoint.get("phase") == "P5"
        ),
        "embedded_report_passed": (
            report.get("passed") is True
            and isinstance(report.get("verification"), Mapping)
            and report["verification"].get("passed") is True
        ),
        "source_checkpoint_bound": (
            report.get("source", {}).get("checkpoint_sha256")
            == source_checkpoint_sha256
            and report.get("target", {}).get(
                "expected_source_checkpoint_sha256"
            )
            == source_checkpoint_sha256
        ),
        "target_config_bound": (
            report.get("target", {}).get("config_sha256")
            == target_config_sha256
        ),
        "deterministic_seed_bound": (
            report.get("deterministic_seed") == seed
        ),
        "model_config_matches_recreation": (
            target_checkpoint.get("model_config")
            == expected_payload["model_config"]
        ),
        "channel_contract_matches_recreation": (
            target_checkpoint.get("channel_names")
            == expected_payload["channel_names"]
            and target_checkpoint.get("source_channel_names")
            == expected_payload["source_channel_names"]
            and target_checkpoint.get("new_channel_names")
            == expected_payload["new_channel_names"]
        ),
        "source_metadata_matches_recreation": (
            target_checkpoint.get("source_checkpoint_epoch")
            == expected_payload["source_checkpoint_epoch"]
            and target_checkpoint.get("source_validation_objective")
            == expected_payload["source_validation_objective"]
            and target_checkpoint.get("normalization")
            == expected_payload["normalization"]
        ),
        "state_keys_match_recreation": state_keys_match,
        "all_state_tensors_bitwise_match_recreation": bitwise_state_match,
        "initialization_strategies_match_state": strategy_state_matches,
        "source_spec_matches_recreation": (
            report.get("source") == expected_report["source"]
        ),
        "target_spec_matches_recreation": (
            report.get("target") == expected_report["target"]
        ),
        "tensor_mapping_matches_recreation": (
            report.get("tensor_mapping")
            == expected_report["tensor_mapping"]
        ),
        "restriction_evidence_matches_recreation": (
            report.get("verification")
            == expected_report["verification"]
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(
            "Inflated checkpoint integrity verification failed: "
            f"{failed}"
        )
    return {
        "schema_version": 1,
        "verification": "deterministic_full_state_recreation",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "target_config_sha256": target_config_sha256,
        "deterministic_seed": seed,
        "state_tensor_count": len(expected_state),
        "checks": checks,
        "passed": True,
    }


def load_inflated_target(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AxisFactorized2DOperator, dict[str, Any]]:
    """Strictly reconstruct a saved P5 target checkpoint."""

    path = checkpoint_path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Inflated target checkpoint schema is unsupported.")
    spec = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(spec, dict) or not isinstance(state, Mapping):
        raise ValueError("Inflated checkpoint lacks model_config or model state.")
    if (
        spec.get("family") != "axis_factorized_2d"
        or spec.get("temporal_family") != "spectral_noncausal"
        or bool(spec.get("causal"))
    ):
        raise ValueError("Inflated checkpoint model family metadata is invalid.")
    model = AxisFactorized2DOperator(
        source_channel_names=spec["source_channel_names"],
        new_channel_names=spec["new_channel_names"],
        width=int(spec["width"]),
        depth=int(spec["depth"]),
        modes_time=int(spec["modes_time"]),
        modes_z=int(spec["modes_z"]),
        modes_x=int(spec["modes_x"]),
        lateral_rank=int(spec["lateral_rank"]),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Inflated target model state is incompatible.") from error
    model.to(device)
    return model, payload


def inflate_checkpoint_file(
    source_path: Path,
    target_config_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    seed: int = 20260726,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Inflate, verify, and atomically save a target checkpoint and report."""

    source_path = source_path.resolve()
    target_config_path = target_config_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    project_root = _find_project_root(target_config_path)
    for path, label in (
        (source_path, "source checkpoint"),
        (target_config_path, "target config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not dry_run and not overwrite:
        for path in (output_path, report_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing artifact: {path}"
                )
    # The project checkpoint contains normalization and channel metadata, so it
    # must be loaded as a trusted local research artifact.
    source_checkpoint = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    config = load_target_config(target_config_path)
    source_sha256 = _sha256_file(source_path)
    _, payload, report = inflate_checkpoint_payload(
        source_checkpoint,
        config,
        seed=seed,
        source_checkpoint_sha256=source_sha256,
    )
    report["source"]["checkpoint_sha256"] = source_sha256
    report["target"]["config_sha256"] = config["_config_sha256"]
    report["artifacts"] = {
        "source_checkpoint": _portable_path(source_path, project_root),
        "target_config": _portable_path(target_config_path, project_root),
        "target_checkpoint": _portable_path(output_path, project_root),
        "report": _portable_path(report_path, project_root),
        "dry_run": bool(dry_run),
    }
    payload["inflation_report"] = report
    if dry_run:
        return report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    try:
        torch.save(payload, checkpoint_temporary)
        report_temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(checkpoint_temporary, output_path)
        os.replace(report_temporary, report_path)
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
    return report
