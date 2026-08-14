"""Exact causal-P3 to causal true-2-D checkpoint inflation.

This module is deliberately separate from ``checkpoint_inflation``.  The
existing RP-FFNO pilot transfers a noncausal temporal Fourier model, whereas
this path copies a structurally causal, dilated-convolution source into a
target with no temporal FFT.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    CausalTemporalConv1d,
    TemporalSpectralConv1d,
)
from cdcureno.models.target_operators import (
    CausalAxisFactorized2DBlock,
    CausalAxisFactorized2DOperator,
)


CAUSAL_SHARED_LIFT_MAPPING = {
    "lift.0.weight": "lift.shared.weight",
    "lift.0.bias": "lift.shared.bias",
    "lift.2.weight": "lift.projection.weight",
    "lift.2.bias": "lift.projection.bias",
}
CAUSAL_VERIFICATION_ATOL = {
    "temperature": 1.0e-6,
    "temperature_residual": 1.0e-6,
    "alpha": 1.0e-6,
    "cure_rate": 1.0e-5,
}
CAUSAL_OUTPUT_FIELDS = tuple(CAUSAL_VERIFICATION_ATOL)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact dtype-level bytes of a tensor on CPU."""

    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        values.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return value.lower()


def _find_project_root(path: Path) -> Path | None:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _portable_path(path: Path, project_root: Path | None) -> str:
    if project_root is not None and path.is_relative_to(project_root):
        return path.relative_to(project_root).as_posix()
    return path.as_posix()


def load_causal_target_config(path: Path) -> dict[str, Any]:
    """Load the causal target YAML and attach its exact file digest."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Causal target config must be a YAML mapping.")
    result = dict(payload)
    result["_config_sha256"] = sha256_file(path)
    return result


def load_causal_source_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[
    CausalFactorizedOperator,
    dict[str, Any],
    Mapping[str, torch.Tensor],
]:
    """Strictly reconstruct and describe a causal P3 source checkpoint."""

    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Causal source checkpoint has no model state.")
    if checkpoint.get("model_family") != "causal_factorized":
        raise ValueError("Source model_family must be 'causal_factorized'.")
    model_spec = checkpoint.get("model_spec")
    if not isinstance(model_spec, Mapping):
        raise ValueError("Causal source checkpoint has no model_spec mapping.")
    if (
        model_spec.get("family") != "causal_factorized"
        or model_spec.get("structurally_causal") is not True
    ):
        raise ValueError("Source model_spec does not certify structural causality.")
    lift_weight = state.get("lift.0.weight")
    if not isinstance(lift_weight, torch.Tensor) or lift_weight.ndim != 2:
        raise ValueError("Source lift.0.weight must be a rank-2 tensor.")
    block_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("blocks.") and key.count(".") >= 2
        }
    )
    if not block_indices or block_indices != list(range(len(block_indices))):
        raise ValueError("Source block indices must be contiguous and nonempty.")
    required = (
        "blocks.0.spatial.weight",
        "blocks.0.temporal.convolution.weight",
        "blocks.0.temporal.convolution.bias",
        "head.temperature_residual.2.weight",
        "head.cure_rate.2.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Causal source checkpoint is missing tensors: {missing}")
    channel_names_raw = checkpoint.get("channel_names")
    if not isinstance(channel_names_raw, Sequence) or isinstance(
        channel_names_raw, (str, bytes)
    ):
        raise ValueError("Causal source must record ordered channel_names.")
    channel_names = tuple(str(value) for value in channel_names_raw)
    width = int(lift_weight.shape[0])
    input_channels = int(lift_weight.shape[1])
    depth = len(block_indices)
    modes_z = int(state["blocks.0.spatial.weight"].shape[-1])
    temporal_weight = state["blocks.0.temporal.convolution.weight"]
    if not isinstance(temporal_weight, torch.Tensor) or temporal_weight.ndim != 3:
        raise ValueError("Source temporal convolution weight has the wrong rank.")
    temporal_kernel_size = int(temporal_weight.shape[-1])
    temporal_dilations_raw = model_spec.get("temporal_dilations")
    if not isinstance(temporal_dilations_raw, Sequence) or isinstance(
        temporal_dilations_raw, (str, bytes)
    ):
        raise ValueError("Source model_spec has no temporal_dilations sequence.")
    temporal_dilations = tuple(int(value) for value in temporal_dilations_raw)
    expected_dilations = tuple(2 ** (index % 8) for index in range(depth))
    temporal_receptive_field = 1 + sum(
        (temporal_kernel_size - 1) * dilation
        for dilation in expected_dilations
    )
    expected = {
        "input_channels": input_channels,
        "channel_names": list(channel_names),
        "width": width,
        "depth": depth,
        "modes_space": modes_z,
        "temporal_kernel_size": temporal_kernel_size,
        "temporal_dilations": list(expected_dilations),
        "temporal_receptive_field": temporal_receptive_field,
    }
    for key, value in expected.items():
        if model_spec.get(key) != value:
            raise ValueError(
                f"Source model_spec {key!r} does not match its model tensors."
            )
    if len(channel_names) != input_channels:
        raise ValueError("Source channel_names do not match the lift width.")
    if temporal_dilations != expected_dilations:
        raise ValueError(
            "Source dilations do not match CausalFactorizedOperator construction."
        )
    model = CausalFactorizedOperator(
        input_channels=input_channels,
        width=width,
        depth=depth,
        modes_space=modes_z,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Source state is not a strict CausalFactorizedOperator checkpoint."
        ) from error
    spec = {
        "family": "causal_factorized",
        "temporal_family": "causal_dilated_convolution",
        "causal": True,
        "structurally_causal": True,
        "input_channels": input_channels,
        "channel_names": channel_names,
        "width": width,
        "depth": depth,
        "modes_z": modes_z,
        "temporal_kernel_size": temporal_kernel_size,
        "temporal_dilations": temporal_dilations,
        "temporal_receptive_field": temporal_receptive_field,
    }
    return model, spec, state


def _validated_target_spec(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
    target_config_sha256: str,
    inflation_seed: int | None = None,
) -> dict[str, Any]:
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("Causal target config schema_version must be 1.")
    if config.get("family") != "causal_axis_factorized_2d":
        raise ValueError("Target family must be 'causal_axis_factorized_2d'.")
    if config.get("source_family") != "causal_factorized":
        raise ValueError("source_family must be 'causal_factorized'.")
    actual_source_sha = _validate_sha256(
        source_checkpoint_sha256,
        label="source_checkpoint_sha256",
    )
    expected_source_sha = _validate_sha256(
        config.get("expected_source_checkpoint_sha256"),
        label="expected_source_checkpoint_sha256",
    )
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            "Source checkpoint SHA-256 differs from the causal target contract."
        )
    actual_config_sha = _validate_sha256(
        target_config_sha256,
        label="target_config_sha256",
    )
    frozen_inflation_seed = int(config.get("inflation_seed", -1))
    if frozen_inflation_seed < 0:
        raise ValueError("inflation_seed must be a nonnegative integer.")
    if (
        inflation_seed is not None
        and int(inflation_seed) != frozen_inflation_seed
    ):
        raise ValueError(
            "Requested inflation seed differs from the causal target contract."
        )
    embedded_config_sha = config.get("_config_sha256")
    if embedded_config_sha is not None and _validate_sha256(
        embedded_config_sha,
        label="_config_sha256",
    ) != actual_config_sha:
        raise ValueError("Target config SHA-256 binding is inconsistent.")
    if (
        config.get("temporal_family") != source["temporal_family"]
        or config.get("causal") is not True
        or config.get("structurally_causal") is not True
    ):
        raise ValueError(
            "Causal source and target temporal/causality contracts differ."
        )
    source_names_raw = config.get("source_channel_names")
    new_names_raw = config.get("new_channel_names")
    if not isinstance(source_names_raw, list) or not isinstance(
        new_names_raw, list
    ):
        raise ValueError(
            "source_channel_names and new_channel_names must be YAML lists."
        )
    source_names = tuple(str(value) for value in source_names_raw)
    new_names = tuple(str(value) for value in new_names_raw)
    if source_names != tuple(source["channel_names"]):
        raise ValueError(
            "Target shared-channel prefix differs from the source checkpoint."
        )
    if not new_names:
        raise ValueError("Causal target config must introduce new channels.")
    inherited = {
        "width": int(config.get("width", -1)),
        "depth": int(config.get("depth", -1)),
        "modes_z": int(config.get("modes_z", -1)),
        "temporal_kernel_size": int(
            config.get("temporal_kernel_size", -1)
        ),
    }
    for name, target_value in inherited.items():
        if target_value != int(source[name]):
            raise ValueError(
                f"Target {name}={target_value} differs from source "
                f"{source[name]}; exact copying is impossible."
            )
    dilations_raw = config.get("temporal_dilations")
    if not isinstance(dilations_raw, list):
        raise ValueError("temporal_dilations must be a YAML list.")
    dilations = tuple(int(value) for value in dilations_raw)
    if dilations != tuple(source["temporal_dilations"]):
        raise ValueError("Target temporal dilations differ from the source.")
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
        "temporal_dilations": dilations,
        "temporal_receptive_field": int(source["temporal_receptive_field"]),
        "modes_x": modes_x,
        "lateral_rank": lateral_rank,
        "source_channel_names": source_names,
        "new_channel_names": new_names,
        "channel_names": (*source_names, *new_names),
        "input_channels": len(source_names) + len(new_names),
        "family": "causal_axis_factorized_2d",
        "source_family": "causal_factorized",
        "expected_source_checkpoint_sha256": expected_source_sha,
        "source_checkpoint_sha256": actual_source_sha,
        "target_config_sha256": actual_config_sha,
        "inflation_seed": frozen_inflation_seed,
        "temporal_family": "causal_dilated_convolution",
        "causal": True,
        "structurally_causal": True,
        "axis_order": axis_order,
    }


def _construct_target(
    spec: Mapping[str, Any],
    *,
    adapter_seed: int = 0,
) -> CausalAxisFactorized2DOperator:
    return CausalAxisFactorized2DOperator(
        source_channel_names=spec["source_channel_names"],
        new_channel_names=spec["new_channel_names"],
        width=int(spec["width"]),
        depth=int(spec["depth"]),
        modes_z=int(spec["modes_z"]),
        modes_x=int(spec["modes_x"]),
        lateral_rank=int(spec["lateral_rank"]),
        temporal_kernel_size=int(spec["temporal_kernel_size"]),
        temporal_dilations=spec["temporal_dilations"],
        adapter_seed=adapter_seed,
    )


def _copy_source_state(
    target: CausalAxisFactorized2DOperator,
    source_state: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_state = target.state_dict()
    copied: list[dict[str, Any]] = []
    with torch.no_grad():
        for source_key, source_tensor in source_state.items():
            target_key = CAUSAL_SHARED_LIFT_MAPPING.get(source_key, source_key)
            target_tensor = target_state.get(target_key)
            if target_tensor is None:
                raise ValueError(
                    f"No causal target mapping for source tensor {source_key!r}."
                )
            if target_tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {source_key!r} -> {target_key!r}."
                )
            target_tensor.copy_(source_tensor)
            copied.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "shape": list(source_tensor.shape),
                    "dtype": str(source_tensor.dtype),
                    "source_sha256": tensor_sha256(source_tensor),
                    "target_sha256": tensor_sha256(target_tensor),
                }
            )
    target.load_state_dict(target_state, strict=True)
    copied_targets = {item["target"] for item in copied}
    initialized: list[dict[str, Any]] = []
    for target_key, tensor in target.state_dict().items():
        if target_key in copied_targets:
            continue
        if target_key == "lift.geometry.weight":
            strategy = "zero_target_only_input_contribution"
            if torch.count_nonzero(tensor):
                raise AssertionError("Target-only lift must be exactly zero.")
        elif target_key.endswith(".lateral.input_factor"):
            strategy = "deterministic_nonzero_high_pass_input_factor"
            if not torch.count_nonzero(tensor):
                raise AssertionError("Lateral input factor must be nonzero.")
        elif target_key.endswith(".lateral.output_factor"):
            strategy = "zero_high_pass_output_factor"
            if torch.count_nonzero(tensor):
                raise AssertionError("Lateral output factor must be exactly zero.")
        else:
            raise AssertionError(f"Unclassified initialized tensor: {target_key}")
        initialized.append(
            {
                "target": target_key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "strategy": strategy,
                "sha256": tensor_sha256(tensor),
            }
        )
    for item in copied:
        source_tensor = source_state[item["source"]]
        target_tensor = target.state_dict()[item["target"]]
        if not torch.equal(source_tensor, target_tensor):
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
    values[..., names.index("time_normalized")] = torch.linspace(
        0.0, 1.0, time_count
    )[None, :, None]
    values[
        ..., names.index("through_thickness_position_normalized")
    ] = torch.linspace(0.0, 1.0, z_count)[None, None, :]
    mask = torch.ones(batch, z_count)
    mask[:, : max(1, z_count // 4)] = 0.0
    values[..., names.index("composite_mask")] = mask[:, None, :]
    initial_alpha = (
        0.05 * torch.rand(batch, z_count, generator=generator) * mask
    )
    values[..., names.index("initial_degree_of_cure")] = initial_alpha[
        :, None, :
    ]
    return values


def verify_causal_models_on_input(
    source: CausalFactorizedOperator,
    target: CausalAxisFactorized2DOperator,
    inputs: torch.Tensor,
    *,
    seed: int,
    nx_values: Sequence[int] = (1, 2, 7, 40),
    tolerances: Mapping[str, float] = CAUSAL_VERIFICATION_ATOL,
) -> dict[str, Any]:
    """Verify exact/tight restriction on one source-compatible input batch."""

    if (
        inputs.ndim != 4
        or inputs.shape[-1] != len(target.source_channel_names)
    ):
        raise ValueError("Expected source-compatible [B,Nt,Nz,C] inputs.")
    if not nx_values or any(int(value) < 1 for value in nx_values):
        raise ValueError("nx_values must contain positive grid sizes.")
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
            generator = torch.Generator(device="cpu").manual_seed(seed + nx)
            geometry = torch.randn(
                *shared.shape[:-1],
                len(target.new_channel_names),
                generator=generator,
                dtype=inputs.dtype,
            )
            actual = target(torch.cat((shared, geometry), dim=-1))
            fields: dict[str, Any] = {}
            nx_passed = True
            for field, tolerance in tolerances.items():
                reference = expected[field][:, :, :, None].expand(
                    -1, -1, -1, nx
                )
                difference = actual[field] - reference
                maximum = float(torch.max(torch.abs(difference)))
                relative = float(
                    torch.linalg.vector_norm(difference.double())
                    / torch.linalg.vector_norm(reference.double()).clamp_min(
                        torch.finfo(torch.float64).eps
                    )
                )
                lateral_range = float(
                    torch.max(
                        torch.amax(actual[field], dim=3)
                        - torch.amin(actual[field], dim=3)
                    )
                )
                field_passed = (
                    maximum <= float(tolerance)
                    and lateral_range <= float(tolerance)
                )
                nx_passed &= field_passed
                fields[field] = {
                    "maximum_absolute_mismatch": maximum,
                    "relative_l2_mismatch": relative,
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
                "nx_one_source_target_bitwise_equal": bitwise,
                "passed": nx_passed,
            }
            passed &= nx_passed
    return {
        "uses_labels": False,
        "input_shape": list(inputs.shape),
        "tested_nx": [int(value) for value in nx_values],
        "per_nx": rows,
        "tolerances": {key: float(value) for key, value in tolerances.items()},
        "passed": passed,
    }


def verify_target_future_invariance(
    target: CausalAxisFactorized2DOperator,
    inputs: torch.Tensor,
    *,
    seed: int,
    nx: int = 7,
    tolerance: float = 1.0e-7,
    cutoff_fractions: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Perturb every future target channel and compare all output prefixes."""

    if inputs.ndim != 4:
        raise ValueError("Expected source inputs shaped [B,Nt,Nz,C].")
    time_count = int(inputs.shape[1])
    cutoffs = sorted(
        {
            min(
                time_count - 2,
                max(0, int(round((time_count - 1) * float(fraction)))),
            )
            for fraction in cutoff_fractions
        }
    )
    if not cutoffs or any(not 0.0 < float(value) < 1.0 for value in cutoff_fractions):
        raise ValueError("Causality cutoff fractions must lie in (0, 1).")
    shared = inputs[:, :, :, None, :].expand(-1, -1, -1, nx, -1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    geometry = torch.randn(
        *shared.shape[:-1],
        len(target.new_channel_names),
        generator=generator,
        dtype=inputs.dtype,
    )
    target_inputs = torch.cat((shared, geometry), dim=-1)
    offsets = torch.linspace(
        -0.17,
        0.19,
        target_inputs.shape[-1],
        dtype=target_inputs.dtype,
        device=target_inputs.device,
    )
    rows: list[dict[str, Any]] = []
    target.eval()
    with torch.no_grad():
        reference = target(target_inputs)
        for cutoff in cutoffs:
            counterfactual = target_inputs.clone()
            future = counterfactual[:, cutoff + 1 :]
            counterfactual[:, cutoff + 1 :] = (
                torch.flip(future, dims=(1,)) + offsets
            )
            changed = target(counterfactual)
            for field in CAUSAL_OUTPUT_FIELDS:
                baseline_prefix = reference[field][:, : cutoff + 1]
                difference = changed[field][:, : cutoff + 1] - baseline_prefix
                maximum = float(torch.max(torch.abs(difference)))
                relative = float(
                    torch.linalg.vector_norm(difference.double())
                    / torch.linalg.vector_norm(
                        baseline_prefix.double()
                    ).clamp_min(torch.finfo(torch.float64).eps)
                )
                rows.append(
                    {
                        "cutoff_index": cutoff,
                        "future_start_index": cutoff + 1,
                        "output": field,
                        "maximum_prefix_abs_difference": maximum,
                        "prefix_relative_l2": relative,
                        "passed": maximum <= tolerance,
                    }
                )
    return {
        "test": "all_channel_counterfactual_future_perturbation",
        "perturbed_channel_count": int(target_inputs.shape[-1]),
        "tested_nx": int(nx),
        "cutoff_indices": cutoffs,
        "tolerance": float(tolerance),
        "maximum_prefix_abs_difference": max(
            float(row["maximum_prefix_abs_difference"]) for row in rows
        ),
        "maximum_prefix_relative_l2": max(
            float(row["prefix_relative_l2"]) for row in rows
        ),
        "rows": rows,
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
    }


def inspect_causal_target_architecture(
    target: CausalAxisFactorized2DOperator,
) -> dict[str, Any]:
    """Fail closed if a temporal FFT or noncausal target block is present."""

    blocks = list(target.blocks)
    temporal_fft_count = sum(
        isinstance(module, TemporalSpectralConv1d)
        for module in target.modules()
    )
    causal_temporal_count = sum(
        isinstance(module, CausalTemporalConv1d)
        for module in target.modules()
    )
    block_types_valid = all(
        isinstance(block, CausalAxisFactorized2DBlock) for block in blocks
    )
    dilations = [
        int(block.temporal.convolution.dilation[0]) for block in blocks
    ]
    left_paddings = [int(block.temporal.left_padding) for block in blocks]
    passed = bool(
        block_types_valid
        and temporal_fft_count == 0
        and causal_temporal_count == target.depth
        and tuple(dilations) == target.temporal_dilations
        and all(value >= 0 for value in left_paddings)
    )
    return {
        "model_family": target.family,
        "temporal_family": target.temporal_family,
        "structurally_causal": target.structurally_causal,
        "target_block_count": len(blocks),
        "causal_temporal_module_count": causal_temporal_count,
        "temporal_fft_module_count": temporal_fft_count,
        "temporal_dilations": dilations,
        "left_paddings": left_paddings,
        "passed": passed,
    }


def verify_causal_checkpoint_integrity(
    source_checkpoint: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
    target_config_sha256: str,
) -> dict[str, Any]:
    """Recompute every tensor/config binding in an inflated checkpoint."""

    source_model, source_spec, source_state = load_causal_source_checkpoint(
        source_checkpoint
    )
    expected_spec = _validated_target_spec(
        target_config,
        source_spec,
        source_checkpoint_sha256=source_checkpoint_sha256,
        target_config_sha256=target_config_sha256,
    )
    if (
        target_payload.get("schema_version") != 1
        or target_payload.get("phase") != "P5"
        or target_payload.get("experiment")
        != "causal_source_to_causal_target_inflation_v1"
        or target_payload.get("model_family")
        != "causal_axis_factorized_2d"
    ):
        raise ValueError("Inflated causal target payload metadata is invalid.")
    metadata_checks = {
        "channel_names": tuple(
            target_payload.get("channel_names", ())
        )
        == tuple(expected_spec["channel_names"]),
        "source_channel_names": tuple(
            target_payload.get("source_channel_names", ())
        )
        == tuple(expected_spec["source_channel_names"]),
        "new_channel_names": tuple(
            target_payload.get("new_channel_names", ())
        )
        == tuple(expected_spec["new_channel_names"]),
        "source_checkpoint_epoch": target_payload.get(
            "source_checkpoint_epoch"
        )
        == source_checkpoint.get("epoch"),
        "source_validation_objective": target_payload.get(
            "source_validation_objective"
        )
        == source_checkpoint.get("best_validation"),
        "normalization": target_payload.get("normalization")
        == source_checkpoint.get("normalization"),
    }
    if not all(metadata_checks.values()):
        failed = [name for name, value in metadata_checks.items() if not value]
        raise ValueError(
            "Inflated causal target source metadata differs: "
            f"{failed}."
        )
    state = target_payload.get("model")
    model_config = target_payload.get("model_config")
    report = target_payload.get("inflation_report")
    if (
        not isinstance(state, Mapping)
        or not isinstance(model_config, Mapping)
        or not isinstance(report, Mapping)
    ):
        raise ValueError("Inflated causal target payload is incomplete.")
    for key, expected_value in expected_spec.items():
        actual_value = model_config.get(key)
        if isinstance(expected_value, tuple):
            actual_value = tuple(actual_value) if actual_value is not None else None
        if actual_value != expected_value:
            raise ValueError(f"Target model_config binding differs for {key!r}.")
    report_source = report.get("source")
    report_target = report.get("target")
    tensor_mapping = report.get("tensor_mapping")
    if (
        not isinstance(report_source, Mapping)
        or not isinstance(report_target, Mapping)
        or not isinstance(tensor_mapping, Mapping)
    ):
        raise ValueError("Inflation report binding sections are missing.")
    if (
        report_source.get("checkpoint_sha256")
        != expected_spec["source_checkpoint_sha256"]
        or report_target.get("config_sha256")
        != expected_spec["target_config_sha256"]
        or report.get("deterministic_seed")
        != expected_spec["inflation_seed"]
        or report_source.get("selected_epoch")
        != source_checkpoint.get("epoch")
        or report_source.get("validation_objective")
        != source_checkpoint.get("best_validation")
        or tuple(report_target.get("channel_names", ()))
        != tuple(expected_spec["channel_names"])
    ):
        raise ValueError(
            "Inflation report source/config/seed binding differs."
        )
    if (
        report_source.get("family") != "causal_factorized"
        or report_target.get("family") != "causal_axis_factorized_2d"
        or report.get("passed") is not True
    ):
        raise ValueError("Inflation report family or pass metadata differs.")
    model = _construct_target(model_config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Inflated causal target state is incompatible.") from error
    target_state = model.state_dict()
    expected_model = _construct_target(
        expected_spec,
        adapter_seed=int(expected_spec["inflation_seed"]),
    )
    with torch.no_grad():
        expected_model.lift.geometry.weight.zero_()
        for layer_index, block in enumerate(expected_model.blocks):
            block.lateral.reset_zero_residual(
                int(expected_spec["inflation_seed"]) + layer_index
            )
    _copy_source_state(expected_model, source_state)
    expected_target_state = expected_model.state_dict()
    if set(target_state) != set(expected_target_state):
        raise ValueError("Inflated target state key set differs from expectation.")
    for target_key, target_tensor in target_state.items():
        if not torch.equal(target_tensor, expected_target_state[target_key]):
            raise ValueError(
                "Inflated target tensor differs from deterministic "
                f"source+config+seed reconstruction: {target_key!r}."
            )
    copied_raw = tensor_mapping.get("copied")
    initialized_raw = tensor_mapping.get("initialized")
    if not isinstance(copied_raw, list) or not isinstance(initialized_raw, list):
        raise ValueError("Inflation report tensor lists are missing.")
    if (
        tensor_mapping.get("copied_count") != len(copied_raw)
        or tensor_mapping.get("initialized_count") != len(initialized_raw)
        or tensor_mapping.get("all_copied_tensors_bitwise_equal") is not True
    ):
        raise ValueError("Inflation report tensor counts or equality flag differ.")
    copied_by_source = {
        item.get("source"): item
        for item in copied_raw
        if isinstance(item, Mapping)
    }
    if len(copied_by_source) != len(source_state):
        raise ValueError("Copied tensor report is incomplete or duplicated.")
    copied_targets: set[str] = set()
    for source_key, source_tensor in source_state.items():
        expected_target_key = CAUSAL_SHARED_LIFT_MAPPING.get(
            source_key, source_key
        )
        item = copied_by_source.get(source_key)
        if item is None or item.get("target") != expected_target_key:
            raise ValueError(f"Copied tensor mapping differs for {source_key!r}.")
        target_tensor = target_state[expected_target_key]
        source_hash = tensor_sha256(source_tensor)
        target_hash = tensor_sha256(target_tensor)
        if (
            source_hash != target_hash
            or item.get("source_sha256") != source_hash
            or item.get("target_sha256") != target_hash
            or item.get("shape") != list(source_tensor.shape)
            or item.get("dtype") != str(source_tensor.dtype)
            or not torch.equal(source_tensor, target_tensor)
        ):
            raise ValueError(f"Copied tensor integrity failed for {source_key!r}.")
        copied_targets.add(expected_target_key)
    expected_initialized = set(target_state).difference(copied_targets)
    initialized_by_target = {
        item.get("target"): item
        for item in initialized_raw
        if isinstance(item, Mapping)
    }
    if set(initialized_by_target) != expected_initialized:
        raise ValueError("Initialized tensor report does not cover target state.")
    for target_key in sorted(expected_initialized):
        tensor = target_state[target_key]
        item = initialized_by_target[target_key]
        if (
            item.get("sha256") != tensor_sha256(tensor)
            or item.get("shape") != list(tensor.shape)
            or item.get("dtype") != str(tensor.dtype)
        ):
            raise ValueError(
                f"Initialized tensor hash differs for {target_key!r}."
            )
        if target_key == "lift.geometry.weight":
            expected_strategy = "zero_target_only_input_contribution"
            valid = not bool(torch.count_nonzero(tensor))
        elif target_key.endswith(".lateral.input_factor"):
            expected_strategy = (
                "deterministic_nonzero_high_pass_input_factor"
            )
            valid = bool(torch.count_nonzero(tensor))
        elif target_key.endswith(".lateral.output_factor"):
            expected_strategy = "zero_high_pass_output_factor"
            valid = not bool(torch.count_nonzero(tensor))
        else:
            raise ValueError(f"Unexpected initialized tensor {target_key!r}.")
        if item.get("strategy") != expected_strategy or not valid:
            raise ValueError(
                f"Initialized tensor strategy failed for {target_key!r}."
            )
    architecture = inspect_causal_target_architecture(model)
    if not architecture["passed"]:
        raise ValueError("Inflated target architecture is not structurally causal.")
    verification_inputs = _verification_source_input(
        source_spec,
        seed=int(expected_spec["inflation_seed"]) + 100_003,
    )
    expected_verification = {
        "restriction": verify_causal_models_on_input(
            source_model,
            model,
            verification_inputs,
            seed=int(expected_spec["inflation_seed"]) + 200_003,
        ),
        "future_invariance": verify_target_future_invariance(
            model,
            verification_inputs,
            seed=int(expected_spec["inflation_seed"]) + 300_003,
        ),
        "architecture": architecture,
        "uses_labels": False,
        "passed": True,
    }
    if report.get("verification") != expected_verification:
        raise ValueError(
            "Inflation report verification evidence differs from "
            "independent recomputation."
        )
    return {
        "source_checkpoint_sha256": expected_spec[
            "source_checkpoint_sha256"
        ],
        "target_config_sha256": expected_spec["target_config_sha256"],
        "copied_tensor_count": len(copied_targets),
        "initialized_tensor_count": len(expected_initialized),
        "all_copied_tensors_bitwise_equal": True,
        "all_target_tensors_match_deterministic_reconstruction": True,
        "all_tensor_hashes_match_embedded_report": True,
        "all_source_metadata_matches_checkpoint": True,
        "embedded_verification_matches_recomputation": True,
        "initial_lateral_output_is_exact_zero": True,
        "architecture": architecture,
        "passed": True,
    }


def inflate_causal_checkpoint_payload(
    source_checkpoint: Mapping[str, Any],
    target_config: Mapping[str, Any],
    *,
    seed: int = 20260726,
    source_checkpoint_sha256: str,
    target_config_sha256: str,
) -> tuple[
    CausalAxisFactorized2DOperator,
    dict[str, Any],
    dict[str, Any],
]:
    """Inflate an in-memory causal source checkpoint and verify it."""

    source_model, source_spec, source_state = load_causal_source_checkpoint(
        source_checkpoint
    )
    target_spec = _validated_target_spec(
        target_config,
        source_spec,
        source_checkpoint_sha256=source_checkpoint_sha256,
        target_config_sha256=target_config_sha256,
        inflation_seed=seed,
    )
    target_model = _construct_target(target_spec, adapter_seed=seed)
    with torch.no_grad():
        target_model.lift.geometry.weight.zero_()
        for layer_index, block in enumerate(target_model.blocks):
            block.lateral.reset_zero_residual(seed + layer_index)
    copied, initialized = _copy_source_state(target_model, source_state)
    verification_inputs = _verification_source_input(
        source_spec, seed=seed + 100_003
    )
    restriction = verify_causal_models_on_input(
        source_model,
        target_model,
        verification_inputs,
        seed=seed + 200_003,
    )
    causality = verify_target_future_invariance(
        target_model,
        verification_inputs,
        seed=seed + 300_003,
    )
    architecture = inspect_causal_target_architecture(target_model)
    if not (
        restriction["passed"]
        and causality["passed"]
        and architecture["passed"]
    ):
        raise RuntimeError("Inflated causal target failed its scientific gates.")
    parameter_count = sum(
        parameter.numel() for parameter in target_model.parameters()
    )
    expected_parameter_count = target_config.get("expected_parameter_count")
    if (
        expected_parameter_count is not None
        and parameter_count != int(expected_parameter_count)
    ):
        raise ValueError(
            "Causal target parameter count differs from the frozen config."
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P5",
        "operation": (
            "causal_restriction_preserving_1d_to_2d_checkpoint_inflation"
        ),
        "pilot_separation": {
            "existing_rp_ffno_family": "axis_factorized_2d",
            "existing_rp_ffno_temporal_family": "spectral_noncausal",
            "this_family": "causal_axis_factorized_2d",
            "this_temporal_family": "causal_dilated_convolution",
            "shares_noncausal_pilot_weights": False,
        },
        "deterministic_seed": int(seed),
        "source": {
            **source_spec,
            "channel_names": list(source_spec["channel_names"]),
            "temporal_dilations": list(source_spec["temporal_dilations"]),
            "checkpoint_sha256": target_spec["source_checkpoint_sha256"],
            "selected_epoch": source_checkpoint.get("epoch"),
            "validation_objective": source_checkpoint.get("best_validation"),
        },
        "target": {
            **target_spec,
            "source_channel_names": list(target_spec["source_channel_names"]),
            "new_channel_names": list(target_spec["new_channel_names"]),
            "channel_names": list(target_spec["channel_names"]),
            "temporal_dilations": list(target_spec["temporal_dilations"]),
            "config_sha256": target_spec["target_config_sha256"],
            "parameter_count": parameter_count,
        },
        "tensor_mapping": {
            "copied_count": len(copied),
            "initialized_count": len(initialized),
            "copied": copied,
            "initialized": initialized,
            "all_copied_tensors_bitwise_equal": True,
        },
        "verification": {
            "restriction": restriction,
            "future_invariance": causality,
            "architecture": architecture,
            "uses_labels": False,
            "passed": True,
        },
        "passed": True,
    }
    serializable_spec = {
        key: (
            list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in target_spec.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P5",
        "experiment": "causal_source_to_causal_target_inflation_v1",
        "model_family": "causal_axis_factorized_2d",
        "model": target_model.state_dict(),
        "model_config": serializable_spec,
        "channel_names": target_spec["channel_names"],
        "source_channel_names": target_spec["source_channel_names"],
        "new_channel_names": target_spec["new_channel_names"],
        "source_checkpoint_epoch": source_checkpoint.get("epoch"),
        "source_validation_objective": source_checkpoint.get(
            "best_validation"
        ),
        "normalization": source_checkpoint.get("normalization"),
        "inflation_report": report,
    }
    integrity = verify_causal_checkpoint_integrity(
        source_checkpoint,
        payload,
        target_config,
        source_checkpoint_sha256=source_checkpoint_sha256,
        target_config_sha256=target_config_sha256,
    )
    report["integrity"] = integrity
    return target_model, payload, report


def load_inflated_causal_target(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CausalAxisFactorized2DOperator, dict[str, Any]]:
    """Strictly reconstruct a saved causal target checkpoint."""

    payload = torch.load(
        checkpoint_path.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("model_family")
        != "causal_axis_factorized_2d"
    ):
        raise ValueError("Inflated causal target checkpoint schema is invalid.")
    spec = payload.get("model_config")
    state = payload.get("model")
    if not isinstance(spec, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Causal target checkpoint lacks config or model state.")
    if (
        spec.get("family") != "causal_axis_factorized_2d"
        or spec.get("temporal_family") != "causal_dilated_convolution"
        or spec.get("causal") is not True
        or spec.get("structurally_causal") is not True
    ):
        raise ValueError("Causal target model-family metadata is invalid.")
    model = _construct_target(spec)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Causal target model state is incompatible.") from error
    if not inspect_causal_target_architecture(model)["passed"]:
        raise ValueError("Loaded target architecture is not structurally causal.")
    model.to(device)
    return model, payload


def inflate_causal_checkpoint_file(
    source_path: Path,
    target_config_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    seed: int = 20260726,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Inflate, verify, and atomically save the causal target artifacts."""

    source_path = source_path.resolve()
    target_config_path = target_config_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    for path, label in (
        (source_path, "causal source checkpoint"),
        (target_config_path, "causal target config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not dry_run and not overwrite:
        for path in (output_path, report_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing artifact: {path}"
                )
    source_checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )
    config = load_causal_target_config(target_config_path)
    source_sha = sha256_file(source_path)
    config_sha = config["_config_sha256"]
    _, payload, report = inflate_causal_checkpoint_payload(
        source_checkpoint,
        config,
        seed=seed,
        source_checkpoint_sha256=source_sha,
        target_config_sha256=config_sha,
    )
    project_root = _find_project_root(target_config_path)
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
