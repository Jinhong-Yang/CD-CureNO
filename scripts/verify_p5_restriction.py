"""Verify checkpoint restriction on real P3 validation and P4 F0 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cdcureno.data.source_1d import prepare_source_1d
from cdcureno.data.target_2d import (
    TARGET_ADAPTER_CHANNELS,
    build_target_2d_input,
    lift_homogeneous_source_input,
    load_source_normalization_contract,
)
from cdcureno.models.joint_operators import FactorizedFNO
from cdcureno.models.checkpoint_inflation import (
    load_target_config,
    verify_inflated_checkpoint_integrity,
)
from cdcureno.models.target_operators import AxisFactorized2DOperator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CHECKPOINT = (
    ROOT / "outputs" / "runs" / "p3-source-factorized-v4-seed0" / "best.pt"
)
DEFAULT_TARGET_CHECKPOINT = (
    ROOT / "outputs" / "checkpoints" / "p5_inflated_source.pt"
)
DEFAULT_TARGET_CONFIG = ROOT / "configs" / "model" / "cdcureno_2d.yaml"
DEFAULT_SOURCE_DATA = ROOT / "data" / "processed" / "p3_source_1d_v3.npz"
DEFAULT_SOURCE_SPLIT = ROOT / "splits" / "p3_source_1d_v3.json"
DEFAULT_P4_PLAN = ROOT / "splits" / "p4_2d_core_v1_plan.json"
DEFAULT_P4_MANIFEST = ROOT / "splits" / "p4_2d_core_v1.json"
DEFAULT_P4_ID_SPLIT = ROOT / "splits" / "2d_id_v1.json"
DEFAULT_P4_ARRAY_ROOT = ROOT / "data" / "processed" / "p4_2d_core_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "tables" / "p5_restriction_gate.json"

FIELD_TOLERANCES = {
    "temperature": 1.0e-6,
    "temperature_residual": 1.0e-6,
    "alpha": 1.0e-6,
    "cure_rate": 1.0e-5,
}
RVS_TOLERANCE = 1.0e-6
LIS_TOLERANCE = 1.0e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _source_model(
    checkpoint: dict[str, Any],
) -> FactorizedFNO:
    state = checkpoint["model"]
    lift = state["lift.0.weight"]
    depth = len(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("blocks.")
        }
    )
    model = FactorizedFNO(
        input_channels=int(lift.shape[1]),
        width=int(lift.shape[0]),
        depth=depth,
        modes_time=int(state["blocks.0.temporal.weight"].shape[-1]),
        modes_space=int(state["blocks.0.spatial.weight"].shape[-1]),
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


def _target_model(
    checkpoint: dict[str, Any],
) -> AxisFactorized2DOperator:
    config = checkpoint["model_config"]
    model = AxisFactorized2DOperator(
        source_channel_names=config["source_channel_names"],
        new_channel_names=config["new_channel_names"],
        width=int(config["width"]),
        depth=int(config["depth"]),
        modes_time=int(config["modes_time"]),
        modes_z=int(config["modes_z"]),
        modes_x=int(config["modes_x"]),
        lateral_rank=int(config["lateral_rank"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def _field_metrics(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], bool]:
    fields: dict[str, Any] = {}
    passed = True
    for name, tolerance in FIELD_TOLERANCES.items():
        expected = source[name].double()
        actual = target[name].double()
        if actual.ndim != expected.ndim + 1:
            raise ValueError(f"Target field {name} has the wrong rank.")
        reference = expected.unsqueeze(3).expand_as(actual)
        difference = actual - reference
        maximum = float(torch.max(torch.abs(difference)))
        relative = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(
                torch.finfo(torch.float64).eps
            )
        )
        restricted = torch.mean(actual, dim=3)
        rvs = float(
            torch.linalg.vector_norm(restricted - expected)
            / torch.linalg.vector_norm(expected).clamp_min(
                torch.finfo(torch.float64).eps
            )
        )
        lateral_variance = torch.var(actual, dim=3, unbiased=False)
        lateral_std = torch.sqrt(torch.mean(lateral_variance))
        field_std = torch.std(actual, unbiased=False)
        lis = float(
            lateral_std
            / field_std.clamp_min(torch.finfo(torch.float64).eps)
        )
        lateral_range = float(
            torch.max(
                torch.amax(actual, dim=3) - torch.amin(actual, dim=3)
            )
        )
        checks = {
            "maximum_absolute_mismatch": maximum <= tolerance,
            "restriction_relative_l2": (
                True
                if name not in {"temperature", "alpha"}
                else rvs <= RVS_TOLERANCE
            ),
            "lateral_invariance": (
                True
                if name not in {"temperature", "alpha"}
                else lis <= LIS_TOLERANCE
            ),
        }
        field_passed = all(checks.values())
        passed &= field_passed
        fields[name] = {
            "maximum_absolute_mismatch": maximum,
            "relative_l2_mismatch": relative,
            "restriction_relative_l2": rvs,
            "lateral_invariance_score": lis,
            "maximum_lateral_range": lateral_range,
            "absolute_tolerance": tolerance,
            "rvs_tolerance": (
                RVS_TOLERANCE if name in {"temperature", "alpha"} else None
            ),
            "lis_tolerance": (
                LIS_TOLERANCE if name in {"temperature", "alpha"} else None
            ),
            "checks": checks,
            "passed": field_passed,
        }
    return fields, passed


def _verify_source_validation(
    source_model: FactorizedFNO,
    target_model: AxisFactorized2DOperator,
    source_data: Path,
    source_split: Path,
) -> dict[str, Any]:
    prepared = prepare_source_1d(
        source_data,
        source_split,
        time_stride=2,
    )
    dataset = prepared.dataset("validation")
    inputs, _, _, case_id, family_id = dataset[0]
    source_input = inputs.unsqueeze(0)
    per_nx: dict[str, Any] = {}
    passed = True
    with torch.no_grad():
        source_output = source_model(source_input)
        for nx in (1, 2, 7, 40):
            target_input = lift_homogeneous_source_input(
                source_input, nx
            )
            target_output = target_model(target_input)
            fields, nx_passed = _field_metrics(
                source_output, target_output
            )
            if nx == 1:
                bitwise = {
                    name: bool(
                        torch.equal(
                            target_output[name].squeeze(3),
                            source_output[name],
                        )
                    )
                    for name in FIELD_TOLERANCES
                }
                nx_passed &= all(bitwise.values())
            else:
                bitwise = {
                    name: False for name in FIELD_TOLERANCES
                }
            per_nx[str(nx)] = {
                "fields": fields,
                "nx_one_bitwise_equal": bitwise,
                "passed": nx_passed,
            }
            passed &= nx_passed
    return {
        "source": "actual_frozen_p3_validation_input",
        "case_id": int(case_id),
        "family_id": int(family_id),
        "input_shape": list(source_input.shape),
        "tested_nx": [1, 2, 7, 40],
        "per_nx": per_nx,
        "passed": passed,
    }


def _verified_exogenous_array(
    root: Path,
    manifest: dict[str, Any],
    name: str,
) -> np.ndarray:
    path = root / f"{name}.npy"
    if _sha256(path) != manifest["array_sha256"][name]:
        raise ValueError(f"P4 exogenous array hash mismatch: {name}")
    return np.load(path, allow_pickle=False)


def _verify_p4_f0(
    source_model: FactorizedFNO,
    target_model: AxisFactorized2DOperator,
    source_checkpoint: Path,
    plan_path: Path,
    manifest_path: Path,
    id_split_path: Path,
    array_root: Path,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    manifest = _read_json(manifest_path)
    id_split = _read_json(id_split_path)
    if plan.get("plan_sha256") != manifest.get("plan_sha256"):
        raise ValueError("P4 manifest and pre-label plan semantic hashes differ.")
    if (
        id_split.get("dataset_plan_sha256") != plan.get("plan_sha256")
        or id_split.get("source_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("P4 ID split is not bound to the canonical plan.")
    validation_ids = set(id_split["splits"]["validation"])
    candidates = [
        entry["definition"]
        for entry in plan["cases"]
        if entry["definition"]["case_id"] in validation_ids
        and entry["definition"]["difficulty_family"] == "F0"
    ]
    if not candidates:
        raise ValueError("P4 validation split contains no F0 anchor.")
    definition = candidates[0]
    resolved = plan["resolved_config"]["geometry"]
    geometry = {
        name: float(resolved[name])
        for name in (
            "width_m",
            "tool_thickness_m",
            "composite_thickness_m",
        )
    }
    time = _verified_exogenous_array(array_root, manifest, "time_s")
    z = _verified_exogenous_array(array_root, manifest, "z_m")
    x = _verified_exogenous_array(array_root, manifest, "x_m")
    mask = _verified_exogenous_array(
        array_root, manifest, "composite_mask"
    )
    normalization = load_source_normalization_contract(source_checkpoint)
    inputs = build_target_2d_input(
        definition,
        time,
        z,
        x,
        mask,
        normalization,
        geometry,
    )
    if np.count_nonzero(inputs[..., -len(TARGET_ADAPTER_CHANNELS) :]):
        raise ValueError("Selected P4 F0 input has nonzero target-only channels.")
    shared = torch.from_numpy(inputs[:, :, 0, :14]).unsqueeze(0)
    target_input = torch.from_numpy(inputs).unsqueeze(0)
    with torch.no_grad():
        source_output = source_model(shared)
        target_output = target_model(target_input)
    fields, passed = _field_metrics(source_output, target_output)
    return {
        "source": "actual_pre_label_p4_validation_f0_input",
        "case_id": int(definition["case_id"]),
        "case_key": definition["case_key"],
        "difficulty_family": definition["difficulty_family"],
        "source_input_shape": list(shared.shape),
        "target_input_shape": list(target_input.shape),
        "target_only_nonzero_count": 0,
        "fields": fields,
        "passed": passed,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        args.source_checkpoint,
        args.target_checkpoint,
        args.target_config,
        args.source_data,
        args.source_split,
        args.p4_plan,
        args.p4_manifest,
        args.p4_id_split,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required P5 artifacts: {missing}")
    if not args.p4_array_root.is_dir():
        raise FileNotFoundError(
            f"Missing P4 array root: {args.p4_array_root}"
        )
    source_sha256 = _sha256(args.source_checkpoint)
    target_config_sha256 = _sha256(args.target_config)
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    target_checkpoint = torch.load(
        args.target_checkpoint, map_location="cpu", weights_only=False
    )
    target_config = load_target_config(args.target_config)
    integrity = verify_inflated_checkpoint_integrity(
        source_checkpoint,
        target_checkpoint,
        target_config,
        source_checkpoint_sha256=source_sha256,
        target_config_sha256=target_config_sha256,
    )
    source_model = _source_model(source_checkpoint)
    target_model = _target_model(target_checkpoint)
    source_validation = _verify_source_validation(
        source_model,
        target_model,
        args.source_data,
        args.source_split,
    )
    p4_f0 = _verify_p4_f0(
        source_model,
        target_model,
        args.source_checkpoint,
        args.p4_plan,
        args.p4_manifest,
        args.p4_id_split,
        args.p4_array_root,
    )
    report = {
        "schema_version": 2,
        "phase": "P5",
        "gate": "canonical_inflation_and_source_to_target_restriction",
        "dry_run": bool(args.dry_run),
        "uses_target_temperature_or_cure_labels": False,
        "source_checkpoint_sha256": source_sha256,
        "target_checkpoint_sha256": _sha256(args.target_checkpoint),
        "target_config_sha256": target_config_sha256,
        "source_data_sha256": _sha256(args.source_data),
        "source_split_sha256": _sha256(args.source_split),
        "p4_plan_file_sha256": _sha256(args.p4_plan),
        "p4_manifest_sha256": _sha256(args.p4_manifest),
        "p4_id_split_sha256": _sha256(args.p4_id_split),
        "tolerances": {
            "maximum_absolute": FIELD_TOLERANCES,
            "temperature_alpha_rvs": RVS_TOLERANCE,
            "temperature_alpha_lis": LIS_TOLERANCE,
        },
        "inflation_integrity": integrity,
        "actual_p3_validation": source_validation,
        "actual_p4_validation_f0": p4_f0,
        "passed": bool(
            integrity["passed"]
            and source_validation["passed"]
            and p4_f0["passed"]
        ),
    }
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an inflated P5 checkpoint on actual frozen P3 validation "
            "and pre-label P4 F0 inputs without opening P4 field labels."
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT,
    )
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=DEFAULT_TARGET_CHECKPOINT,
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=DEFAULT_TARGET_CONFIG,
    )
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument(
        "--source-split", type=Path, default=DEFAULT_SOURCE_SPLIT
    )
    parser.add_argument("--p4-plan", type=Path, default=DEFAULT_P4_PLAN)
    parser.add_argument(
        "--p4-manifest", type=Path, default=DEFAULT_P4_MANIFEST
    )
    parser.add_argument(
        "--p4-id-split", type=Path, default=DEFAULT_P4_ID_SPLIT
    )
    parser.add_argument(
        "--p4-array-root", type=Path, default=DEFAULT_P4_ARRAY_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = verify(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
