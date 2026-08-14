"""Verify the saved causal transfer on a frozen P3 validation input."""

from __future__ import annotations

import argparse
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
    load_source_normalization_contract,
)
from cdcureno.models.causal_checkpoint_inflation import (
    load_causal_source_checkpoint,
    load_causal_target_config,
    load_inflated_causal_target,
    sha256_file,
    verify_causal_checkpoint_integrity,
    verify_causal_models_on_input,
    verify_target_future_invariance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs"
    / "runs"
    / "p3-source-causal-v1-seed0-fix1"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_TARGET = (
    ROOT / "outputs" / "checkpoints" / "p5_causal_inflated_source.pt"
)
DEFAULT_CONFIG = ROOT / "configs" / "model" / "cdcureno_causal_2d.yaml"
DEFAULT_SOURCE_DATA = ROOT / "data" / "processed" / "p3_source_1d_v3.npz"
DEFAULT_SOURCE_SPLIT = ROOT / "splits" / "p3_source_1d_v3.json"
DEFAULT_P4_PLAN = ROOT / "splits" / "p4_2d_core_v1_plan.json"
DEFAULT_P4_MANIFEST = ROOT / "splits" / "p4_2d_core_v1.json"
DEFAULT_P4_ID_SPLIT = ROOT / "splits" / "2d_id_v1.json"
DEFAULT_P4_ARRAY_ROOT = ROOT / "data" / "processed" / "p4_2d_core_v1"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "tables" / "p5_causal_restriction_gate.json"
)
FIELD_TOLERANCES = {
    "temperature": 1.0e-6,
    "temperature_residual": 1.0e-6,
    "alpha": 1.0e-6,
    "cure_rate": 1.0e-5,
}
RVS_TOLERANCE = 1.0e-6
LIS_TOLERANCE = 1.0e-7


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _verified_exogenous_array(
    root: Path,
    manifest: dict[str, Any],
    name: str,
) -> np.ndarray:
    path = root / f"{name}.npy"
    if sha256_file(path) != manifest["array_sha256"][name]:
        raise ValueError(f"P4 exogenous array hash mismatch: {name}")
    return np.load(path, allow_pickle=False)


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
        lis = float(
            torch.sqrt(torch.mean(lateral_variance))
            / torch.std(actual, unbiased=False).clamp_min(
                torch.finfo(torch.float64).eps
            )
        )
        lateral_range = float(
            torch.max(
                torch.amax(actual, dim=3) - torch.amin(actual, dim=3)
            )
        )
        checks = {
            "maximum_absolute_mismatch": maximum <= tolerance,
            "maximum_lateral_range": lateral_range <= tolerance,
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
        field_passed = all(bool(value) for value in checks.values())
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


def _verify_p4_f0(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    *,
    source_checkpoint_path: Path,
    plan_path: Path,
    manifest_path: Path,
    id_split_path: Path,
    array_root: Path,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    manifest = _read_json(manifest_path)
    id_split = _read_json(id_split_path)
    if plan.get("plan_sha256") != manifest.get("plan_sha256"):
        raise ValueError("P4 manifest and plan semantic hashes differ.")
    if (
        id_split.get("dataset_plan_sha256") != plan.get("plan_sha256")
        or id_split.get("source_manifest_sha256")
        != sha256_file(manifest_path)
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
    normalization = load_source_normalization_contract(
        source_checkpoint_path
    )
    inputs = build_target_2d_input(
        definition,
        time,
        z,
        x,
        mask,
        normalization,
        geometry,
    )
    expected_shape = (112, 50, 40, 20)
    if inputs.shape != expected_shape:
        raise ValueError(
            f"Canonical P4 F0 input shape is {inputs.shape}, "
            f"expected {expected_shape}."
        )
    suffix = inputs[..., -len(TARGET_ADAPTER_CHANNELS) :]
    suffix_nonzero_count = int(np.count_nonzero(suffix))
    suffix_exact_zero = bool(suffix_nonzero_count == 0)
    if not suffix_exact_zero:
        raise ValueError("P4 F0 target-only suffix is not exact zero.")
    shared_field = inputs[..., :14]
    shared_reference = np.broadcast_to(
        shared_field[:, :, :1, :], shared_field.shape
    )
    shared_x_invariant = bool(np.array_equal(shared_field, shared_reference))
    if not shared_x_invariant:
        raise ValueError("P4 F0 shared source channels vary over x.")
    source_input = torch.from_numpy(shared_field[:, :, 0, :]).unsqueeze(0)
    target_input = torch.from_numpy(inputs).unsqueeze(0)
    with torch.no_grad():
        source_output = source_model(source_input)
        target_output = target_model(target_input)
    fields, passed = _field_metrics(source_output, target_output)
    return {
        "source": "actual_pre_label_p4_validation_f0_input",
        "case_id": int(definition["case_id"]),
        "case_key": definition["case_key"],
        "difficulty_family": definition["difficulty_family"],
        "source_input_shape": list(source_input.shape),
        "target_input_shape": list(target_input.shape),
        "target_only_channel_names": list(TARGET_ADAPTER_CHANNELS),
        "target_only_nonzero_count": suffix_nonzero_count,
        "target_only_suffix_exact_zero": suffix_exact_zero,
        "shared_source_channels_exactly_x_invariant": shared_x_invariant,
        "fields": fields,
        "passed": bool(passed and suffix_exact_zero and shared_x_invariant),
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
        raise FileNotFoundError(
            f"Missing required causal-transfer artifacts: {missing}"
        )
    if not args.p4_array_root.is_dir():
        raise FileNotFoundError(
            f"Missing P4 array root: {args.p4_array_root}"
        )
    source_sha = sha256_file(args.source_checkpoint)
    target_sha = sha256_file(args.target_checkpoint)
    config_sha = sha256_file(args.target_config)
    source_checkpoint = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    target_payload = torch.load(
        args.target_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    target_config = load_causal_target_config(args.target_config)
    integrity = verify_causal_checkpoint_integrity(
        source_checkpoint,
        target_payload,
        target_config,
        source_checkpoint_sha256=source_sha,
        target_config_sha256=config_sha,
    )
    source_model, source_spec, _ = load_causal_source_checkpoint(
        source_checkpoint
    )
    target_model, _ = load_inflated_causal_target(args.target_checkpoint)
    prepared = prepare_source_1d(
        args.source_data,
        args.source_split,
        time_stride=2,
    )
    dataset = prepared.dataset("validation")
    source_input, _, _, case_id, family_id = dataset[0]
    source_input = source_input.unsqueeze(0)
    restriction = verify_causal_models_on_input(
        source_model,
        target_model,
        source_input,
        seed=20260726 + int(case_id),
        nx_values=(1, 2, 7, 40),
    )
    causality = verify_target_future_invariance(
        target_model,
        source_input,
        seed=20260726 + int(case_id),
        nx=7,
    )
    p4_f0 = _verify_p4_f0(
        source_model,
        target_model,
        source_checkpoint_path=args.source_checkpoint,
        plan_path=args.p4_plan,
        manifest_path=args.p4_manifest,
        id_split_path=args.p4_id_split,
        array_root=args.p4_array_root,
    )
    embedded = target_payload.get("inflation_report", {})
    embedded_verification = embedded.get("verification", {})
    embedded_passed = bool(
        isinstance(embedded_verification, dict)
        and embedded_verification.get("passed") is True
    )
    passed = bool(
        integrity["passed"]
        and restriction["passed"]
        and causality["passed"]
        and p4_f0["passed"]
        and embedded_passed
    )
    report = {
        "schema_version": 1,
        "phase": "P5",
        "gate": "causal_source_to_causal_target_restriction",
        "dry_run": bool(args.dry_run),
        "role": "final_causal_transfer_path",
        "separate_from_noncausal_rp_ffno_pilot": True,
        "p4_fine_temperature_or_cure_labels_loaded": False,
        "restriction_comparisons_use_labels": False,
        "source_validation_labels_loaded_by_preparation_but_not_used": True,
        "source_checkpoint_sha256": source_sha,
        "target_checkpoint_sha256": target_sha,
        "target_config_sha256": config_sha,
        "source_data_sha256": sha256_file(args.source_data),
        "source_split_sha256": sha256_file(args.source_split),
        "p4_plan_file_sha256": sha256_file(args.p4_plan),
        "p4_manifest_sha256": sha256_file(args.p4_manifest),
        "p4_id_split_sha256": sha256_file(args.p4_id_split),
        "source_family": source_spec["family"],
        "target_family": target_model.family,
        "temporal_family": target_model.temporal_family,
        "source_validation_case": {
            "case_id": int(case_id),
            "family_id": int(family_id),
            "input_shape": list(source_input.shape),
        },
        "checkpoint_integrity": integrity,
        "actual_validation_restriction": restriction,
        "actual_validation_future_invariance": causality,
        "actual_p4_validation_f0": p4_f0,
        "embedded_inflation_verification_passed": embedded_passed,
        "passed": passed,
    }
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute source/config/tensor integrity, Nx=1/2/7/40 "
            "restriction, and all-channel future-invariance checks for the "
            "saved causal P5 target."
        )
    )
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE
    )
    parser.add_argument(
        "--target-checkpoint", type=Path, default=DEFAULT_TARGET
    )
    parser.add_argument("--target-config", type=Path, default=DEFAULT_CONFIG)
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
    report = verify(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
