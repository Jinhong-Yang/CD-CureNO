"""Independently recompute tracked P3 summaries and write the phase gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(observed: float, recorded: float) -> bool:
    return bool(np.isclose(observed, recorded, rtol=1.0e-10, atol=1.0e-12))


def main() -> None:
    public_summary = _load_json(
        TABLES / "p3_public_case1_validation.json"
    )
    public_cases = pd.read_csv(TABLES / "p3_public_case1_cases.csv")
    public_recomputed = {
        "case_count": int(len(public_cases)),
        "temperature_field_relative_l2_mean": float(
            public_cases["temperature_relative_l2"].mean()
        ),
        "temperature_field_absolute_error_K_max": float(
            public_cases["temperature_max_abs_K"].max()
        ),
        "alpha_field_relative_l2_mean": float(
            public_cases["alpha_relative_l2"].mean()
        ),
        "alpha_bound_violation_count": int(
            public_cases["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            public_cases["alpha_monotonicity_violation_count"].sum()
        ),
        "maximum_abs_energy_residual_W_per_m3": float(
            public_cases["maximum_abs_energy_residual_W_m3"].max()
        ),
        "maximum_relative_global_energy_residual": float(
            public_cases["maximum_relative_global_energy_residual"].max()
        ),
    }
    public_match = {
        key: (
            public_recomputed[key] == public_summary["aggregate"][key]
            if isinstance(public_recomputed[key], int)
            else _close(
                public_recomputed[key], public_summary["aggregate"][key]
            )
        )
        for key in public_recomputed
    }

    source_summary = _load_json(
        TABLES / "p3_source_pretraining_v4_seed0.json"
    )
    source_cases = pd.read_csv(
        TABLES / "p3_source_pretraining_v4_cases.csv"
    )
    heldout = source_cases[
        source_cases["split"].isin(["smart_cure", "three_hold"])
    ]
    source_recomputed = {
        "case_count": int(len(heldout)),
        "temperature_relative_l2_mean": float(
            heldout["temperature_relative_l2"].mean()
        ),
        "temperature_linf_K_max": float(
            heldout["temperature_linf_K"].max()
        ),
        "alpha_relative_l2_mean": float(
            heldout["alpha_relative_l2"].mean()
        ),
        "alpha_bound_violation_count": int(
            heldout["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            heldout["alpha_monotonicity_violation_count"].sum()
        ),
    }
    source_match = {
        key: (
            source_recomputed[key] == source_summary["held_out_combined"][key]
            if isinstance(source_recomputed[key], int)
            else _close(
                source_recomputed[key],
                source_summary["held_out_combined"][key],
            )
        )
        for key in source_recomputed
    }
    family_match = {}
    for family in ("smart_cure", "three_hold"):
        frame = source_cases[source_cases["split"] == family]
        recorded = source_summary["split_metrics"][family]
        family_match[family] = bool(
            len(frame) == recorded["case_count"]
            and _close(
                float(frame["temperature_relative_l2"].mean()),
                recorded["temperature_relative_l2_mean"],
            )
            and _close(
                float(frame["alpha_relative_l2"].mean()),
                recorded["alpha_relative_l2_mean"],
            )
        )

    manifest = _load_json(ROOT / "splits" / "p3_source_1d_v3.json")
    array_path = ROOT / manifest["array_artifact"]
    splits = manifest["splits"]
    split_sets = [
        set(splits["train"]),
        set(splits["validation"]),
        set(splits["in_family_test"]),
        *[
            set(values)
            for values in splits["held_out_family_test"].values()
        ],
    ]
    no_split_overlap = all(
        not split_sets[left].intersection(split_sets[right])
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    )
    complete_partition = len(set().union(*split_sets)) == 400
    failed_runs = {
        version: _load_json(
            TABLES / f"p3_source_pretraining_{version}_failed.json"
        )
        for version in ("v1", "v2", "v3")
    }
    convergence = _load_json(TABLES / "p3_solver_convergence.json")
    generation = _load_json(TABLES / "p3_source_1d_v3_generation.json")
    checks = {
        "public_case_rows_match_summary": all(public_match.values()),
        "public_case_acceptance_passed": bool(public_summary["passed"]),
        "mesh_time_convergence_passed": bool(convergence["passed"]),
        "source_generation_energy_passed": bool(generation["passed"]),
        "source_array_checksum_matches": (
            _sha256(array_path) == manifest["array_sha256"]
        ),
        "source_splits_do_not_overlap": no_split_overlap,
        "source_splits_partition_all_cases": complete_partition,
        "held_out_labels_excluded_from_training": (
            not source_summary["training"]["held_out_labels_used"]
        ),
        "source_case_rows_match_summary": all(source_match.values()),
        "source_family_rows_match_summary": all(family_match.values()),
        "source_pretraining_acceptance_passed": bool(source_summary["passed"]),
        "failed_v1_v2_v3_preserved": all(
            not payload["passed"] for payload in failed_runs.values()
        ),
    }
    gate = {
        "schema_version": 1,
        "phase": "P3",
        "solver": {
            "public_case_validation": public_summary["aggregate"],
            "convergence_reference": convergence["reference"],
        },
        "source_data": {
            "dataset_id": generation["dataset_id"],
            "case_count": generation["case_count"],
            "array_sha256": generation["array_sha256"],
            "maximum_abs_energy_residual_W_m3": generation[
                "maximum_abs_energy_residual_W_m3"
            ],
            "maximum_relative_global_energy_residual": generation[
                "maximum_relative_global_energy_residual"
            ],
        },
        "source_pretraining": {
            "model": source_summary["model"],
            "training": source_summary["training"],
            "held_out_combined": source_summary["held_out_combined"],
            "split_metrics": source_summary["split_metrics"],
        },
        "failed_development_runs": {
            version: {
                "held_out_temperature_relative_l2_mean": payload[
                    "held_out_combined"
                ]["temperature_relative_l2_mean"],
                "held_out_temperature_linf_K_max": payload[
                    "held_out_combined"
                ]["temperature_linf_K_max"],
                "failed_checks": [
                    key
                    for key, value in payload["acceptance_checks"].items()
                    if not value
                ],
            }
            for version, payload in failed_runs.items()
        },
        "independent_recomputation": {
            "public_metric_matches": public_match,
            "source_metric_matches": source_match,
            "source_family_matches": family_match,
        },
        "gate_checks": checks,
        "passed": bool(all(checks.values())),
    }
    output = TABLES / "p3_gate_summary.json"
    output.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
