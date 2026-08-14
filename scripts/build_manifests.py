"""Build deterministic complete-case split manifests for the ResFNO Case1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from cdcureno.data.splits import validate_disjoint_complete_case_split


SEED = 20260723


def write_manifest(path: Path, payload: dict) -> None:
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(canonical, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Create deterministic whole-case split manifests."
    )
    result.add_argument(
        "--data",
        type=Path,
        default=Path("external/ResFNO/data/Case1.mat"),
    )
    result.add_argument("--output-dir", type=Path, default=Path("splits"))
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_path = (project_root / args.data).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    data = sio.loadmat(data_path)
    n_cases = int(data["dataT"].shape[0])
    case_ids = np.arange(n_cases, dtype=int)

    exact = {
        "train": case_ids[:50].tolist(),
        "validation": [],
        "test": case_ids[50:].tolist(),
    }
    permutation = np.random.default_rng(SEED).permutation(case_ids)
    corrected = {
        "train": sorted(permutation[:50].tolist()),
        "validation": sorted(permutation[50:75].tolist()),
        "test": sorted(permutation[75:].tolist()),
    }
    corrected_matched = {
        "train": case_ids[:50].tolist(),
        "validation": case_ids[50:75].tolist(),
        "test": case_ids[75:].tolist(),
    }
    validate_disjoint_complete_case_split(exact, case_ids.tolist())
    validate_disjoint_complete_case_split(corrected, case_ids.tolist())
    validate_disjoint_complete_case_split(corrected_matched, case_ids.tolist())
    common_holdout = sorted(set(exact["test"]) & set(corrected["test"]))
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()

    manifests = {
        "legacy_case1_exact_v1.json": {
            "version": "legacy_case1_exact_v1",
            "policy": "faithful first-50 training and remaining-150 test split",
            "seed": None,
            "data_path": "external/ResFNO/data/Case1.mat",
            "data_sha256": data_sha256,
            "splits": exact,
        },
        "legacy_case1_corrected_v1.json": {
            "version": "legacy_case1_corrected_v1",
            "policy": "seeded whole-case 50/25/125 train/validation/test split",
            "seed": SEED,
            "data_path": "external/ResFNO/data/Case1.mat",
            "data_sha256": data_sha256,
            "splits": corrected,
        },
        "legacy_case1_corrected_matched_v1.json": {
            "version": "legacy_case1_corrected_matched_v1",
            "policy": (
                "controlled correction: legacy cases 0-49 train, cases 50-74 "
                "validation, cases 75-199 test"
            ),
            "seed": None,
            "data_path": "external/ResFNO/data/Case1.mat",
            "data_sha256": data_sha256,
            "splits": corrected_matched,
        },
        "p1_matched_holdout_v1.json": {
            "version": "p1_matched_holdout_v1",
            "policy": (
                "evaluation-only cases 75-199, excluded from exact/corrected-matched "
                "training and corrected-matched validation"
            ),
            "seed": None,
            "data_path": "external/ResFNO/data/Case1.mat",
            "data_sha256": data_sha256,
            "test": corrected_matched["test"],
            "source_manifests": [
                "legacy_case1_exact_v1.json",
                "legacy_case1_corrected_matched_v1.json",
            ],
        },
        "p1_common_holdout_v1.json": {
            "version": "p1_common_holdout_v1",
            "policy": (
                "evaluation-only intersection of exact and corrected test cases; "
                "excluded from both methods' training and corrected validation"
            ),
            "seed": SEED,
            "data_path": "external/ResFNO/data/Case1.mat",
            "data_sha256": data_sha256,
            "test": common_holdout,
            "source_manifests": [
                "legacy_case1_exact_v1.json",
                "legacy_case1_corrected_v1.json",
            ],
        },
    }
    if args.dry_run:
        print(json.dumps({name: value["policy"] for name, value in manifests.items()}, indent=2))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        write_manifest(output_dir / name, payload)
        print(output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
