"""Aggregate the four frozen P5 RP-FFNO pilot runs without test/OOD data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cdcureno.evaluation.p5_pilot import (
    decide_pilot_gate,
    load_pilot_run,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESTRICTION_GATE = (
    ROOT / "outputs" / "tables" / "p5_restriction_gate.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "tables" / "p5_rp_ffno_pilot_gate.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _run_evidence_summary(evidence: Any) -> dict[str, Any]:
    metrics = evidence.metrics
    return {
        "run_id": metrics["run_id"],
        "method": metrics["method"],
        "label_budget": metrics["label_budget"],
        "seed": metrics["seed"],
        "git_sha": metrics["git_sha"],
        "implementation_sha256": metrics["implementation_sha256"],
        "metrics_sha256": sha256_file(evidence.run_dir / "metrics.json"),
        "config_resolved_sha256": sha256_file(
            evidence.run_dir / "config_resolved.json"
        ),
        "selection": metrics["selection"],
        "training": metrics["training"],
        "model": metrics["model"],
        "initialization": metrics["initialization"],
        "parameter_groups": metrics["parameter_groups"],
        "loss_weights": metrics["loss_weights"],
        "input_checksums": metrics["input_checksums"],
        "target_label_access_audit": metrics[
            "target_label_access_audit"
        ],
        "restriction_validation": metrics["restriction_validation"],
        "validation_metrics_recomputed": evidence.recomputed_summary,
        "checkpoints": metrics["checkpoints"],
        "auditable_artifacts": metrics["auditable_artifacts"],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        ("scratch_ffno", 8): args.scratch_8,
        ("restriction_transfer_ffno", 8): args.transfer_8,
        ("scratch_ffno", 16): args.scratch_16,
        ("restriction_transfer_ffno", 16): args.transfer_16,
    }
    runs = {
        key: load_pilot_run(
            ROOT,
            path,
            expected_method=key[0],
            expected_budget=key[1],
        )
        for key, path in paths.items()
    }
    restriction_gate = _read_json(args.restriction_gate)
    report = decide_pilot_gate(
        runs, restriction_inflation_gate=restriction_gate
    )
    report["schema_version"] = 2
    report["restriction_inflation_gate"] = {
        "path": args.restriction_gate.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(args.restriction_gate),
        "passed": restriction_gate.get("passed") is True,
    }
    report["run_directories"] = {
        f"{method}_budget_{budget}": evidence.run_dir.relative_to(
            ROOT
        ).as_posix()
        for (method, budget), evidence in runs.items()
    }
    report["run_evidence"] = {
        f"{method}_budget_{budget}": _run_evidence_summary(evidence)
        for (method, budget), evidence in runs.items()
    }
    if not args.dry_run:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing pilot gate: {args.output}"
            )
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
            "Independently reproduce validation artifacts and apply the "
            "pre-registered one-seed P5 RP-FFNO pilot gate."
        )
    )
    parser.add_argument("--scratch-8", type=Path, required=True)
    parser.add_argument("--transfer-8", type=Path, required=True)
    parser.add_argument("--scratch-16", type=Path, required=True)
    parser.add_argument("--transfer-16", type=Path, required=True)
    parser.add_argument(
        "--restriction-gate",
        type=Path,
        default=DEFAULT_RESTRICTION_GATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing aggregate for the same quartet.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    report = aggregate(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
