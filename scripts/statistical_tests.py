"""Regenerate every frozen P6 statistical artifact.

The command reads one immutable per-case metric table.  It never reads model
predictions or target arrays.  All outputs are prepared in memory and then
written with atomic replacement; ``--dry-run`` performs validation and hash
generation without writing any file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from cdcureno.evaluation.p6_statistics import (
    BOOTSTRAP_DRAW_ORDER,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPUTE_COLUMNS,
    STATISTICS_SCHEMA_VERSION,
    analyze_p6_results,
    registered_confirmatory_runs,
)
from cdcureno.evaluation.p6_release import (
    COMPUTE_ACCOUNTING_COLUMNS,
    P6ReleasePaths,
    verify_p6_frozen_release,
)
from cdcureno.training.p6_roster import validate_p6_confirmatory_roster


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    ROOT / "src" / "cdcureno" / "evaluation" / "p6_statistics.py",
    ROOT / "src" / "cdcureno" / "training" / "p6_roster.py",
    Path(__file__).resolve(),
)


def _implementation_key(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Any) -> bytes:
    rendered = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return rendered.encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("CSV tables cannot contain NaN or infinity.")
        return format(numeric, ".17g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    return str(value)


def _table_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("A registered output table cannot be empty.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: _csv_value(row.get(key)) for key in fieldnames}
        )
    return stream.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array),
        allow_pickle=False,
    )
    return stream.getvalue()


def _deterministic_npz_bytes(
    arrays: dict[str, np.ndarray],
) -> bytes:
    """Create a compressed NPZ with fixed member order and timestamps."""

    required = {
        "seed_indices",
        "case_indices",
        "mean_relative_improvement",
        "case_median_relative_improvement",
        "peak_error_ratio",
        "final_alpha_ratio",
        "energy_residual_ratio",
    }
    if set(arrays) != required:
        raise ValueError(
            "Bootstrap NPZ members differ from the registered schema."
        )
    payload = {
        **arrays,
        "schema_version": np.asarray(
            STATISTICS_SCHEMA_VERSION, dtype=np.int64
        ),
        "bootstrap_replicates": np.asarray(
            BOOTSTRAP_REPLICATES, dtype=np.int64
        ),
        "bootstrap_seed": np.asarray(BOOTSTRAP_SEED, dtype=np.int64),
        "draw_order": np.asarray(BOOTSTRAP_DRAW_ORDER),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info,
                _npy_bytes(np.asarray(payload[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return stream.getvalue()


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(
        "P6 input must be Parquet, CSV, JSONL, or NDJSON."
    )


def _read_and_validate_roster(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P6 run roster must contain one JSON mapping.")
    validate_p6_confirmatory_roster(value)
    roster_cells = {
        (
            str(row["method"]),
            int(row["label_budget"]),
            int(row["seed"]),
        )
        for row in value["unique_training_runs"]
    }
    registered_cells = {
        (
            run.result_method,
            run.label_budget,
            run.seed,
        )
        for run in registered_confirmatory_runs()
    }
    if roster_cells != registered_cells or len(roster_cells) != 89:
        raise ValueError(
            "Frozen run roster differs from the registered statistical cells."
        )
    return value


def verify_execution_provenance(
    input_path: Path,
    plan_path: Path,
    roster_path: Path,
    *,
    release_paths: P6ReleasePaths | None = None,
) -> dict[str, Any]:
    """Verify the committed final release without opening held-out labels."""

    paths = (
        P6ReleasePaths.from_root(ROOT)
        if release_paths is None
        else release_paths
    )
    resolved_input = input_path.resolve()
    resolved_plan = plan_path.resolve()
    resolved_roster = roster_path.resolve()
    if resolved_input != paths.frozen_results.resolve():
        raise PermissionError(
            "Statistical input must be the release-bound frozen result table."
        )
    if resolved_plan != paths.statistical_plan.resolve():
        raise PermissionError(
            "Statistical plan must be the release-bound canonical plan."
        )
    if resolved_roster != paths.roster.resolve():
        raise PermissionError(
            "Run roster must be the release-bound canonical roster."
        )
    if tuple(COMPUTE_COLUMNS) != tuple(COMPUTE_ACCOUNTING_COLUMNS):
        raise RuntimeError(
            "Statistical compute columns differ from the release schema."
        )
    verification = verify_p6_frozen_release(
        paths,
        require_committed=True,
    )
    live_input = {
        "path": resolved_input.relative_to(
            paths.project_root.resolve()
        ).as_posix(),
        "sha256": _sha256_file(resolved_input),
        "bytes": resolved_input.stat().st_size,
    }
    frozen_record = verification.get("frozen_results")
    plan_record = verification.get("statistical_analysis_plan")
    roster_record = verification.get("confirmatory_roster")
    live_plan = {
        "path": resolved_plan.relative_to(
            paths.project_root.resolve()
        ).as_posix(),
        "sha256": _sha256_file(resolved_plan),
        "bytes": resolved_plan.stat().st_size,
    }
    live_roster = {
        "path": resolved_roster.relative_to(
            paths.project_root.resolve()
        ).as_posix(),
        "sha256": _sha256_file(resolved_roster),
        "bytes": resolved_roster.stat().st_size,
    }
    if (
        not isinstance(frozen_record, dict)
        or any(
            frozen_record.get(key) != value
            for key, value in live_input.items()
        )
        or verification.get("passed") is not True
        or verification.get("unique_training_run_count") != 89
        or verification.get("evaluation_artifact_count") != 445
        or verification.get("authority_files_tracked_and_head_clean")
        is not True
        or verification.get("held_out_label_arrays_opened") is not False
        or not isinstance(plan_record, dict)
        or any(
            plan_record.get(key) != value
            for key, value in live_plan.items()
        )
        or not isinstance(roster_record, dict)
        or any(
            roster_record.get(key) != value
            for key, value in live_roster.items()
        )
        or not isinstance(roster_record.get("roster_sha256"), str)
        or roster_record.get("unique_training_run_count") != 89
    ):
        raise PermissionError(
            "Frozen statistical input differs from the verified release."
        )
    return {
        **verification,
        "statistical_input": live_input,
        "statistical_plan": live_plan,
        "run_roster": {
            **live_roster,
            "roster_sha256": roster_record["roster_sha256"],
            "unique_training_run_count": 89,
        },
        "compute_columns_match_release_schema": True,
    }


def build_parser() -> argparse.ArgumentParser:
    output_root = ROOT / "outputs" / "tables"
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete P6 metric table and atomically regenerate "
            "the primary decision, durable bootstrap distributions, exact "
            "secondary/Holm family, roster, efficiency, tails, and "
            "violation tables."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=output_root / "p6_frozen_results.parquet",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "analysis" / "statistical_analysis_plan.md",
        help="Frozen, pre-label P6 statistical analysis plan.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=output_root / "p6_run_roster.json",
        help="Frozen and reconstruction-validated 89-run roster.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_root / "p6_primary_comparison.json",
        help="Machine-readable primary and execution decision JSON.",
    )
    parser.add_argument(
        "--bootstrap-output",
        type=Path,
        default=output_root / "p6_primary_bootstrap_distributions.npz",
    )
    parser.add_argument(
        "--secondary-output",
        type=Path,
        default=output_root / "p6_secondary_holm.csv",
    )
    parser.add_argument(
        "--roster-output",
        type=Path,
        default=output_root / "p6_roster_completeness.csv",
    )
    parser.add_argument(
        "--label-efficiency-output",
        type=Path,
        default=output_root / "p6_label_efficiency.csv",
    )
    parser.add_argument(
        "--tail-output",
        type=Path,
        default=output_root / "p6_tail_distribution.csv",
    )
    parser.add_argument(
        "--violation-output",
        type=Path,
        default=output_root / "p6_violation_summary.csv",
    )
    parser.add_argument(
        "--primary-per-case-output",
        type=Path,
        default=output_root / "p6_primary_per_case.csv",
    )
    parser.add_argument(
        "--compute-output",
        type=Path,
        default=output_root / "p6_compute_accounting.csv",
    )
    parser.add_argument(
        "--alpha-attainment-output",
        type=Path,
        default=output_root / "p6_alpha_attainment.csv",
    )
    parser.add_argument(
        "--regime-output",
        type=Path,
        default=output_root / "p6_regime_summary.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the report without writing any output.",
    )
    return parser


def _artifact_record(
    path: Path,
    value: bytes,
    *,
    media_type: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_bytes(value),
        "bytes": len(value),
        "media_type": media_type,
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def generate(
    *,
    input_path: Path,
    plan_path: Path,
    roster_path: Path,
    execution_provenance: dict[str, Any],
    output_paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build the report and all bytes without mutating the filesystem."""

    if not plan_path.is_file() or plan_path.stat().st_size == 0:
        raise ValueError("Frozen P6 statistical analysis plan is missing.")
    if (
        execution_provenance.get("passed") is not True
        or execution_provenance.get(
            "compute_columns_match_release_schema"
        )
        is not True
    ):
        raise PermissionError(
            "P6 statistics require verified final execution provenance."
        )
    live_inputs = (
        (
            input_path,
            execution_provenance.get("statistical_input"),
            "frozen result table",
        ),
        (
            plan_path,
            execution_provenance.get("statistical_plan"),
            "statistical plan",
        ),
        (
            roster_path,
            execution_provenance.get("run_roster"),
            "run roster",
        ),
    )
    def require_live_inputs_unchanged() -> None:
        for path, record, label in live_inputs:
            if (
                not isinstance(record, dict)
                or record.get("sha256") != _sha256_file(path)
                or record.get("bytes") != path.stat().st_size
            ):
                raise PermissionError(
                    f"Verified {label} changed before statistical generation."
                )

    # Check on both sides of parsing so a concurrent replacement cannot feed
    # unverified bytes into the analysis and then restore the committed file.
    require_live_inputs_unchanged()
    roster = _read_and_validate_roster(roster_path)
    frame = _read_input(input_path)
    require_live_inputs_unchanged()
    report, distributions, tables = analyze_p6_results(
        frame,
        execution_provenance_verified=True,
    )
    artifact_bytes = {
        "bootstrap_distributions": _deterministic_npz_bytes(distributions),
        "secondary_contrasts": _table_bytes(
            tables["secondary_contrasts"]
        ),
        "roster_completeness": _table_bytes(
            tables["roster_completeness"]
        ),
        "label_efficiency": _table_bytes(tables["label_efficiency"]),
        "tail_distribution": _table_bytes(tables["tail_distribution"]),
        "violations": _table_bytes(tables["violations"]),
        "primary_per_case": _table_bytes(tables["primary_per_case"]),
        "compute_accounting": _table_bytes(tables["compute_accounting"]),
        "alpha_attainment": _table_bytes(tables["alpha_attainment"]),
        "regime_summary": _table_bytes(tables["regime_summary"]),
    }
    report["provenance"] = {
        "input": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
            "bytes": input_path.stat().st_size,
            "row_count": int(len(frame)),
        },
        "statistical_analysis_plan": {
            "path": str(plan_path),
            "sha256": _sha256_file(plan_path),
            "bytes": plan_path.stat().st_size,
        },
        "confirmatory_run_roster": {
            "path": str(roster_path),
            "sha256": _sha256_file(roster_path),
            "bytes": roster_path.stat().st_size,
            "roster_sha256": roster["roster_sha256"],
            "unique_training_run_count": (
                roster["unique_training_run_count"]
            ),
        },
        "implementation": {
            _implementation_key(path): _sha256_file(path)
            for path in IMPLEMENTATION_PATHS
        },
        "implementation_bundle_sha256": hashlib.sha256(
            "".join(
                f"{_implementation_key(path)}:{_sha256_file(path)}\n"
                for path in IMPLEMENTATION_PATHS
            ).encode("utf-8")
        ).hexdigest(),
        "execution_release_verification": dict(execution_provenance),
    }
    report["artifacts"] = {
        "bootstrap_distributions": {
            **_artifact_record(
                output_paths["bootstrap_distributions"],
                artifact_bytes["bootstrap_distributions"],
                media_type="application/x-npz",
            ),
            "members": sorted(
                [
                    *distributions,
                    "schema_version",
                    "bootstrap_replicates",
                    "bootstrap_seed",
                    "draw_order",
                ]
            ),
            "all_distribution_values_stored_as_float64": True,
            "all_index_values_stored_as_int64": True,
        },
        "secondary_contrasts": _artifact_record(
            output_paths["secondary_contrasts"],
            artifact_bytes["secondary_contrasts"],
            media_type="text/csv",
            row_count=len(tables["secondary_contrasts"]),
        ),
        "roster_completeness": _artifact_record(
            output_paths["roster_completeness"],
            artifact_bytes["roster_completeness"],
            media_type="text/csv",
            row_count=len(tables["roster_completeness"]),
        ),
        "label_efficiency": _artifact_record(
            output_paths["label_efficiency"],
            artifact_bytes["label_efficiency"],
            media_type="text/csv",
            row_count=len(tables["label_efficiency"]),
        ),
        "tail_distribution": _artifact_record(
            output_paths["tail_distribution"],
            artifact_bytes["tail_distribution"],
            media_type="text/csv",
            row_count=len(tables["tail_distribution"]),
        ),
        "violations": _artifact_record(
            output_paths["violations"],
            artifact_bytes["violations"],
            media_type="text/csv",
            row_count=len(tables["violations"]),
        ),
        "primary_per_case": _artifact_record(
            output_paths["primary_per_case"],
            artifact_bytes["primary_per_case"],
            media_type="text/csv",
            row_count=len(tables["primary_per_case"]),
        ),
        "compute_accounting": _artifact_record(
            output_paths["compute_accounting"],
            artifact_bytes["compute_accounting"],
            media_type="text/csv",
            row_count=len(tables["compute_accounting"]),
        ),
        "alpha_attainment": _artifact_record(
            output_paths["alpha_attainment"],
            artifact_bytes["alpha_attainment"],
            media_type="text/csv",
            row_count=len(tables["alpha_attainment"]),
        ),
        "regime_summary": _artifact_record(
            output_paths["regime_summary"],
            artifact_bytes["regime_summary"],
            media_type="text/csv",
            row_count=len(tables["regime_summary"]),
        ),
    }
    artifact_bytes["report"] = _json_bytes(report)
    return report, artifact_bytes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input result table does not exist: {input_path}")
    plan_path = args.plan.resolve()
    roster_path = args.roster.resolve()
    if not plan_path.is_file():
        raise SystemExit(f"Frozen statistical plan does not exist: {plan_path}")
    if not roster_path.is_file():
        raise SystemExit(f"Frozen P6 run roster does not exist: {roster_path}")
    output_paths = {
        "report": args.output.resolve(),
        "bootstrap_distributions": args.bootstrap_output.resolve(),
        "secondary_contrasts": args.secondary_output.resolve(),
        "roster_completeness": args.roster_output.resolve(),
        "label_efficiency": args.label_efficiency_output.resolve(),
        "tail_distribution": args.tail_output.resolve(),
        "violations": args.violation_output.resolve(),
        "primary_per_case": args.primary_per_case_output.resolve(),
        "compute_accounting": args.compute_output.resolve(),
        "alpha_attainment": args.alpha_attainment_output.resolve(),
        "regime_summary": args.regime_output.resolve(),
    }
    if len(set(output_paths.values())) != len(output_paths):
        raise SystemExit("Every registered output path must be distinct.")
    execution_provenance = verify_execution_provenance(
        input_path,
        plan_path,
        roster_path,
    )
    report, values = generate(
        input_path=input_path,
        plan_path=plan_path,
        roster_path=roster_path,
        execution_provenance=execution_provenance,
        output_paths=output_paths,
    )
    if args.dry_run:
        print(values["report"].decode("utf-8"), end="")
    else:
        # The report is last so it never points at a partially generated
        # collection when a preceding atomic replacement fails.
        for name in (
            "bootstrap_distributions",
            "secondary_contrasts",
            "roster_completeness",
            "label_efficiency",
            "tail_distribution",
            "violations",
            "primary_per_case",
            "compute_accounting",
            "alpha_attainment",
            "regime_summary",
            "report",
        ):
            _atomic_bytes(output_paths[name], values[name])
        print(output_paths["report"])
    return 0 if report["execution_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
