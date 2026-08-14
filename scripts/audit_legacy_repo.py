"""Generate the P0 legacy repository and public-data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cdcureno.legacy.audit import build_audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Audit the pinned ResFNO repository without modifying it."
    )
    result.add_argument(
        "--repo-root",
        type=Path,
        default=Path("external/ResFNO"),
        help="Path to the pinned upstream ResFNO checkout.",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit"),
        help="Directory for deterministic audit artifacts.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the planned audit without writing outputs.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    repo_root = (project_root / args.repo_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    if not repo_root.is_dir():
        print(f"error: repository does not exist: {repo_root}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"repo_root={repo_root}")
        print(f"output_dir={output_dir}")
        print(
            "planned_outputs=repo_inventory,data_shapes,model_metadata,cpu_smoke,"
            "environment,raw_mirror_checksums,legacy_issues,license_report,"
            "reproduction_plan,audit_summary,DATA_MANIFEST"
        )
        return 0
    summary = build_audit(repo_root, project_root, output_dir)
    print(f"P0 audit passed={summary['passed']}")
    print(f"upstream_commit={summary['upstream_commit']}")
    print(f"confirmed_issues={summary['confirmed_issue_count']}")
    print(f"model_parameters={summary['model_parameter_count']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
