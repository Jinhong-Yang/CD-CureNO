from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "external" / "ResFNO"
UPSTREAM_CASE1 = UPSTREAM_ROOT / "data" / "Case1.mat"
UPSTREAM_DEPENDENT_MODULES = {
    "tests/integration/test_joint_training_run.py",
    "tests/integration/test_legacy_training_run.py",
    "tests/regression/test_legacy_artifacts.py",
    "tests/regression/test_legacy_issues.py",
    "tests/unit/test_joint_case1.py",
    "tests/unit/test_legacy_training.py",
}


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del config
    if (UPSTREAM_ROOT / "Step1_main.py").is_file() and UPSTREAM_CASE1.is_file():
        return
    marker = pytest.mark.skip(
        reason=(
            "requires separately acquired upstream ResFNO code and Case1.mat; "
            "those artifacts are not redistributed by CD-CureNO"
        )
    )
    for item in items:
        relative = item.path.resolve().relative_to(REPO_ROOT).as_posix()
        if relative in UPSTREAM_DEPENDENT_MODULES:
            item.add_marker(marker)
