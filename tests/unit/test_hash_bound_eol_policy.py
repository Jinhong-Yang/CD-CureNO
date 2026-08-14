from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


PLAN_HASHES = {
    "analysis/statistical_analysis_plan.md": (
        "83db461e069bf43f41d5927ec3477053bf0b8ff806fc7f8605a6368413afe9f7"
    ),
    "analysis/statistical_analysis_plan_p6_v2.md": (
        "1741fe088f573f050fe92605716570609ecfd372773fdf7d09c37696c9f97bdb"
    ),
    "analysis/statistical_analysis_plan_p6_v2_supplemental_v1.md": (
        "e99a2339b312ba2e9a4ad6164fdf34529bc157cca20aa67f169130e644c21eef"
    ),
}


def _git(*arguments: str, cwd: Path) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout


def test_hash_bound_text_remains_lf_under_autocrlf_checkout(
    tmp_path: Path,
) -> None:
    """Exercise a real clone rather than only inspecting attributes."""

    repository = Path.cwd().resolve()
    source = tmp_path / "source"
    source.mkdir()
    attributes = (repository / ".gitattributes").read_bytes()
    (source / ".gitattributes").write_bytes(attributes)
    expected_bytes: dict[str, bytes] = {}
    for relative, expected_sha256 in PLAN_HASHES.items():
        raw = _git("show", f"HEAD:{relative}", cwd=repository)
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        assert b"\r\n" not in raw
        destination = source.joinpath(*Path(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        expected_bytes[relative] = raw
    for arguments in (
        ("init", "--quiet"),
        ("config", "user.email", "eol-test@example.invalid"),
        ("config", "user.name", "EOL Test"),
        ("config", "core.autocrlf", "true"),
        ("add", "--", "."),
        ("commit", "--quiet", "-m", "hash-bound fixture"),
    ):
        _git(*arguments, cwd=source)
    clone = tmp_path / "clone"
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "clone",
            "--quiet",
            "--no-local",
            str(source),
            str(clone),
        ],
        check=True,
        capture_output=True,
    )
    for relative, expected in expected_bytes.items():
        observed = clone.joinpath(*Path(relative).parts).read_bytes()
        assert observed == expected
        assert hashlib.sha256(observed).hexdigest() == PLAN_HASHES[relative]
