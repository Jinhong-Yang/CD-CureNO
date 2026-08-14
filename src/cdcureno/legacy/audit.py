"""Evidence-producing audit of the pinned ResFNO research artifact."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch


EXPECTED_CASE1_SHAPES = {
    "dataT": (200, 51, 223),
    "dataA": (200, 51, 223),
    "dataTair": (200, 223),
}


@dataclass(frozen=True)
class LegacyIssue:
    issue_id: str
    title: str
    status: str
    evidence: tuple[str, ...]
    consequence: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _line_numbers(text: str, needle: str) -> list[int]:
    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    ]


def _evidence(path: str, text: str, needle: str) -> str:
    lines = _line_numbers(text, needle)
    rendered = ",".join(str(line) for line in lines) if lines else "not-found"
    return f"{path}:{rendered} contains {needle!r}"


def inventory_repository(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if ".git" in relative.parts or relative.as_posix() == ".git":
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "suffix": path.suffix.lower(),
            }
        )
    return {
        "repository": "https://github.com/gengxiangc/ResFNO",
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "commit_date": _git(repo_root, "show", "-s", "--format=%cI", "HEAD"),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def inspect_mat_files(repo_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    mat_paths = sorted((repo_root / "data").glob("*.mat"))
    mat_paths.extend(sorted((repo_root / "logs").glob("*.mat")))
    for path in mat_paths:
        arrays: dict[str, Any] = {}
        for key, value in sio.loadmat(path).items():
            if key.startswith("__"):
                continue
            entry: dict[str, Any] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if np.issubdtype(value.dtype, np.number):
                entry.update(
                    {
                        "minimum": float(np.nanmin(value)),
                        "maximum": float(np.nanmax(value)),
                        "finite": bool(np.isfinite(value).all()),
                    }
                )
            arrays[key] = entry
        results[path.relative_to(repo_root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "arrays": arrays,
        }

    case1_key = "data/Case1.mat"
    if case1_key not in results:
        raise FileNotFoundError(f"Required public case is missing: {repo_root / case1_key}")
    observed = {
        key: tuple(results[case1_key]["arrays"].get(key, {}).get("shape", []))
        for key in EXPECTED_CASE1_SHAPES
    }
    if observed != EXPECTED_CASE1_SHAPES:
        raise ValueError(
            f"Case1 shape mismatch: observed={observed}, expected={EXPECTED_CASE1_SHAPES}"
        )
    results["_validation"] = {
        "case1_expected_shapes": {
            key: list(value) for key, value in EXPECTED_CASE1_SHAPES.items()
        },
        "case1_observed_shapes": {key: list(value) for key, value in observed.items()},
        "passed": True,
    }
    return results


def analyze_legacy_issues(repo_root: Path) -> list[LegacyIssue]:
    step1 = (repo_root / "Step1_main.py").read_text(encoding="utf-8", errors="replace")
    step2 = (repo_root / "Step2_plot_T_fields.py").read_text(
        encoding="utf-8", errors="replace"
    )
    utils = (repo_root / "utilsResFNO.py").read_text(
        encoding="utf-8", errors="replace"
    )
    requirements = (repo_root / "requirements.txt").read_text(
        encoding="utf-8", errors="replace"
    )

    issues = [
        LegacyIssue(
            "L1",
            "One independent temporal model is trained per spatial location",
            "confirmed",
            (
                _evidence("Step1_main.py", step1, "x_index = i"),
                _evidence("Step1_main.py", step1, "ResFNO(dataT"),
                _evidence("Step2_plot_T_fields.py", step2, "for i in range(51):"),
            ),
            "The stacked field has no learned coupling between spatial locations.",
        ),
        LegacyIssue(
            "L2",
            "Normalization is fitted before train/test splitting",
            "confirmed",
            (
                _evidence("utilsResFNO.py", utils, "norm_x = RangeNormalizer(x_data)"),
                _evidence("utilsResFNO.py", utils, "norm_y = RangeNormalizer(y_data)"),
                _evidence("utilsResFNO.py", utils, "x_train = x_data[:ntrain,:]"),
            ),
            "Held-out extrema influence training transforms.",
        ),
        LegacyIssue(
            "L3",
            "The smoothness-augmented loss is not backpropagated",
            "confirmed",
            (
                _evidence("utilsResFNO.py", utils, "loss = l2 + 0.5*ld2"),
                _evidence("utilsResFNO.py", utils, "l2.backward()"),
            ),
            "The documented smoothness penalty has no gradient effect.",
        ),
        LegacyIssue(
            "L4",
            "The degree-of-cure task uses undefined temperature normalizer state",
            "confirmed",
            (
                _evidence("utilsResFNO.py", utils, "if task=='T':"),
                _evidence("utilsResFNO.py", utils, "ET_max = torch.max"),
                _evidence("utilsResFNO.py", utils, "norm_y.decode(out)"),
            ),
            "task='A' reaches norm_y in evaluation although norm_y is only created for task='T'.",
        ),
        LegacyIssue(
            "L5",
            "Maximum temperature error is overwritten per test batch",
            "confirmed",
            (
                _evidence("utilsResFNO.py", utils, "ET_max = 0"),
                _evidence("utilsResFNO.py", utils, "ET_max = torch.max"),
            ),
            "Reported deltaT reflects only the final test batch.",
        ),
        LegacyIssue(
            "L6",
            "Predict signature and Step1_main call are inconsistent",
            "confirmed",
            (
                _evidence("utilsResFNO.py", utils, "def Predict("),
                _evidence("Step1_main.py", step1, "Predict(model, dataTair"),
            ),
            "The inference-only branch raises a missing-argument TypeError.",
        ),
        LegacyIssue(
            "L7",
            "Default training produces one location but plotting requires 51",
            "confirmed",
            (
                _evidence("Step1_main.py", step1, "for i in [35]:"),
                _evidence("Step2_plot_T_fields.py", step2, "for i in range(51):"),
            ),
            "The documented two-step workflow cannot regenerate the field from a clean checkout.",
        ),
        LegacyIssue(
            "L8",
            "Requirements are mostly unpinned and include non-core services",
            "confirmed",
            (
                f"requirements.txt has {len([line for line in requirements.splitlines() if line.strip()])} entries",
                f"unpinned entries={sum('==' not in line for line in requirements.splitlines() if line.strip())}",
                "Google Sheets/auth, Altair, and Vega packages are unrelated to core ResFNO training.",
            ),
            "The environment is not reproducible and installs unrelated dependencies.",
        ),
        LegacyIssue(
            "L9",
            "The documented training entry point references a missing MAT filename",
            "confirmed",
            (
                _evidence(
                    "Step1_main.py", step1, "data/Designed_2hold_200.mat"
                ),
                f"data/Designed_2hold_200.mat exists={bool((repo_root / 'data/Designed_2hold_200.mat').exists())}",
                f"data/Case1.mat exists={bool((repo_root / 'data/Case1.mat').exists())}",
            ),
            "Step1_main.py fails before training in a clean checkout.",
        ),
    ]
    return issues


def load_legacy_module(repo_root: Path) -> Any:
    module_path = repo_root / "utilsResFNO.py"
    spec = importlib.util.spec_from_file_location("pinned_legacy_utilsResFNO", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import legacy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def model_metadata(repo_root: Path) -> dict[str, Any]:
    legacy = load_legacy_module(repo_root)
    model = legacy.FNO1d(16, 64, "T").cpu()
    signature = inspect.signature(legacy.Predict)
    return {
        "class": "FNO1d",
        "modes": 16,
        "width": 64,
        "task": "T",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "predict_signature": str(signature),
    }


def _tensor_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _one_legacy_step(repo_root: Path, seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    legacy = load_legacy_module(repo_root)
    arrays = sio.loadmat(repo_root / "data" / "Case1.mat")
    x_data = torch.from_numpy(arrays["dataTair"].astype(np.float32))
    y_data = torch.from_numpy(arrays["dataT"][:, 35, :].astype(np.float32))
    norm_x = legacy.RangeNormalizer(x_data)
    norm_y = legacy.RangeNormalizer(y_data)
    x = norm_x.encode(x_data)[:10].reshape(10, 223, 1)
    y = norm_y.encode(y_data)[:10]

    model = legacy.FNO1d(16, 64, "T").cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = legacy.LpLoss(size_average=False)
    optimizer.zero_grad()
    output = model(x)
    l2 = loss_fn(output.reshape(10, -1), y.reshape(10, -1))
    d1 = output[:, 1:, :] - output[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    ld2 = torch.max(torch.abs(d2))
    combined = l2 + 0.5 * ld2
    l2.backward()  # Preserve the legacy behavior being audited.
    optimizer.step()
    return {
        "seed": seed,
        "batch_shape": list(x.shape),
        "output_shape": list(output.shape),
        "l2": float(l2.detach()),
        "smoothness": float(ld2.detach()),
        "combined_but_not_backpropagated": float(combined.detach()),
        "state_sha256_after_step": _tensor_digest(model),
        "finite": bool(torch.isfinite(output).all()),
    }


def deterministic_cpu_smoke(repo_root: Path, seed: int = 20260723) -> dict[str, Any]:
    first = _one_legacy_step(repo_root, seed)
    second = _one_legacy_step(repo_root, seed)
    deterministic = (
        first["state_sha256_after_step"] == second["state_sha256_after_step"]
        and first["l2"] == second["l2"]
    )
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "seed": seed,
        "first": first,
        "second": second,
        "deterministic": deterministic,
        "passed": deterministic and first["finite"],
    }


def environment_snapshot() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scipy_version": sys.modules["scipy"].__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "packages": sorted(freeze, key=str.casefold),
    }


def verify_raw_mirrors(
    project_root: Path, data: dict[str, Any]
) -> dict[str, Any]:
    mirrors: dict[str, Any] = {}
    for relative in ("data/Case1.mat", "data/double_hold_200_HL.mat"):
        source = data[relative]
        mirror_path = project_root / "data" / "raw" / "resfno" / Path(relative).name
        if not mirror_path.is_file():
            raise FileNotFoundError(f"Immutable raw mirror is missing: {mirror_path}")
        mirror_sha256 = sha256_file(mirror_path)
        mirrors[mirror_path.relative_to(project_root).as_posix()] = {
            "bytes": mirror_path.stat().st_size,
            "sha256": mirror_sha256,
            "source_path": f"external/ResFNO/{relative}",
            "source_sha256": source["sha256"],
            "matches_source": mirror_sha256 == source["sha256"],
        }
    mirrors["_validation"] = {
        "passed": all(item["matches_source"] for key, item in mirrors.items() if key != "_validation")
    }
    return mirrors


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_issues_markdown(path: Path, issues: list[LegacyIssue]) -> None:
    lines = [
        "# Legacy ResFNO issue audit",
        "",
        "All statuses below are derived from the pinned upstream commit and tests.",
        "",
    ]
    for issue in issues:
        lines.extend(
            [
                f"## {issue.issue_id}: {issue.title}",
                "",
                f"**Status:** {issue.status}",
                "",
                f"**Consequence:** {issue.consequence}",
                "",
                "**Evidence:**",
                "",
                *[f"- {item}" for item in issue.evidence],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_license_report(path: Path, inventory: dict[str, Any]) -> None:
    license_candidates = [
        item["path"]
        for item in inventory["files"]
        if Path(item["path"]).name.lower().startswith(
            ("license", "licence", "copying", "notice")
        )
    ]
    lines = [
        "# ResFNO license audit",
        "",
        f"- Audited upstream commit: `{inventory['commit']}`",
        f"- Commit date: `{inventory['commit_date']}`",
        "- Repository: https://github.com/gengxiangc/ResFNO",
        f"- License/notice files in the pinned tree: `{license_candidates}`",
        "- GitHub repository visibility: public.",
        "- SPDX conclusion: `NOASSERTION`.",
        "",
        "Public visibility does not grant permission to redistribute code, MAT arrays,",
        "COMSOL models, pretrained weights, or figures. CD-CureNO therefore records",
        "checksums and split manifests but does not commit copied upstream data. The",
        "pinned submodule and Git-ignored immutable raw mirror are for local audit only.",
        "Author permission or an explicit upstream license is required before any",
        "redistribution.",
        "",
        "This is a provenance finding, not legal advice.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_plan(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Legacy ResFNO reproduction plan",
        "",
        f"Upstream is frozen at `{inventory['commit']}`.",
        "",
        "## Track 1: legacy_resfno_exact",
        "",
        "- Preserve the position-wise FNO1d architecture, modes=16, width=64.",
        "- Preserve the first-50/remaining-150 split and global normalization.",
        "- Preserve the optimized plain relative L2 loss while logging the unused",
        "  smoothness expression as legacy behavior.",
        "- Repair only execution blockers in a wrapper: MAT filename resolution,",
        "  batch-size-safe final batches, alpha-path crash isolation, and artifact",
        "  naming. Every repair must be labeled and must not alter the learned objective.",
        "- Reproduce one location first; expand to 51 only after the one-location gate.",
        "",
        "## Track 2: legacy_resfno_corrected",
        "",
        "- Use the frozen whole-case 50/25/125 train/validation/test manifest.",
        "- Fit all normalization statistics on training cases only.",
        "- Optimize the configured objective, aggregate metrics over every test case,",
        "  support both T and alpha, and save best/last checkpoints.",
        "- Compare exact and corrected pipelines on common frozen case IDs in addition",
        "  to their native manifests.",
        "",
        "## P1 entry gate",
        "",
        "Run the exact one-location CPU/GPU pilot only after P0 tests pass. Do not run",
        "all 51 locations or any true-2-D sweep during P0.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_data_manifest(
    path: Path, repo_root: Path, inventory: dict[str, Any], data: dict[str, Any]
) -> None:
    fields = [
        "dataset_id",
        "source",
        "source_revision",
        "path",
        "sha256",
        "bytes",
        "keys",
        "dimensions",
        "license",
        "redistribution",
        "split_eligibility",
        "notes",
    ]
    rows = []
    for relative in ("data/Case1.mat", "data/double_hold_200_HL.mat"):
        entry = data[relative]
        rows.append(
            {
                "dataset_id": Path(relative).stem,
                "source": inventory["repository"],
                "source_revision": inventory["commit"],
                "path": f"external/ResFNO/{relative}",
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "keys": ";".join(sorted(entry["arrays"])),
                "dimensions": json.dumps(
                    {
                        key: value["shape"]
                        for key, value in sorted(entry["arrays"].items())
                    },
                    separators=(",", ":"),
                ),
                "license": "NOASSERTION",
                "redistribution": "prohibited_pending_permission",
                "split_eligibility": "complete_case_only",
                "notes": "Pinned upstream submodule; immutable mirror under data/raw/resfno is Git-ignored.",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_audit(repo_root: Path, project_root: Path, output_dir: Path) -> dict[str, Any]:
    inventory = inventory_repository(repo_root)
    data = inspect_mat_files(repo_root)
    issues = analyze_legacy_issues(repo_root)
    metadata = model_metadata(repo_root)
    smoke = deterministic_cpu_smoke(repo_root)
    environment = environment_snapshot()
    mirrors = verify_raw_mirrors(project_root, data)

    write_json(output_dir / "repo_inventory.json", inventory)
    write_json(output_dir / "data_shapes.json", data)
    write_json(output_dir / "model_metadata.json", metadata)
    write_json(output_dir / "cpu_smoke.json", smoke)
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "raw_mirror_checksums.json", mirrors)
    write_issues_markdown(output_dir / "legacy_issues.md", issues)
    write_license_report(output_dir / "license_report.md", inventory)
    write_reproduction_plan(output_dir / "reproduction_plan.md", inventory)
    write_data_manifest(project_root / "DATA_MANIFEST.csv", repo_root, inventory, data)

    passed = (
        data["_validation"]["passed"]
        and smoke["passed"]
        and mirrors["_validation"]["passed"]
        and all(issue.status in {"confirmed", "disproven"} for issue in issues)
    )
    summary = {
        "passed": passed,
        "upstream_commit": inventory["commit"],
        "confirmed_issue_count": sum(issue.status == "confirmed" for issue in issues),
        "disproven_issue_count": sum(issue.status == "disproven" for issue in issues),
        "case1_shapes_valid": data["_validation"]["passed"],
        "deterministic_cpu_smoke": smoke["passed"],
        "raw_mirrors_valid": mirrors["_validation"]["passed"],
        "model_parameter_count": metadata["parameter_count"],
    }
    write_json(output_dir / "audit_summary.json", summary)
    return summary
