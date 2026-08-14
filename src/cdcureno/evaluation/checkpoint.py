"""Evaluate selected legacy checkpoints on a frozen evaluation-only manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cdcureno.legacy.audit import load_legacy_module, sha256_file
from cdcureno.legacy.training import LegacyTrainConfig, _evaluate, prepare_data


def _config_from_run(run_dir: Path) -> LegacyTrainConfig:
    payload = json.loads((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    for key in ("data_path", "split_manifest", "output_root", "project_root"):
        payload[key] = Path(payload[key])
    payload["resume"] = False
    payload["register_result"] = False
    return LegacyTrainConfig(**payload).validated()


def _evaluation_case_ids(manifest: dict[str, Any]) -> list[int]:
    if "test" in manifest:
        return [int(value) for value in manifest["test"]]
    return [int(value) for value in manifest["splits"]["test"]]


def evaluate_run_checkpoint(
    run_dir: Path, evaluation_manifest: Path, device: str = "cpu"
) -> dict[str, Any]:
    config = _config_from_run(run_dir)
    manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    case_ids = _evaluation_case_ids(manifest)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation manifest contains duplicate case IDs.")
    prepared, _, y_normalizer, _ = prepare_data(config)
    legacy = load_legacy_module(config.project_root / "external" / "ResFNO")
    torch_device = torch.device(device)
    model = legacy.FNO1d(config.modes, config.width, config.task).to(torch_device)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, cases, predictions = _evaluate(
        model,
        case_ids,
        prepared,
        y_normalizer,
        config.batch_size,
        torch_device,
    )

    output_dir = run_dir / "evaluations" / evaluation_manifest.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.insert(0, "split", evaluation_manifest.stem)
    cases.to_parquet(output_dir / "metrics_per_case.parquet", index=False)
    index = torch.tensor(case_ids, dtype=torch.long)
    np.savez_compressed(
        output_dir / "predictions.npz",
        case_ids=np.asarray(case_ids, dtype=np.int64),
        prediction=predictions,
        target=prepared["y_physical"][index].numpy(),
        input_air=prepared["x_physical"][index].numpy(),
    )
    result = {
        "source_run_id": run_dir.name,
        "experiment": config.experiment,
        "seed": config.seed,
        "location": config.location,
        "selected_epoch": int(checkpoint["epoch"]),
        "evaluation_manifest": evaluation_manifest.name,
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
        "case_count": len(case_ids),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def evaluate_upstream_checkpoint(
    project_root: Path,
    checkpoint_path: Path,
    evaluation_manifest: Path,
    location: int = 35,
) -> dict[str, Any]:
    exact_manifest = project_root / "splits" / "legacy_case1_exact_v1.json"
    config = LegacyTrainConfig(
        experiment="legacy_resfno_exact",
        data_path=project_root / "external" / "ResFNO" / "data" / "Case1.mat",
        split_manifest=exact_manifest,
        output_root=project_root / "outputs" / "runs",
        project_root=project_root,
        location=location,
        task="T",
        epochs=300,
        seed=1,
        device="cpu",
        register_result=False,
    ).validated()
    manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    case_ids = _evaluation_case_ids(manifest)
    prepared, _, y_normalizer, normalization = prepare_data(config)
    legacy = load_legacy_module(project_root / "external" / "ResFNO")
    model = legacy.FNO1d(16, 64, "T").cpu()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    metrics, _, _ = _evaluate(
        model, case_ids, prepared, y_normalizer, batch_size=10, device=torch.device("cpu")
    )
    return {
        "source": "pinned_upstream_checkpoint",
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_manifest": evaluation_manifest.name,
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
        "case_count": len(case_ids),
        "location": location,
        "normalization": normalization,
        "metrics": metrics,
    }
