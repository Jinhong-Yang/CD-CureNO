"""Run the public workflow and prove postflight detects a forged metric."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_public_demo_and_metric_tampering(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "demo"
    completed = subprocess.run([sys.executable, str(root / "scripts/run_public_demo.py"), "--output", str(output)],
                               cwd=root, check=True, capture_output=True, text=True)
    receipt = json.loads((output / "postflight_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "passed", completed.stdout
    assert [r["case_id"] for r in receipt["recomputed_metrics"]] == [6, 7]
    # Rehash the modified CSV to establish that recomputation, not just a
    # checksum discrepancy, rejects the wrong numerical value.
    metrics = output / "per_case_metrics.csv"
    rows = metrics.read_text(encoding="utf-8").splitlines()
    cells = rows[1].split(",")
    cells[1] = str(float(cells[1]) + 0.25)
    rows[1] = ",".join(cells)
    metrics.write_text("\n".join(rows)+"\n", encoding="utf-8")
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[metrics.name] = hashlib.sha256(metrics.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rejected = subprocess.run([sys.executable, str(root / "scripts/verify_public_demo.py"), "--input", str(output)],
                              cwd=root, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "Metric mismatch" in rejected.stderr
    assert json.loads((output / "postflight_receipt.json").read_text(encoding="utf-8"))["status"] == "failed"
