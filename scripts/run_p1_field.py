"""Launch a resumable local P1 field sweep over independent spatial models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from cdcureno.evaluation.checkpoint import _config_from_run
from cdcureno.legacy.training import _register_result


def _git(project_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_tasks(
    sweep_id: str,
    experiment: str,
    seeds: list[int],
    locations: list[int],
    git_sha: str,
) -> list[dict[str, Any]]:
    tasks = []
    for seed in seeds:
        for location in locations:
            run_id = (
                f"{sweep_id}__P1__{experiment}-x{location:02d}__Case1field__"
                f"n50__seed{seed}__{git_sha[:7]}"
            )
            tasks.append(
                {
                    "run_id": run_id,
                    "experiment": experiment,
                    "seed": seed,
                    "location": location,
                    "status": "pending",
                    "return_code": None,
                }
            )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run independent P1 location models with bounded local concurrency."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["legacy_resfno_exact", "legacy_resfno_corrected"],
    )
    parser.add_argument("--seeds", default="1", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--locations", default="0-50", help="Inclusive range A-B or comma-separated IDs."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sweep-id",
        default=datetime.now().astimezone().strftime("%Y%m%d-%H%M"),
        help="Stable YYYYMMDD-HHMM prefix used for deterministic run IDs.",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    git_sha = _git(project_root, "rev-parse", "HEAD")
    git_status = _git(project_root, "status", "--short")
    if git_status and not args.allow_dirty:
        print("error: worktree is dirty; commit the sweep implementation first.", file=sys.stderr)
        return 2
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if "-" in args.locations and "," not in args.locations:
        start, end = (int(value) for value in args.locations.split("-", maxsplit=1))
        locations = list(range(start, end + 1))
    else:
        locations = [int(value) for value in args.locations.split(",") if value.strip()]
    if not seeds or not locations:
        raise ValueError("At least one seed and location are required.")
    if args.max_workers < 1 or args.num_threads < 1:
        raise ValueError("Worker and thread counts must be positive.")

    tasks = build_tasks(
        args.sweep_id, args.experiment, seeds, locations, git_sha
    )
    state_path = (
        project_root
        / "outputs"
        / "runs"
        / "_sweeps"
        / f"{args.sweep_id}__{args.experiment}.json"
    )
    state = {
        "sweep_id": args.sweep_id,
        "experiment": args.experiment,
        "git_sha": git_sha,
        "epochs": args.epochs,
        "max_workers": args.max_workers,
        "num_threads_per_run": args.num_threads,
        "device": args.device,
        "tasks": tasks,
        "started_at": None,
        "completed_at": None,
    }
    if args.dry_run:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    lock = threading.Lock()
    state["started_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(state_path, state)
    task_by_id = {task["run_id"]: task for task in tasks}

    def run_task(task: dict[str, Any]) -> tuple[str, int, str]:
        run_dir = project_root / "outputs" / "runs" / task["run_id"]
        if (run_dir / "DONE").is_file():
            return task["run_id"], 0, "already_done"
        command = [
            str(project_root / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "cdcureno.cli.train",
            "--experiment",
            task["experiment"],
            "--location",
            str(task["location"]),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(task["seed"]),
            "--device",
            args.device,
            "--num-threads",
            str(args.num_threads),
            "--run-id",
            task["run_id"],
            "--no-register-result",
        ]
        if (run_dir / "checkpoints" / "last.pt").is_file() and not (run_dir / "DONE").is_file():
            command.append("--resume")
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        message = result.stdout[-2000:] + result.stderr[-2000:]
        return task["run_id"], result.returncode, message

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            run_id, return_code, message = future.result()
            with lock:
                task = task_by_id[run_id]
                task["return_code"] = return_code
                task["status"] = "completed" if return_code == 0 else "failed"
                task["message_tail"] = message
                _atomic_json(state_path, state)
            completed = sum(task["status"] == "completed" for task in tasks)
            failed = sum(task["status"] == "failed" for task in tasks)
            print(
                f"[{completed + failed}/{len(tasks)}] {run_id} "
                f"status={task['status']} completed={completed} failed={failed}",
                flush=True,
            )

    state["completed_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(state_path, state)
    all_completed = all(task["status"] == "completed" for task in tasks)
    if all_completed:
        for task in tasks:
            run_dir = project_root / "outputs" / "runs" / task["run_id"]
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = _config_from_run(run_dir)
            _register_result(
                project_root,
                config,
                task["run_id"],
                git_sha,
                metrics["started_at"],
                metrics["completed_at"],
                metrics_path,
            )
        state["results_index_registered"] = True
        _atomic_json(state_path, state)
    return 0 if all_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
