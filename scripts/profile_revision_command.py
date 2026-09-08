"""Profile an explicit command without altering its numerical implementation.

Install psutil==7.2.2 for the recorded revision environment. Peak tree RSS is
sampled and sums resident memory of the command process and live descendants;
shared pages can be counted more than once. It is not exclusive physical RAM.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil


def git_info(cwd: Path) -> dict:
    result = {}
    for key, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "tracked_status": ["git", "status", "--porcelain", "--untracked-files=no"],
        "diff": ["git", "diff", "HEAD", "--"],
    }.items():
        proc = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
        result[key] = proc.stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode:
            result[key + "_error"] = proc.stderr.decode("utf-8", errors="replace")
    result["diff_sha256"] = hashlib.sha256(result.pop("diff").encode()).hexdigest()
    return result


def cpu_name() -> str:
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--context", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.sample_interval <= 0:
        parser.error("Provide a command after -- and a positive sample interval.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cwd = args.cwd.resolve()
    initial_mem = psutil.virtual_memory()
    metadata = {
        "schema_version": 1,
        "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(cwd),
        "context": args.context,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_model": cpu_name(),
        "cpu_logical_count": psutil.cpu_count(),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "visible_total_memory_bytes": initial_mem.total,
        "initial_available_memory_bytes": initial_mem.available,
        "psutil_version": psutil.__version__,
        "sample_interval_seconds": args.sample_interval,
        "memory_method": "sampled sum of RSS for the command process and its live descendants",
        "memory_limitations": "Shared resident pages may be counted for each process. Sub-sample peaks can be missed. This is not exclusive physical memory.",
        "environment_controls": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")},
        "source": git_info(cwd),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "run_started.json").write_text(json.dumps(metadata, indent=2) + "\n")
    peak_rss = 0
    peak_main_rss = 0
    peak_process_count = 0
    samples = 0
    first = time.perf_counter()
    with (output / "stdout.log").open("wb") as log, (output / "memory_samples.csv").open("w") as trace:
        trace.write("elapsed_seconds,tree_rss_bytes,main_rss_bytes,live_process_count\n")
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        main_process = psutil.Process(child.pid)
        while True:
            elapsed = time.perf_counter() - first
            tree_rss = main_rss = count = 0
            try:
                members = [main_process] + main_process.children(recursive=True)
            except psutil.Error:
                members = []
            for process in members:
                try:
                    rss = process.memory_info().rss
                    tree_rss += rss
                    count += 1
                    if process.pid == child.pid:
                        main_rss = rss
                except psutil.Error:
                    continue
            peak_rss = max(peak_rss, tree_rss)
            peak_main_rss = max(peak_main_rss, main_rss)
            peak_process_count = max(peak_process_count, count)
            samples += 1
            trace.write(f"{elapsed:.6f},{tree_rss},{main_rss},{count}\n")
            if child.poll() is not None:
                break
            time.sleep(args.sample_interval)
        code = child.wait()
    metadata.update({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command_wall_seconds": time.perf_counter() - first,
        "exit_code": code,
        "sampled_peak_tree_rss_bytes": peak_rss,
        "sampled_peak_main_rss_bytes": peak_main_rss,
        "maximum_live_process_count": peak_process_count,
        "sample_count": samples,
        "stdout_sha256": hashlib.sha256((output / "stdout.log").read_bytes()).hexdigest(),
        "source_after": git_info(cwd),
    })
    metadata["tracked_source_unchanged_during_run"] = metadata["source"] == metadata["source_after"]
    (output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "label", "command_wall_seconds", "exit_code", "sampled_peak_tree_rss_bytes",
        "maximum_live_process_count", "tracked_source_unchanged_during_run")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
