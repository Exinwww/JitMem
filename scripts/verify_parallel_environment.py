#!/usr/bin/env python3
"""Verify isolated concurrent ALFWorld environments without model API calls.

Plain threads are unsuitable for TextWorld's process-global parsers and the
Fast Downward translator's process-global options/stdout. Spawned workers
keep those objects isolated. This script checks the environment only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_native_environment import DEFAULT_DATA_ROOT, _verify_task

from jitmem.environments import TASK_TYPES, TaskSpec, discover_tasks

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/parallel_environment_smoke.json"


def _tree_fingerprint(root: Path) -> tuple[str, dict[str, tuple[int, int, str]]]:
    """Record complete file content, sizes and modification times read only."""
    inventory = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        inventory[str(path.relative_to(root))] = (
            stat.st_size,
            stat.st_mtime_ns,
            digest.hexdigest(),
        )
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), inventory


def _worker(task: TaskSpec) -> dict[str, Any]:
    started = time.time()
    result = _verify_task(task, max_steps=30)
    result["worker_pid"] = os.getpid()
    result["started_at_unix"] = started
    result["finished_at_unix"] = time.time()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data_root = args.data_root.expanduser().resolve()
    if (data_root / "data/alfworld").is_dir():
        fingerprint_root = data_root
    elif data_root.name == "json_2.1.1":
        fingerprint_root = data_root.parent
    else:
        fingerprint_root = data_root
    print("Parallel environment verifier: is_model_benchmark=false, expert_only=true", flush=True)
    tasks = []
    for split in ("valid_seen", "valid_unseen"):
        candidates = discover_tasks(data_root, split)
        for task_type in TASK_TYPES.values():
            tasks.append(next(task for task in candidates if task.task_type == task_type))
    tasks = tasks[:10]
    print("Fingerprinting external dataset files before the check (read only).", flush=True)
    before_digest, before = _tree_fingerprint(fingerprint_root)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_worker, tasks))
    elapsed = time.monotonic() - started
    print("Fingerprinting external dataset files after the check (read only).", flush=True)
    after_digest, after = _tree_fingerprint(fingerprint_root)
    changed = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    events = [(result["started_at_unix"], 1) for result in results]
    events += [(result["finished_at_unix"], -1) for result in results]
    active = maximum_active = 0
    for _, change in sorted(events):
        active += change
        maximum_active = max(maximum_active, active)
    passed = all(result["success"] for result in results) and not changed
    report = {
        "kind": "native_parallel_environment_verification",
        "is_model_benchmark": False,
        "expert_only": True,
        "model_api_calls": 0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "ProcessPoolExecutor",
        "process_start_method": "spawn",
        "workers": args.workers,
        "unique_worker_pids": sorted({result["worker_pid"] for result in results}),
        "maximum_overlapping_episodes": maximum_active,
        "elapsed_s": round(elapsed, 3),
        "episodes": results,
        "dataset_integrity": {
            "root": str(fingerprint_root),
            "file_count": len(before),
            "fingerprint_definition": "SHA256 of all relative file paths, sizes, modification times, and complete file SHA256s",
            "before_sha256": before_digest,
            "after_sha256": after_digest,
            "unchanged": before_digest == after_digest,
            "changed_files": changed,
        },
        "summary": {
            "episodes": len(results),
            "native_won": sum(result["native_won"] for result in results),
            "passed_episodes": sum(result["success"] for result in results),
            "inadmissible_episodes": sum(
                not result["all_actions_admissible"] for result in results
            ),
        },
        "passed": passed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    print(f"Maximum overlapping episodes: {maximum_active}; unchanged dataset: {not changed}")
    print(f"Saved verifier report: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
