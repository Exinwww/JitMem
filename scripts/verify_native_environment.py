#!/usr/bin/env python3
"""Verify real ALFWorld state transitions using official expert walkthroughs.

This script is exclusively an environment installation check. It is never a
model benchmark, never constructs memory, and never invokes a model API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from jitmem.environments import (
    TASK_TYPES,
    ALFWorldEnvironment,
    TaskSpec,
    discover_tasks,
    resolve_data_root,
)

DEFAULT_DATA_ROOT = Path("/Users/linbei/workspace/experiential_memory/data/alfworld")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/native_environment_smoke.json"
SPLITS = ("valid_seen", "valid_unseen")


def _provenance(data_root: Path) -> dict[str, Any]:
    manifest_file = data_root / "download_manifest.json"
    if not manifest_file.is_file():
        return {"download_manifest": None}
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "download_manifest": str(manifest_file),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_commit": manifest.get("source_commit"),
        "archive_hash_note": (
            "Hashes recorded by the data downloader; archive bytes are not rehashed by this verifier."
        ),
        "dataset_archives": [
            {key: asset[key] for key in ("name", "url", "release", "sha256", "size_bytes")}
            for asset in manifest.get("assets", [])
            if asset.get("name", "").endswith(".zip")
        ],
    }


def _verify_task(task: TaskSpec, max_steps: int) -> dict[str, Any]:
    env = ALFWorldEnvironment(max_steps=max_steps)
    start = time.monotonic()
    result: dict[str, Any] = {
        "task_id": task.task_id,
        "split": task.split,
        "task_type": task.task_type,
        "game_file": task.game_file,
        "expert_only": True,
        "is_model_benchmark": False,
        "success": False,
        "native_won": False,
        "all_actions_admissible": True,
        "trace": [],
    }
    try:
        game_bytes = Path(task.game_file).read_bytes()
        result["game_sha256"] = hashlib.sha256(game_bytes).hexdigest()
        # Ground-truth actions are permitted only in this isolated verifier.
        # Neither the benchmark evaluator nor agent receives this field.
        walkthrough = json.loads(game_bytes).get("walkthrough")
        if not isinstance(walkthrough, list) or not walkthrough:
            raise ValueError("Game does not contain a nonempty expert walkthrough")
        if not all(isinstance(action, str) for action in walkthrough):
            raise ValueError("Expert walkthrough must be a list of command strings")
        state = env.reset(task)
        result["initial_observation"] = state.observation
        look_state = env.step("look")
        result["look_check"] = {
            "observation": look_state.observation,
            "native_won": look_state.success,
        }
        # Start a fresh episode so the look check cannot change the walkthrough budget.
        state = env.reset(task)
        for action in walkthrough:
            if state.done:
                break
            admissible = action in state.admissible_actions
            if not admissible:
                result["all_actions_admissible"] = False
                result["error"] = f"Expert action is inadmissible: {action}"
                result["trace"].append({"action": action, "admissible_before": False})
                break
            state = env.step(action)
            result["trace"].append(
                {
                    "action": action,
                    "admissible_before": True,
                    "observation": state.observation,
                    "native_won": state.success,
                    "native_score": state.reward,
                    "done": state.done,
                }
            )
        result["native_won"] = state.success
        result["success"] = state.success and result["all_actions_admissible"]
        result["done"] = state.done
        result["native_score"] = state.reward
        result["steps"] = sum(item["admissible_before"] for item in result["trace"])
        if not result["success"] and "error" not in result:
            result["error"] = (
                "Expert walkthrough did not reach native won=True within the step budget"
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        env.close()
    result["elapsed_s"] = round(time.monotonic() - start, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--all",
        dest="all_tasks",
        action="store_true",
        help="Verify every eligible game in the selected splits; default is one per task type.",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    print("Native ALFWorld verifier: is_model_benchmark=false, expert_only=true", flush=True)
    dataset_root = resolve_data_root(args.data_root)
    report: dict[str, Any] = {
        "kind": "native_environment_verification",
        "is_model_benchmark": False,
        "expert_only": True,
        "note": (
            "Official expert walkthroughs verify environment state transitions only. "
            "No model API or memory pipeline is evaluated. native_won is the "
            "environment's won flag exposed through the adapter."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            package: version(package)
            for package in ("alfworld", "textworld", "fast-downward-textworld")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "data_root": str(dataset_root.parent),
        "dataset_provenance": _provenance(dataset_root.parent),
        "max_steps": args.max_steps,
        "selection": "all_eligible_tasks" if args.all_tasks else "one_per_task_type",
        "selected_splits": args.splits,
        "episodes": [],
    }
    expected_episodes = 0
    for split in args.splits:
        tasks = discover_tasks(dataset_root, split)
        if not tasks:
            parser.error(f"No eligible tasks found for split {split}")
        selected = (
            tasks
            if args.all_tasks
            else [
                next((task for task in tasks if task.task_type == task_type), None)
                for task_type in TASK_TYPES.values()
            ]
        )
        expected_episodes += len(selected)
        for index, task in enumerate(selected):
            if task is None:
                result = {
                    "split": split,
                    "task_type": list(TASK_TYPES.values())[index],
                    "success": False,
                    "native_won": False,
                    "error": "No eligible game for this task type",
                }
            else:
                result = _verify_task(task, args.max_steps)
            report["episodes"].append(result)
            print(
                f"{split} {index + 1}/{len(selected)} {result['task_type']}: "
                f"native_won={result['native_won']}, "
                f"steps={result.get('steps', 0)}, passed={result['success']}",
                flush=True,
            )
    episodes = report["episodes"]
    report["summary"] = {
        "expected_episodes": expected_episodes,
        "verified_episodes": len(episodes),
        "passed_episodes": sum(result["success"] for result in episodes),
        "native_won_episodes": sum(result["native_won"] for result in episodes),
        "inadmissible_episodes": sum(
            not result.get("all_actions_admissible", True) for result in episodes
        ),
        "failed_episodes": sum(not result["success"] for result in episodes),
        "by_split": {
            split: {
                "passed": sum(result["success"] for result in episodes if result["split"] == split),
                "total": sum(result["split"] == split for result in episodes),
            }
            for split in args.splits
        },
    }
    report["passed"] = len(episodes) == expected_episodes and all(
        result["success"] for result in episodes
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Saved verifier report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
