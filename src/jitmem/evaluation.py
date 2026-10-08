from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .memory import MemoryBank, Trajectory
from .usage import token_total


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))
    temporary.replace(path)


def summarize(results: list[dict]) -> dict:
    def average(key):
        return statistics.mean(result[key] for result in results) if results else 0.0

    by_type = {}
    for name in sorted({r["task_type"] for r in results}):
        group = [r for r in results if r["task_type"] == name]
        by_type[name] = {
            "tasks": len(group),
            "success_rate": statistics.mean(r["success"] for r in group),
        }
    labeled = [r for r in results if r["judge"] is not None]
    confusion = {
        "true_positive": sum(r["success"] and r["judge"]["success"] for r in labeled),
        "false_positive": sum(not r["success"] and r["judge"]["success"] for r in labeled),
        "false_negative": sum(r["success"] and not r["judge"]["success"] for r in labeled),
        "true_negative": sum(not r["success"] and not r["judge"]["success"] for r in labeled),
    }
    role_usage = {}
    for result in results:
        for role, usage in result["usage"].get("by_role", {}).items():
            aggregate = role_usage.setdefault(
                role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            for key, value in usage.items():
                aggregate[key] = (
                    aggregate[key] + value
                    if key == "calls"
                    else token_total([aggregate[key], value])
                )
    batches = {}
    for result in results:
        batches.setdefault(str(result.get("batch_index", 0)), []).append(result)
    executor_usage = role_usage.get("executor", {"prompt_tokens": 0, "completion_tokens": 0})

    def mean_executor(field):
        total = executor_usage[field]
        return (
            total / len(results)
            if results and total is not None
            else (None if total is None else 0.0)
        )

    return {
        "tasks": len(results),
        "successes": sum(r["success"] for r in results),
        "success_rate": average("success"),
        "mean_reward": average("reward"),
        "mean_steps": average("steps"),
        "mean_environment_steps": average("environment_steps"),
        "mean_latency_seconds": average("latency_seconds"),
        "invalid_actions": sum(r["invalid_actions"] for r in results),
        "truncated_episodes": sum(r["truncated"] for r in results),
        "stored_episodes": sum(r["stored"] for r in results),
        "prompt_tokens": token_total(r["usage"]["prompt_tokens"] for r in results),
        "completion_tokens": token_total(r["usage"]["completion_tokens"] for r in results),
        "usage_complete": all(r["usage"]["complete"] for r in results),
        "incomplete_generations": sum(len(r.get("generation_failures", [])) for r in results),
        "usage_by_role": role_usage,
        "mean_executor_prompt_tokens": mean_executor("prompt_tokens"),
        "mean_executor_completion_tokens": mean_executor("completion_tokens"),
        "judge_parse_errors": sum(r["judge"].get("parse_error", False) for r in labeled),
        "by_batch": {
            index: {
                "tasks": len(group),
                "success_rate": statistics.mean(r["success"] for r in group),
                "memory_size_before": group[0]["memory_size_before"],
            }
            for index, group in batches.items()
        },
        "judge_confusion": confusion,
        "by_task_type": by_type,
    }


def initial_bank(config, task_ids: set[str]) -> MemoryBank:
    source = config.experiment.warm_start
    if not source:
        return MemoryBank()
    entries = [
        Trajectory.from_dict(json.loads(line))
        for line in Path(source).read_text().splitlines()
        if line.strip()
    ]
    if any(item.split != "train" for item in entries):
        raise ValueError("Warm-start memory must contain only train-split trajectories.")
    if any(item.task_id in task_ids for item in entries):
        raise ValueError("Warm-start memory overlaps evaluation task IDs.")
    if config.experiment.store_policy == "judge" and any(
        item.judge_success is not True for item in entries
    ):
        raise ValueError(
            "Judge-filtered warm-start memory requires positive executor judge labels."
        )
    return MemoryBank(entries)


def _run_episode_worker(config, task, bank_entries: list[dict]) -> tuple[dict, dict | None]:
    """Construct private runtime objects in a spawned worker, without artifact writes.

    RunConfig contains credential variable names only. Actual keys are read from
    the child's inherited environment when its own clients are constructed.
    Every submitted episode receives the same serialized batch memory snapshot.
    """
    from .api import ChatClient
    from .environments import ALFWorldEnvironment, FakeHouseholdEnvironment
    from .pipeline import MockChatClient, Pipeline

    if config.environment.backend == "mock":
        executor = curator = MockChatClient()
        environment = FakeHouseholdEnvironment(max_steps=config.experiment.max_steps)
    elif config.environment.backend == "alfworld":
        executor = ChatClient(config.executor)
        curator = (
            ChatClient(config.curator)
            if config.experiment.method in {"jitmem", "write-summary"}
            else executor
        )
        environment = ALFWorldEnvironment(max_steps=config.experiment.max_steps)
    else:
        raise ValueError("Parallel evaluation supports only standard mock and alfworld backends.")
    bank = MemoryBank([Trajectory.from_dict(entry) for entry in bank_entries])
    pipeline = Pipeline(config, executor, curator)
    try:
        result, stored = pipeline.run_episode(task, environment, bank)
        result["worker_pid"] = os.getpid()
        return result, stored.to_dict() if stored is not None else None
    finally:
        environment.close()


def _completed_episodes(config, batch, pipeline, environment, bank, pool):
    """Yield promptly completed episodes while keeping their scheduled indices."""
    if pool is None:
        for index, task in enumerate(batch):
            try:
                result, stored = pipeline.run_episode(task, environment, bank)
                result["worker_pid"] = os.getpid()
            except Exception as error:
                yield index, None, None, error
                return
            yield index, result, stored, None
        return

    snapshot = [entry.to_dict() for entry in bank.entries]
    futures = {
        pool.submit(_run_episode_worker, config, task, snapshot): index
        for index, task in enumerate(batch)
    }
    for future in as_completed(futures):
        index = futures[future]
        try:
            result, stored_data = future.result()
            stored = Trajectory.from_dict(stored_data) if stored_data is not None else None
        except Exception as error:
            # Pending work is canceled; already active requests finish in their
            # isolated processes. The failed batch cannot advance the checkpoint.
            for pending in futures:
                pending.cancel()
            yield index, None, None, error
            return
        yield index, result, stored, None


def evaluate(config, tasks, pipeline, environment_factory, *, resume: bool = False) -> dict:
    if not tasks:
        raise ValueError("No eligible tasks were discovered.")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Evaluation task IDs must be unique.")
    workers = getattr(config.experiment, "workers", 1)
    if workers > 1:
        from .pipeline import Pipeline

        if type(pipeline) is not Pipeline:
            raise ValueError("Parallel evaluation requires the standard Pipeline.")
        if config.environment.backend not in {"mock", "alfworld"}:
            raise ValueError(
                "Parallel evaluation supports only standard mock and alfworld backends."
            )
    configuration = asdict(config)
    task_manifest = []
    for task in tasks:
        item = asdict(task)
        if task.game_file:
            item["game_sha256"] = hashlib.sha256(Path(task.game_file).read_bytes()).hexdigest()
        task_manifest.append(item)
    code_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    packages = {}
    for package in ["jitmem-reproduction", "alfworld", "textworld", "fast-downward-textworld"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    warm_start_hash = (
        hashlib.sha256(Path(config.experiment.warm_start).read_bytes()).hexdigest()
        if config.experiment.warm_start
        else None
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "config": configuration,
                "tasks": task_manifest,
                "source_hashes": code_hashes,
                "packages": packages,
                "python_version": sys.version_info[:3],
                "warm_start_sha256": warm_start_hash,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = Path(config.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": configuration,
        "fingerprint": fingerprint,
        "tasks": task_manifest,
        "source_hashes": code_hashes,
        "warm_start_sha256": warm_start_hash,
        "bm25": {
            "idf": "log(1 + (N-df+0.5)/(df+0.5))",
            "k1": 1.5,
            "b": 0.75,
            "tokenizer": "lowercase ASCII alphanumeric runs",
            "tie_break": "insertion order",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "benchmark": config.environment.backend == "alfworld",
        "variant": "prompted curator; no local training",
        "prompt_source": "semantic paraphrases of JITMEM Appendix A and SkillOS A.4",
        "batch_execution": (
            "spawn process parallel execution with a shared frozen bank per batch"
            if workers > 1
            else "sequential execution with a shared frozen bank per batch"
        ),
        "workers": workers,
        "process_start_method": "spawn" if workers > 1 else None,
        "memory_commit_order": "predetermined task order after the complete batch succeeds",
    }
    existing_manifest = output / "manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if not resume:
            raise ValueError(
                f"Output already contains a run: {output}. Choose a new output or --resume."
            )
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume with a different config or task manifest.")
    write_json(existing_manifest, metadata)
    summaries = []
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        for seed in config.experiment.seeds:
            run_dir = output / f"seed_{seed}"
            run_dir.mkdir(exist_ok=True)
            checkpoint_path = run_dir / "checkpoint.json"
            ordered = list(tasks)
            random.Random(seed).shuffle(ordered)
            write_json(run_dir / "task_order.json", [task.task_id for task in ordered])
            results = []
            bank = initial_bank(config, {task.task_id for task in tasks})
            if resume and checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text())
                if checkpoint["fingerprint"] != fingerprint:
                    raise ValueError("Checkpoint fingerprint differs from this experiment.")
                results = checkpoint["results"]
                bank = MemoryBank([Trajectory.from_dict(entry) for entry in checkpoint["bank"]])
                if [r["task_id"] for r in results] != [
                    task.task_id for task in ordered[: len(results)]
                ]:
                    raise ValueError("Checkpoint task order is inconsistent.")
                if len(results) != len(ordered) and len(results) % config.experiment.batch_size:
                    raise ValueError("Checkpoint contains a partially committed batch.")
            (run_dir / "episodes").mkdir(exist_ok=True)
            # Parallel workers create standard private environments from RunConfig;
            # the supplied factory remains the serial extension/testing interface.
            environment = environment_factory() if pool is None else None
            try:
                for offset in range(len(results), len(ordered), config.experiment.batch_size):
                    batch = ordered[offset : offset + config.experiment.batch_size]
                    batch_results = [None] * len(batch)
                    pending = [None] * len(batch)
                    for index, result, stored, error in _completed_episodes(
                        config, batch, pipeline, environment, bank, pool
                    ):
                        task = batch[index]
                        if error is not None:
                            # Infrastructure errors abort the entire current batch;
                            # already written episode artifacts remain uncommitted.
                            write_json(
                                run_dir / "error.json",
                                {
                                    "task_id": task.task_id,
                                    "error_type": type(error).__name__,
                                    "message": str(error),
                                    "committed_tasks": len(results),
                                    "batch_offset": offset,
                                    "checkpoint": (
                                        str(checkpoint_path) if checkpoint_path.exists() else None
                                    ),
                                    "workers": workers,
                                },
                            )
                            print(
                                f"seed={seed} aborted batch={offset // config.experiment.batch_size} "
                                f"task={task.task_id} error={type(error).__name__} "
                                f"committed_tasks={len(results)}",
                                flush=True,
                            )
                            raise error
                        episode_number = offset + index
                        result["batch_index"] = offset // config.experiment.batch_size
                        write_json(run_dir / "episodes" / f"{episode_number:04d}.json", result)
                        batch_results[index] = result
                        pending[index] = stored
                        print(
                            f"seed={seed} task={episode_number + 1}/{len(ordered)} "
                            f"success={result['success']} steps={result['steps']} "
                            f"bank={len(bank.entries)} worker_pid={result['worker_pid']}",
                            flush=True,
                        )
                    # Completion order never changes insertion/retrieval tie order.
                    # No task sees another task's newly collected batch memory.
                    bank.append_batch([item for item in pending if item is not None])
                    results.extend(batch_results)
                    write_json(
                        checkpoint_path,
                        {
                            "fingerprint": fingerprint,
                            "results": results,
                            "bank": [item.to_dict() for item in bank.entries],
                        },
                    )
            finally:
                if environment is not None:
                    environment.close()
            compact = [
                {key: value for key, value in result.items() if key not in {"calls", "trajectory"}}
                for result in results
            ]
            write_jsonl(run_dir / "results.jsonl", compact)
            write_jsonl(run_dir / "memory.jsonl", [item.to_dict() for item in bank.entries])
            summary = {"seed": seed, "benchmark": metadata["benchmark"], **summarize(results)}
            write_json(run_dir / "summary.json", summary)
            if (run_dir / "error.json").exists():
                (run_dir / "error.json").unlink()
            summaries.append(summary)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    rates = [item["success_rate"] for item in summaries]
    aggregate = {
        "method": config.experiment.method,
        "benchmark": metadata["benchmark"],
        "split": config.environment.split,
        "runs": summaries,
        "success_rate_mean": statistics.mean(rates),
        "success_rate_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "std_definition": "sample standard deviation (ddof=1)",
        "variant": metadata["variant"],
    }
    write_json(output / "summary.json", aggregate)
    return aggregate
