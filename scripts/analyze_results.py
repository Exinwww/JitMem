#!/usr/bin/env python3
"""Compare completed, matched native ALFWorld no-memory and prompted JITMEM runs.

Reads local artifacts only. No model calls, environment actions, training, or
pooled significance tests are performed. The default requires all 140 seen tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AnalysisError(ValueError):
    """Artifacts are incomplete, inconsistent, or do not describe matched runs."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Cannot read valid JSON: {path} ({type(error).__name__})") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def integer(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def nullable_total(values: list[int | None]) -> int | None:
    return None if any(value is None for value in values) else sum(values)


def check_rate(value: Any, expected: float, label: str) -> None:
    require(
        type(value) in {int, float}
        and math.isfinite(value)
        and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label} disagrees with complete results",
    )


@dataclass
class CompletedArm:
    root: Path
    manifest: dict
    summary: dict
    tasks: dict[str, dict]
    seeds: list[int]
    results: dict[int, list[dict]]
    orders: dict[int, list[str]]
    evidence_files: list[Path]


def load_arm(root: str | Path, method: str, expected_tasks: int) -> CompletedArm:
    root = Path(root).expanduser().resolve()
    manifest_path, summary_path = root / "manifest.json", root / "summary.json"
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    require(isinstance(manifest, dict) and isinstance(summary, dict), f"Invalid arm: {root}")
    require(manifest.get("benchmark") is True, f"{root}: manifest is not a native benchmark")
    require(summary.get("benchmark") is True, f"{root}: summary is not a native benchmark")
    config = manifest.get("config", {})
    environment, experiment = config.get("environment", {}), config.get("experiment", {})
    require(environment.get("backend") == "alfworld", f"{root}: backend must be alfworld")
    require(
        experiment.get("method") == summary.get("method") == method,
        f"{root}: expected method {method}",
    )
    require(summary.get("split") == environment.get("split"), f"{root}: split is inconsistent")
    require(environment.get("limit") is None, f"{root}: limited pilot is not a full benchmark")
    workers = integer(experiment.get("workers"), f"{root}: workers", 1)
    batch_size = integer(experiment.get("batch_size"), f"{root}: batch_size", 1)
    max_steps = integer(experiment.get("max_steps"), f"{root}: max_steps", 1)
    integer(experiment.get("history_window"), f"{root}: history_window", 1)
    require(manifest.get("workers") == workers, f"{root}: worker metadata disagrees")
    require(
        manifest.get("process_start_method") == ("spawn" if workers > 1 else None),
        f"{root}: process start method disagrees with workers",
    )
    expected_execution = (
        "spawn process parallel execution with a shared frozen bank per batch"
        if workers > 1
        else "sequential execution with a shared frozen bank per batch"
    )
    require(
        manifest.get("batch_execution") == expected_execution,
        f"{root}: frozen batch execution metadata disagrees",
    )
    require(
        manifest.get("memory_commit_order")
        == "predetermined task order after the complete batch succeeds",
        f"{root}: memory commit order is undeclared or unsupported",
    )
    seeds = experiment.get("seeds")
    require(isinstance(seeds, list) and bool(seeds), f"{root}: seeds must be a nonempty list")
    for seed in seeds:
        integer(seed, f"{root}: seed")
    require(len(set(seeds)) == len(seeds), f"{root}: duplicate seeds")
    tasks_list = manifest.get("tasks")
    require(isinstance(tasks_list, list), f"{root}: task manifest is missing")
    require(len(tasks_list) == expected_tasks, f"{root}: expected {expected_tasks} manifest tasks")
    tasks = {}
    for task in tasks_list:
        require(isinstance(task, dict), f"{root}: invalid manifest task")
        task_id = task.get("task_id")
        require(isinstance(task_id, str) and bool(task_id), f"{root}: invalid task ID")
        require(task_id not in tasks, f"{root}: duplicate manifest task {task_id}")
        require(task.get("split") == summary["split"], f"{root}: task split disagrees")
        require(isinstance(task.get("task_type"), str), f"{root}: task type is missing")
        tasks[task_id] = task
    declared_runs = summary.get("runs")
    require(isinstance(declared_runs, list), f"{root}: declared runs are missing")
    require(
        len(declared_runs) == len(seeds)
        and {run.get("seed") for run in declared_runs if isinstance(run, dict)} == set(seeds),
        f"{root}: summary seeds disagree with manifest",
    )
    declared = {run["seed"]: run for run in declared_runs}
    results, orders = {}, {}
    evidence_files = [manifest_path, summary_path]
    for seed in sorted(seeds):
        run_dir = root / f"seed_{seed}"
        require(not (run_dir / "error.json").exists(), f"{run_dir}: unresolved run error")
        order_path, result_path = run_dir / "task_order.json", run_dir / "results.jsonl"
        run_summary_path = run_dir / "summary.json"
        order, run_summary = read_json(order_path), read_json(run_summary_path)
        require(
            isinstance(order, list)
            and all(isinstance(task_id, str) for task_id in order)
            and len(order) == len(tasks)
            and len(set(order)) == len(order)
            and set(order) == set(tasks),
            f"{run_dir}: task order must cover the full manifest exactly once",
        )
        seeded_order = list(tasks)
        random.Random(seed).shuffle(seeded_order)
        require(order == seeded_order, f"{run_dir}: task order disagrees with seed shuffle")
        try:
            result_lines = result_path.read_text(encoding="utf-8").splitlines()
            require(all(line.strip() for line in result_lines), f"{run_dir}: blank result lines")
            rows = [json.loads(line) for line in result_lines]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AnalysisError(f"Cannot read complete results: {result_path}") from error
        require(len(rows) == len(tasks), f"{run_dir}: results do not cover the full manifest")
        require(all(isinstance(row, dict) for row in rows), f"{run_dir}: invalid result row")
        require([row.get("task_id") for row in rows] == order, f"{run_dir}: result order disagrees")
        batch_memory_size = None
        batch_stored = 0
        for index, row in enumerate(rows):
            task = tasks[row["task_id"]]
            require(type(row.get("success")) is bool, f"{run_dir}: success must be boolean")
            require(row.get("split") == task["split"], f"{run_dir}: result split disagrees")
            require(row.get("task_type") == task["task_type"], f"{run_dir}: task type disagrees")
            steps = integer(row.get("steps"), f"{run_dir}: steps")
            require(steps <= max_steps, f"{run_dir}: decisions exceed declared cap")
            env_steps = integer(row.get("environment_steps"), f"{run_dir}: environment_steps")
            require(env_steps <= steps, f"{run_dir}: environment steps exceed decisions")
            integer(row.get("invalid_actions"), f"{run_dir}: invalid_actions")
            require(type(row.get("truncated")) is bool, f"{run_dir}: truncated must be boolean")
            require(
                isinstance(row.get("generation_failures"), list)
                and all(isinstance(failure, dict) for failure in row["generation_failures"]),
                f"{run_dir}: failures missing or invalid",
            )
            judgment = row.get("judge")
            require(
                judgment is None
                or (isinstance(judgment, dict) and type(judgment.get("success")) is bool),
                f"{run_dir}: invalid judge verdict",
            )
            require(method != "no-memory" or judgment is None, f"{run_dir}: baseline used a judge")
            require(method != "jitmem" or judgment is not None, f"{run_dir}: JITMEM judge missing")
            integer(row.get("worker_pid"), f"{run_dir}: worker_pid", 1)
            require(
                row.get("batch_index") == index // batch_size,
                f"{run_dir}: batch index disagrees with scheduled task position",
            )
            memory_size = integer(row.get("memory_size_before"), f"{run_dir}: memory_size_before")
            require(type(row.get("stored")) is bool, f"{run_dir}: stored must be boolean")
            require(
                method != "no-memory" or not row["stored"],
                f"{run_dir}: baseline stored memory",
            )
            if index % batch_size == 0:
                if batch_memory_size is not None:
                    require(
                        memory_size == batch_memory_size + batch_stored,
                        f"{run_dir}: memory commit size disagrees with previous complete batch",
                    )
                elif experiment.get("warm_start") is None:
                    require(memory_size == 0, f"{run_dir}: cold start bank is not empty")
                batch_memory_size, batch_stored = memory_size, 0
            require(
                memory_size == batch_memory_size,
                f"{run_dir}: memory changed within a frozen batch",
            )
            batch_stored += row["stored"]
        success_rate = statistics.mean(row["success"] for row in rows)
        for declaration in (run_summary, declared[seed]):
            require(isinstance(declaration, dict), f"{run_dir}: invalid declared summary")
            require(declaration.get("benchmark") is True, f"{run_dir}: not a benchmark run")
            require(declaration.get("seed") == seed, f"{run_dir}: declared seed disagrees")
            require(declaration.get("tasks") == len(tasks), f"{run_dir}: declared count is partial")
            require(
                declaration.get("successes") == sum(row["success"] for row in rows),
                f"{run_dir}: declared successes disagree",
            )
            check_rate(declaration.get("success_rate"), success_rate, f"{run_dir}: declared SR")
        results[seed], orders[seed] = rows, order
        evidence_files.extend([order_path, result_path, run_summary_path])
    rates = [statistics.mean(row["success"] for row in results[seed]) for seed in sorted(seeds)]
    check_rate(summary.get("success_rate_mean"), statistics.mean(rates), f"{root}: aggregate SR")
    check_rate(summary.get("success_rate_std"), mean_std(rates)["std"], f"{root}: aggregate std")
    return CompletedArm(
        root, manifest, summary, tasks, sorted(seeds), results, orders, evidence_files
    )


def arm_metrics(rows: list[dict]) -> dict:
    labels = [row for row in rows if row["judge"] is not None]
    confusion = {
        "labeled_episodes": len(labels),
        "true_positive": sum(row["success"] and row["judge"]["success"] for row in labels),
        "false_positive": sum(not row["success"] and row["judge"]["success"] for row in labels),
        "false_negative": sum(row["success"] and not row["judge"]["success"] for row in labels),
        "true_negative": sum(not row["success"] and not row["judge"]["success"] for row in labels),
        "parse_errors": sum(bool(row["judge"].get("parse_error")) for row in labels),
        "generation_errors": sum(bool(row["judge"].get("generation_error")) for row in labels),
    }
    roles = {role for row in rows for role in row.get("usage", {}).get("by_role", {})}
    roles.add("executor")
    if labels:
        roles.update({"curator", "judge"})
    usage = {}
    for role in sorted(roles):
        pieces = []
        for row in rows:
            role_data = row.get("usage", {}).get("by_role", {}).get(role)
            if role_data is None:
                # These roles are called even if the provider omitted their usage.
                called = role == "executor" or (
                    role in {"curator", "judge"} and row["judge"] is not None
                )
                role_data = {
                    "calls": row["steps"] if role == "executor" else int(called),
                    "prompt_tokens": None if called else 0,
                    "completion_tokens": None if called else 0,
                }
            integer(role_data.get("calls"), f"{role}: calls")
            for field in ("prompt_tokens", "completion_tokens"):
                if role_data.get(field) is not None:
                    integer(role_data[field], f"{role}: {field}")
            pieces.append(role_data)
        usage[role] = {
            "calls": sum(piece["calls"] for piece in pieces),
            "prompt_tokens": nullable_total([piece.get("prompt_tokens") for piece in pieces]),
            "completion_tokens": nullable_total(
                [piece.get("completion_tokens") for piece in pieces]
            ),
        }
    failures = [failure for row in rows for failure in row["generation_failures"]]
    return {
        "tasks": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": statistics.mean(row["success"] for row in rows),
        "mean_decisions": statistics.mean(row["steps"] for row in rows),
        "mean_environment_steps": statistics.mean(row["environment_steps"] for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "truncated_episodes": sum(row["truncated"] for row in rows),
        "empty_bank_episodes": sum(row["memory_size_before"] == 0 for row in rows),
        "stored_episodes": sum(row["stored"] for row in rows),
        "native_failed_stored_episodes": sum(row["stored"] and not row["success"] for row in rows),
        "generation_failures": {
            "calls": len(failures),
            "episodes": sum(bool(row["generation_failures"]) for row in rows),
            "by_role": dict(Counter(failure.get("role", "unknown") for failure in failures)),
            "by_finish_reason": dict(
                Counter(failure.get("finish_reason", "unknown") for failure in failures)
            ),
        },
        "native_vs_judge": confusion,
        "usage_by_role": usage,
        "prompt_tokens": nullable_total(
            [row.get("usage", {}).get("prompt_tokens") for row in rows]
        ),
        "completion_tokens": nullable_total(
            [row.get("usage", {}).get("completion_tokens") for row in rows]
        ),
    }


def build_comparison(
    baseline_dir: str | Path, jitmem_dir: str | Path, *, expected_tasks: int = 140
) -> dict:
    integer(expected_tasks, "expected_tasks", 1)
    baseline = load_arm(baseline_dir, "no-memory", expected_tasks)
    jitmem = load_arm(jitmem_dir, "jitmem", expected_tasks)
    require(baseline.summary["split"] == jitmem.summary["split"], "Split mismatch")
    require(baseline.seeds == jitmem.seeds, "Seed mismatch")
    require(set(baseline.tasks) == set(jitmem.tasks), "Task set mismatch")
    for task_id in baseline.tasks:
        first, second = baseline.tasks[task_id], jitmem.tasks[task_id]
        for field in ("task_type", "split", "description", "game_sha256"):
            require(first.get(field) == second.get(field), f"Task {task_id}: {field} mismatch")
    first_config, second_config = baseline.manifest["config"], jitmem.manifest["config"]
    require(
        first_config.get("executor") == second_config.get("executor"), "Executor config mismatch"
    )
    require(bool(first_config.get("executor", {}).get("model")), "Executor model is undeclared")
    require(
        bool(second_config.get("curator", {}).get("model")), "JITMEM curator model is undeclared"
    )
    for field in ("batch_size", "workers", "max_steps", "history_window"):
        require(
            first_config["experiment"].get(field) == second_config["experiment"].get(field),
            f"Experiment protocol mismatch: {field}",
        )
    for field in ("task_types", "annotation_index"):
        require(
            first_config["environment"].get(field) == second_config["environment"].get(field),
            f"Environment protocol mismatch: {field}",
        )
    for field in (
        "source_hashes",
        "packages",
        "prompt_source",
        "batch_execution",
        "workers",
        "process_start_method",
        "memory_commit_order",
    ):
        require(
            baseline.manifest.get(field) == jitmem.manifest.get(field),
            f"Implementation protocol mismatch: {field}",
        )
    seed_reports = []
    types = sorted({task["task_type"] for task in baseline.tasks.values()})
    for seed in baseline.seeds:
        require(baseline.orders[seed] == jitmem.orders[seed], f"Task order mismatch: seed {seed}")
        first, second = baseline.results[seed], jitmem.results[seed]
        pair_counts = {
            name: 0 for name in ("both_success", "baseline_only", "jitmem_only", "both_failure")
        }
        pairs = []
        for index, (left, right) in enumerate(zip(first, second), 1):
            require(
                left.get("task_description") == right.get("task_description"),
                f"Observed task description mismatch: seed {seed}, {left['task_id']}",
            )
            outcome = (
                "both_success"
                if left["success"] and right["success"]
                else "baseline_only"
                if left["success"]
                else "jitmem_only"
                if right["success"]
                else "both_failure"
            )
            pair_counts[outcome] += 1
            sources = {}
            for name, arm in (("baseline", baseline), ("jitmem", jitmem)):
                run_dir = arm.root / f"seed_{seed}"
                episode_path = run_dir / "episodes" / f"{index - 1:04d}.json"
                sources[name] = {
                    "results_jsonl": str(run_dir / "results.jsonl"),
                    "line": index,
                    "episode_json": str(episode_path) if episode_path.exists() else None,
                }
            pairs.append(
                {
                    "task_id": left["task_id"],
                    "task_type": left["task_type"],
                    "baseline_success": left["success"],
                    "jitmem_success": right["success"],
                    "jitmem_judge_success": right["judge"]["success"],
                    "jitmem_stored": right["stored"],
                    "outcome": outcome,
                    "sources": sources,
                }
            )
        metrics = {"baseline": arm_metrics(first), "jitmem": arm_metrics(second)}
        by_type = {}
        for task_type in types:
            left = [row for row in first if row["task_type"] == task_type]
            right = [row for row in second if row["task_type"] == task_type]
            base_rate = statistics.mean(row["success"] for row in left)
            jit_rate = statistics.mean(row["success"] for row in right)
            by_type[task_type] = {
                "tasks": len(left),
                "baseline_sr": base_rate,
                "jitmem_sr": jit_rate,
                "delta_pp": 100 * (jit_rate - base_rate),
            }
        seed_reports.append(
            {
                "seed": seed,
                "tasks": len(first),
                "paired_counts": pair_counts,
                "delta_pp": 100
                * (metrics["jitmem"]["success_rate"] - metrics["baseline"]["success_rate"]),
                "mean_decision_delta": metrics["jitmem"]["mean_decisions"]
                - metrics["baseline"]["mean_decisions"],
                "mean_environment_step_delta": metrics["jitmem"]["mean_environment_steps"]
                - metrics["baseline"]["mean_environment_steps"],
                **metrics,
                "by_task_type": by_type,
                "task_pairs": pairs,
            }
        )
    sources = {}
    for name, arm in (("baseline", baseline), ("jitmem", jitmem)):
        sources[name] = {
            "root": str(arm.root),
            "fingerprint": arm.manifest.get("fingerprint"),
            "files": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in arm.evidence_files
            ],
        }
    return {
        "kind": "matched_native_alfworld_comparison",
        "benchmark": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": baseline.summary["split"],
        "tasks_per_seed": expected_tasks,
        "seeds": baseline.seeds,
        "executor_model": first_config["executor"]["model"],
        "curator_model": second_config["curator"]["model"],
        "protocol": {
            "matched_executor_config": first_config["executor"],
            "batch_size": first_config["experiment"].get("batch_size"),
            "workers": first_config["experiment"]["workers"],
            "process_start_method": baseline.manifest["process_start_method"],
            "batch_execution": baseline.manifest["batch_execution"],
            "memory_commit_order": baseline.manifest["memory_commit_order"],
            "task_order": "random.Random(seed).shuffle(manifest task order), preserved after asynchronous completion",
            "max_decisions": first_config["experiment"].get("max_steps"),
            "history_window": first_config["experiment"].get("history_window"),
            "jitmem_memory_settings": {
                field: second_config["experiment"].get(field)
                for field in ("retrieval_k", "task_adaptive", "store_policy", "warm_start")
            },
            "prompt_source": jitmem.manifest["prompt_source"],
        },
        "variant": "API prompted curator; no local training; not the paper's RL-trained curator",
        "paper_executor_models": ["Qwen3-8B", "Gemini-2.5-Pro", "GPT-5.4"],
        "sources": sources,
        "per_seed": seed_reports,
        "across_seed": {
            "baseline_sr_percent": mean_std(
                [100 * run["baseline"]["success_rate"] for run in seed_reports]
            ),
            "jitmem_sr_percent": mean_std(
                [100 * run["jitmem"]["success_rate"] for run in seed_reports]
            ),
            "delta_pp": mean_std([run["delta_pp"] for run in seed_reports]),
            "mean_decision_delta": mean_std([run["mean_decision_delta"] for run in seed_reports]),
            "mean_environment_step_delta": mean_std(
                [run["mean_environment_step_delta"] for run in seed_reports]
            ),
            "by_task_type": {
                task_type: {
                    "tasks_per_seed": seed_reports[0]["by_task_type"][task_type]["tasks"],
                    "baseline_sr_percent": mean_std(
                        [
                            100 * run["by_task_type"][task_type]["baseline_sr"]
                            for run in seed_reports
                        ]
                    ),
                    "jitmem_sr_percent": mean_std(
                        [100 * run["by_task_type"][task_type]["jitmem_sr"] for run in seed_reports]
                    ),
                    "delta_pp": mean_std(
                        [run["by_task_type"][task_type]["delta_pp"] for run in seed_reports]
                    ),
                }
                for task_type in types
            },
        },
        "statistical_scope": {
            "std_definition": "sample standard deviation across seeds (ddof=1)",
            "unique_task_count": expected_tasks,
            "repeated_episode_pairs": expected_tasks * len(baseline.seeds),
            "note": "Seeds repeat the same task set and memory couples tasks within a stream. Counts and mean/std are descriptive; no independence assumption or pooled task-level significance claim is made.",
            "pooled_significance_test": None,
        },
        "interpretation": {
            "outcome_source": "native ALFWorld verifier, never the executor judge verdict",
            "model_identity_source": "declared API model configuration; not author model/checkpoint equivalence",
            "memory_gate": "judge false positives can store native-failed trajectories; judge false negatives can exclude native-successful trajectories",
            "cold_start": "curator is called with empty context and may provide general guidance; empty-bank episodes do not demonstrate use of retrieved experience",
            "usage": "missing input/output token usage remains null; no cost estimate is inferred",
        },
    }


def link(path: str, label: str, line: int | None = None) -> str:
    return f"[{label}](<{path}{f':{line}' if line else ''}>)"


def shown(value: Any) -> str:
    return "未知" if value is None else str(value)


def render_markdown(report: dict) -> str:
    overall = report["across_seed"]
    delta = overall["delta_pp"]
    lines = [
        "# ALFWorld 成对评测报告",
        "",
        f"本次 matched API 评测中，prompted JITMEM 相对 no-memory 的原生 SR 差为 **{delta['mean']:+.2f} ± {delta['std']:.2f} 个百分点**。"
        f"split 为 `{report['split']}`，每轮 {report['tasks_per_seed']} 个任务，"
        f"任务顺序种子为 {report['seeds']}。标准差按种子计算，ddof=1。",
        "",
        f"Executor：`{report['executor_model']}`；curator：`{report['curator_model']}`。"
        "模型名来自 API 配置。当前复现使用 prompted curator 和语义改写 prompts，没有本地训练。"
        "[原论文](https://arxiv.org/pdf/2609.27334)的 executor 为 Qwen3-8B、Gemini-2.5-Pro、GPT-5.4，"
        "并包含 RL-trained curator；本次结果只适用于所声明的模型与实现，不能视为作者训练后数字的等价复现。",
        "",
        f"两组均使用 {report['protocol']['workers']} 个 worker，"
        f"batch size 为 {report['protocol']['batch_size']}；"
        f"启动方式为 {report['protocol']['process_start_method'] or '串行'}。"
        "已核对种子确定的任务顺序、批内冻结记忆和整批完成后的顺序提交；异步完成顺序不会重排结果。",
        "",
        "同一批任务在不同种子下重复出现。下述差值和分布是描述性结果，"
        f"每组共 {report['statistical_scope']['repeated_episode_pairs']} 个 episode，"
        f"覆盖 {report['statistical_scope']['unique_task_count']} 个唯一任务；"
        "不能把多轮重复合并成独立任务样本。没有计算 pooled p-value。",
        "",
        f"no-memory 平均 SR：{overall['baseline_sr_percent']['mean']:.2f} ± "
        f"{overall['baseline_sr_percent']['std']:.2f}%；JITMEM 平均 SR："
        f"{overall['jitmem_sr_percent']['mean']:.2f} ± {overall['jitmem_sr_percent']['std']:.2f}%。",
        "",
        "## 每轮结果",
        "",
        "| Seed | no-memory SR | JITMEM SR | Δ pp | Both success | Baseline only | JITMEM only | Both failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["per_seed"]:
        counts = run["paired_counts"]
        lines.append(
            f"| {run['seed']} | {100 * run['baseline']['success_rate']:.2f}% | "
            f"{100 * run['jitmem']['success_rate']:.2f}% | {run['delta_pp']:+.2f} | "
            f"{counts['both_success']} | {counts['baseline_only']} | {counts['jitmem_only']} | {counts['both_failure']} |"
        )
    lines.extend(
        [
            "",
            "## 任务类型",
            "",
            "| Type | Tasks / seed | no-memory SR mean ± std | JITMEM SR mean ± std | Δ pp mean ± std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_type, metrics in overall["by_task_type"].items():
        base, jit, change = (
            metrics[key] for key in ("baseline_sr_percent", "jitmem_sr_percent", "delta_pp")
        )
        lines.append(
            f"| {task_type} | {metrics['tasks_per_seed']} | {base['mean']:.2f} ± {base['std']:.2f} | {jit['mean']:.2f} ± {jit['std']:.2f} | {change['mean']:+.2f} ± {change['std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 交互、生成和 judge",
            "",
            "| Seed / arm | Mean decisions | Mean env steps | Invalid actions | Truncated episodes | Incomplete calls | Judge TP / FP / FN / TN | Judge parse / generation errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for run in report["per_seed"]:
        for arm in ("baseline", "jitmem"):
            metrics = run[arm]
            judge = metrics["native_vs_judge"]
            matrix = (
                " / ".join(
                    str(judge[key])
                    for key in (
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "true_negative",
                    )
                )
                if judge["labeled_episodes"]
                else "未调用 judge"
            )
            lines.append(
                f"| {run['seed']} / {arm} | {metrics['mean_decisions']:.2f} | {metrics['mean_environment_steps']:.2f} | {metrics['invalid_actions']} | {metrics['truncated_episodes']} | {metrics['generation_failures']['calls']} | {matrix} | {judge['parse_errors']} / {judge['generation_errors']} |"
            )
    lines.extend(
        [
            "",
            "任务 SR 来自原生 verifier。FP 表示原生失败而 judge 判成功，FN 表示原生成功而 judge 判失败；"
            "这些判定影响后续记忆写入，而不是更改评测标签。生成失败角色和 finish reasons 可在 comparison.json 的每轮结果中查看。",
            "",
            "| Seed | Empty-bank episodes | Stored episodes | Native-failed stored episodes |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for run in report["per_seed"]:
        metrics = run["jitmem"]
        lines.append(
            f"| {run['seed']} | {metrics['empty_bank_episodes']} | {metrics['stored_episodes']} | {metrics['native_failed_stored_episodes']} |"
        )
    false_positive_count = sum(
        run["jitmem"]["native_vs_judge"]["false_positive"] for run in report["per_seed"]
    )
    false_negative_count = sum(
        run["jitmem"]["native_vs_judge"]["false_negative"] for run in report["per_seed"]
    )
    native_failed_stored_count = sum(
        run["jitmem"]["native_failed_stored_episodes"] for run in report["per_seed"]
    )
    lines.extend(
        [
            "",
            f"全部重复 episode 中观测到 judge FP={false_positive_count}、FN={false_negative_count}；"
            f"其中 {native_failed_stored_count} 条原生失败轨迹被存入记忆。"
            "这些是本次执行记录的计数，不代表独立任务样本。逐任务证据见后文。",
            "",
            "curator 在空库时也会被调用，并允许给出一般操作指导。这是公开的实现选择，"
            "空库 episode 的效果不能作为使用历史检索经验的证据。",
            "",
            "## 按角色 token 用量",
            "",
            "缺失 usage 显示为未知，不补成 0，也不据此推算费用。这里分别报告各角色，避免把 executor token 变化当作全系统成本变化。",
            "",
            "| Seed / arm | Role | Calls | Input tokens | Output tokens |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in report["per_seed"]:
        for arm in ("baseline", "jitmem"):
            for role, usage in run[arm]["usage_by_role"].items():
                lines.append(
                    f"| {run['seed']} / {arm} | {role} | {usage['calls']} | {shown(usage['prompt_tokens'])} | {shown(usage['completion_tokens'])} |"
                )
    lines.extend(["", "## 证据来源", ""])
    for arm, source in report["sources"].items():
        lines.append(
            f"{arm}：{link(str(Path(source['root']) / 'manifest.json'), 'manifest')}，"
            f"{link(str(Path(source['root']) / 'summary.json'), 'summary')}。"
            f"fingerprint：`{source['fingerprint']}`。"
        )
        lines.append("")
    lines.extend(
        [
            "逐任务成对记录及所有输入文件 SHA-256 见 comparison.json。下面每个不一致 outcome 链接到该任务的 results.jsonl 行；完整 episode 在 JSON source 字段中给出。",
            "",
            "| Seed | Task | Outcome | no-memory evidence | JITMEM evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    disagreements = 0
    for run in report["per_seed"]:
        for pair in run["task_pairs"]:
            if pair["outcome"] not in {"baseline_only", "jitmem_only"}:
                continue
            disagreements += 1
            first, second = pair["sources"]["baseline"], pair["sources"]["jitmem"]
            task_id = pair["task_id"].replace("|", "\\|")
            lines.append(
                f"| {run['seed']} | {task_id} | {pair['outcome']} | {link(first['results_jsonl'], 'result', first['line'])} | {link(second['results_jsonl'], 'result', second['line'])} |"
            )
    if not disagreements:
        lines.append("| — | 无成对分歧 | — | — | — |")
    lines.extend(
        [
            "",
            "## Judge 判定不一致的任务",
            "",
            "包括两组原生结果相同但 JITMEM judge 与原生判定不一致的任务；该表不按两组 SR 的差异筛选。"
            "FP/FN 以原生结果为参照，分歧也可能来自原始游戏状态与可观察轨迹的差异，需结合轨迹分析。",
            "",
            "| Seed | Task | Disagreement | Native success | Judge success | Stored | JITMEM evidence |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    judge_disagreements = 0
    for run in report["per_seed"]:
        for pair in run["task_pairs"]:
            if pair["jitmem_success"] == pair["jitmem_judge_success"]:
                continue
            judge_disagreements += 1
            source = pair["sources"]["jitmem"]
            task_id = pair["task_id"].replace("|", "\\|")
            error = "FN" if pair["jitmem_success"] else "FP"
            lines.append(
                f"| {run['seed']} | {task_id} | {error} | {pair['jitmem_success']} | {pair['jitmem_judge_success']} | {pair['jitmem_stored']} | {link(source['results_jsonl'], 'result', source['line'])} |"
            )
    if not judge_disagreements:
        lines.append("| — | 未观察到 native / judge 判定不一致 | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Completed no-memory output directory")
    parser.add_argument("jitmem", type=Path, help="Completed prompted JITMEM output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument(
        "--expected-tasks", type=int, default=140, help="Explicit full manifest count"
    )
    args = parser.parse_args(argv)
    try:
        report = build_comparison(args.baseline, args.jitmem, expected_tasks=args.expected_tasks)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = output / "comparison.json", output / "comparison.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved {json_path}\nSaved {markdown_path}")
    except (AnalysisError, OSError) as error:
        parser.exit(1, f"Analysis rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
