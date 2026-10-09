#!/usr/bin/env python3
"""Compare matched native prompted-JITMEM judge-filtered and label-annotated full storage.

Reads completed local artifacts only; never calls a model or changes evaluation code.
Defaults require valid_seen140, seeds0/1/2, workers10, frozen batches10 and cold starts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Load the sibling's existing stdlib-only validators without changing its interface.
_SPEC = importlib.util.spec_from_file_location(
    "_jitmem_storage_analysis_helpers", Path(__file__).with_name("analyze_results.py")
)
_COMMON = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COMMON
_SPEC.loader.exec_module(_COMMON)
AnalysisError = _COMMON.AnalysisError
require, integer, read_json = _COMMON.require, _COMMON.integer, _COMMON.read_json
load_arm, arm_metrics, mean_std = _COMMON.load_arm, _COMMON.arm_metrics, _COMMON.mean_std
link, shown = _COMMON.link, _COMMON.shown
nullable_total = _COMMON.nullable_total

SEEN_TASK_COUNTS = {
    "pick_and_place_simple": 35,
    "look_at_obj_in_light": 13,
    "pick_clean_then_place_in_recep": 27,
    "pick_heat_then_place_in_recep": 16,
    "pick_cool_then_place_in_recep": 25,
    "pick_two_obj_and_place": 24,
}


def validate_complete_seen_manifest(tasks: dict, environment: dict, label: str) -> None:
    require(
        environment.get("task_types") == [1, 2, 3, 4, 5, 6],
        f"{label}: full seen140 requires all six task_types",
    )
    require(
        type(environment.get("annotation_index")) is int and environment["annotation_index"] == 0,
        f"{label}: full seen140 requires annotation_index=0",
    )
    require(
        Counter(task["task_type"] for task in tasks.values()) == SEEN_TASK_COUNTS,
        f"{label}: full seen140 task-type counts disagree with the declared protocol",
    )


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        require(all(line.strip() for line in lines), f"{path}: blank JSONL lines")
        records = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Cannot read complete JSONL: {path}") from error
    require(all(isinstance(item, dict) for item in records), f"{path}: invalid JSONL record")
    return records


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def render_trajectory(memory: dict) -> str:
    lines = [f"OBSERVATION 0: {memory['initial_observation']}"]
    for index, turn in enumerate(memory["turns"], 1):
        lines.extend(
            [
                f"ACTION {index}: {turn['action']}",
                f"OBSERVATION {index}: {turn['next_observation']}",
            ]
        )
    return "\n".join(lines)


def memory_context(references: list[dict], memory: dict[str, dict], policy: str) -> str:
    parts = []
    for index, reference in enumerate(references, 1):
        entry = memory[reference["task_id"]]
        label = (
            f"\nExecutor judge label: {'success' if entry['judge_success'] else 'failure'}"
            if policy == "all"
            else ""
        )
        parts.append(
            f"Memory {index}:\nQuestion: {entry['task_description']}{label}\n"
            f"Trajectory:\n{render_trajectory(entry)}"
        )
    return "\n\n".join(parts) or "No past episodes are available."


def validate_call_usage(episode: dict, episode_path: Path) -> None:
    """Require aggregate usage to equal recorded calls without inventing unknown counts."""
    calls = episode["calls"]
    fields = ("prompt_tokens", "completion_tokens")
    for call in calls:
        for field in fields:
            require(field in call, f"{episode_path}: recorded call {field} missing")
            if call[field] is not None:
                integer(call[field], f"{episode_path}: recorded call {field}")
    expected = {field: nullable_total([call[field] for call in calls]) for field in fields}
    expected["by_role"] = {
        role: {
            "calls": sum(call["role"] == role for call in calls),
            **{
                field: nullable_total([call[field] for call in calls if call["role"] == role])
                for field in fields
            },
        }
        for role in sorted({call["role"] for call in calls})
    }
    usage = episode.get("usage")
    require(
        isinstance(usage, dict) and all(usage.get(key) == value for key, value in expected.items()),
        f"{episode_path}: aggregate usage disagrees with recorded calls",
    )
    if "complete" in usage:
        require(
            usage["complete"] is all(expected[field] is not None for field in fields),
            f"{episode_path}: usage completeness disagrees with recorded calls",
        )


def validate_protocol(filtered, full, expected_seeds, expected_workers, expected_batch_size):
    for arm, policy in ((filtered, "judge"), (full, "all")):
        config = arm.manifest["config"]
        experiment = config["experiment"]
        require(arm.summary["split"] == "valid_seen", f"{arm.root}: expected valid_seen split")
        if len(arm.tasks) == 140:
            validate_complete_seen_manifest(arm.tasks, config["environment"], str(arm.root))
        require(arm.seeds == list(expected_seeds), f"{arm.root}: expected seeds {expected_seeds}")
        require(experiment.get("store_policy") == policy, f"{arm.root}: expected policy {policy}")
        require(experiment["workers"] == expected_workers, f"{arm.root}: unexpected workers")
        require(
            experiment["batch_size"] == expected_batch_size, f"{arm.root}: unexpected batch_size"
        )
        require(
            experiment.get("warm_start") is None, f"{arm.root}: warm start is not this ablation"
        )
        require(
            experiment.get("task_adaptive") is True, f"{arm.root}: task-adaptive curator required"
        )
        for field, expected in (("retrieval_k", 3), ("max_steps", 30), ("history_window", 3)):
            require(experiment.get(field) == expected, f"{arm.root}: expected {field}={expected}")
        for role in ("executor", "curator"):
            require(bool(config.get(role, {}).get("model")), f"{arm.root}: {role} model missing")
        for field in ("source_hashes", "packages", "bm25", "python", "platform", "prompt_source"):
            require(bool(arm.manifest.get(field)), f"{arm.root}: {field} metadata missing")
        require(
            isinstance(arm.manifest.get("fingerprint"), str) and bool(arm.manifest["fingerprint"]),
            f"{arm.root}: fingerprint missing",
        )
    normalized = []
    for arm in (filtered, full):
        manifest = copy.deepcopy(arm.manifest)
        manifest.pop("fingerprint", None)
        experiment = manifest["config"]["experiment"]
        experiment.pop("store_policy")
        experiment.pop("output_dir", None)
        normalized.append(manifest)
    require(
        normalized[0] == normalized[1],
        "Matched manifests/config mismatch beyond store_policy and output_dir",
    )
    for seed in filtered.seeds:
        require(filtered.orders[seed] == full.orders[seed], f"Task order mismatch: seed {seed}")


def validate_storage(arm, seed: int, policy: str) -> tuple[dict, list[dict]]:
    """Verify persisted bank, observed labels and retrieval from prior complete batches."""
    rows = arm.results[seed]
    run_dir = arm.root / f"seed_{seed}"
    memory_path, checkpoint_path = run_dir / "memory.jsonl", run_dir / "checkpoint.json"
    memory = read_jsonl(memory_path)
    expected_ids = [row["task_id"] for row in rows if row["stored"]]
    require(
        [item.get("task_id") for item in memory] == expected_ids,
        f"{run_dir}: final bank order/count mismatch",
    )
    entries = {item["task_id"]: item for item in memory}
    origin = {row["task_id"]: row for row in rows}
    episodes = []
    curator_systems = []
    for index, row in enumerate(rows):
        episode_path = run_dir / "episodes" / f"{index:04d}.json"
        episode = read_json(episode_path)
        require(isinstance(episode, dict), f"{episode_path}: invalid episode")
        compact = {
            key: value for key, value in episode.items() if key not in {"calls", "trajectory"}
        }
        require(compact == row, f"{episode_path}: episode differs from committed result")
        expected_stored = policy == "all" or row["judge"]["success"]
        require(
            row["stored"] == expected_stored,
            f"{episode_path}: storage gate disagrees with policy/label",
        )
        calls = episode.get("calls")
        require(
            isinstance(calls, list) and all(isinstance(call, dict) for call in calls),
            f"{episode_path}: calls missing",
        )
        require(
            [call.get("role") for call in calls]
            == ["curator", *(["executor"] * row["steps"]), "judge"],
            f"{episode_path}: role calls disagree with episode decisions",
        )
        validate_call_usage(episode, episode_path)
        for call in calls:
            require(
                isinstance(call.get("messages"), list)
                and all(isinstance(message, dict) for message in call["messages"]),
                f"{episode_path}: invalid recorded messages",
            )
        curator_messages = calls[0]["messages"]
        require(
            len(curator_messages) == 2
            and curator_messages[0].get("role") == "system"
            and isinstance(curator_messages[0].get("content"), str)
            and curator_messages[1].get("role") == "user",
            f"{episode_path}: curator messages missing",
        )
        curator_systems.append(curator_messages[0]["content"])
        require(
            isinstance(episode.get("trajectory"), dict), f"{episode_path}: raw trajectory missing"
        )
        if row["stored"]:
            entry = entries[row["task_id"]]
            require(
                set(entry)
                == {
                    "task_id",
                    "task_description",
                    "task_type",
                    "split",
                    "initial_observation",
                    "turns",
                    "judge_success",
                    "summary",
                },
                f"{episode_path}: bank is not lossless raw-memory schema",
            )
            require(
                type(entry.get("judge_success")) is bool
                and entry["judge_success"] == row["judge"]["success"],
                f"{episode_path}: stored judge label disagrees",
            )
            require(
                entry["task_description"] == row["task_description"]
                and entry["task_type"] == row["task_type"]
                and entry["split"] == row["split"]
                and entry["initial_observation"] == episode["trajectory"].get("initial_observation")
                and entry["turns"] == episode["trajectory"].get("turns")
                and entry["summary"] is None,
                f"{episode_path}: stored raw trajectory differs from episode",
            )
        episodes.append(episode)
        arm.evidence_files.append(episode_path)
    checkpoint = read_json(checkpoint_path)
    require(
        isinstance(checkpoint, dict)
        and checkpoint.get("fingerprint") == arm.manifest["fingerprint"]
        and checkpoint.get("results") == episodes
        and checkpoint.get("bank") == memory,
        f"{run_dir}: final checkpoint differs from complete results/bank",
    )
    arm.evidence_files.extend([memory_path, checkpoint_path])
    batch_size = arm.manifest["config"]["experiment"]["batch_size"]
    retrieval_k = arm.manifest["config"]["experiment"]["retrieval_k"]
    available = []
    growth = []
    retrieval_counts = Counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        batch_counts = Counter()
        for index, row in enumerate(batch, offset):
            references = row.get("retrieved")
            require(
                isinstance(references, list)
                and all(isinstance(item, dict) for item in references)
                and len(references) == min(retrieval_k, len(available)),
                f"{run_dir}: retrieved count disagrees with frozen bank",
            )
            ids = [item.get("task_id") for item in references]
            require(
                all(isinstance(task_id, str) and task_id in available for task_id in ids)
                and len(set(ids)) == len(ids),
                f"{run_dir}: retrieval contains unavailable/current/future/duplicate memory",
            )
            for reference in references:
                score = reference.get("score")
                require(
                    type(score) in {int, float} and math.isfinite(score) and score >= 0,
                    f"{run_dir}: invalid retrieval score",
                )
            expected_user = (
                f"Question: {row['task_description']}\n\nRetrieved memories:\n"
                + memory_context(references, entries, policy)
            )
            require(
                episodes[index]["calls"][0]["messages"][1].get("content") == expected_user,
                f"{run_dir}: curator memory input/visible labels disagree with policy",
            )
            batch_counts["references"] += len(ids)
            batch_counts["native_failed_references"] += sum(
                not origin[task_id]["success"] for task_id in ids
            )
            batch_counts["judge_failed_references"] += sum(
                not origin[task_id]["judge"]["success"] for task_id in ids
            )
            batch_counts["episodes_with_retrieval"] += bool(ids)
            batch_counts["episodes_with_native_failed_retrieval"] += any(
                not origin[task_id]["success"] for task_id in ids
            )
            batch_counts["episodes_with_judge_failed_retrieval"] += any(
                not origin[task_id]["judge"]["success"] for task_id in ids
            )
        stored = [row for row in batch if row["stored"]]
        growth.append(
            {
                "batch_index": offset // batch_size,
                "tasks": len(batch),
                "success_rate": statistics.mean(row["success"] for row in batch),
                "bank_before": len(available),
                "new_entries": len(stored),
                "bank_after": len(available) + len(stored),
                "native_failed_new_entries": sum(not row["success"] for row in stored),
                "judge_failed_new_entries": sum(not row["judge"]["success"] for row in stored),
                "retrieval": dict(batch_counts),
            }
        )
        available.extend(row["task_id"] for row in stored)
        retrieval_counts.update(batch_counts)
    stored_rows = [row for row in rows if row["stored"]]
    native_failed = sum(not row["success"] for row in stored_rows)
    judge_failed = sum(not row["judge"]["success"] for row in stored_rows)
    references = retrieval_counts["references"]
    return {
        "stored_entries": len(memory),
        "final_bank_size": len(memory),
        "native_failed_stored_entries": native_failed,
        "judge_failed_stored_entries": judge_failed,
        "native_failed_stored_fraction": ratio(native_failed, len(memory)),
        "judge_failed_stored_fraction": ratio(judge_failed, len(memory)),
        "batch_growth": growth,
        "retrieval": {
            **{
                key: retrieval_counts[key]
                for key in (
                    "references",
                    "native_failed_references",
                    "judge_failed_references",
                    "episodes_with_retrieval",
                    "episodes_with_native_failed_retrieval",
                    "episodes_with_judge_failed_retrieval",
                )
            },
            "native_failed_fraction": ratio(
                retrieval_counts["native_failed_references"], references
            ),
            "judge_failed_fraction": ratio(retrieval_counts["judge_failed_references"], references),
        },
    }, curator_systems


def build_comparison(
    filtered_dir,
    full_dir,
    *,
    expected_tasks=140,
    expected_seeds=(0, 1, 2),
    expected_workers=10,
    expected_batch_size=10,
) -> dict:
    integer(expected_tasks, "expected_tasks", 1)
    integer(expected_workers, "expected_workers", 1)
    integer(expected_batch_size, "expected_batch_size", 1)
    filtered = load_arm(filtered_dir, "jitmem", expected_tasks)
    full = load_arm(full_dir, "jitmem", expected_tasks)
    validate_protocol(filtered, full, expected_seeds, expected_workers, expected_batch_size)
    per_seed = []
    types = sorted({task["task_type"] for task in filtered.tasks.values()})
    for seed in filtered.seeds:
        storage = {}
        systems = {}
        for name, arm, policy in (("filtered", filtered, "judge"), ("full", full, "all")):
            storage[name], systems[name] = validate_storage(arm, seed, policy)
        require(
            systems["filtered"] == systems["full"], f"Curator system prompt mismatch: seed {seed}"
        )
        first, second = filtered.results[seed], full.results[seed]
        counts = dict.fromkeys(("both_success", "filtered_only", "full_only", "both_failure"), 0)
        pairs = []
        for index, (left, right) in enumerate(zip(first, second), 1):
            require(
                left["task_description"] == right["task_description"],
                f"Observed task mismatch: seed {seed}, {left['task_id']}",
            )
            outcome = (
                "both_success"
                if left["success"] and right["success"]
                else "filtered_only"
                if left["success"]
                else "full_only"
                if right["success"]
                else "both_failure"
            )
            counts[outcome] += 1
            pairs.append(
                {
                    "task_id": left["task_id"],
                    "task_type": left["task_type"],
                    "outcome": outcome,
                    "filtered_success": left["success"],
                    "full_success": right["success"],
                    "filtered_judge_success": left["judge"]["success"],
                    "full_judge_success": right["judge"]["success"],
                    "filtered_stored": left["stored"],
                    "full_stored": right["stored"],
                    "sources": {
                        name: {
                            "results_jsonl": str(arm.root / f"seed_{seed}/results.jsonl"),
                            "line": index,
                            "episode_json": str(
                                arm.root / f"seed_{seed}/episodes/{index - 1:04d}.json"
                            ),
                        }
                        for name, arm in (("filtered", filtered), ("full", full))
                    },
                }
            )
        metrics = {
            name: {**arm_metrics(rows), "storage": storage[name]}
            for name, rows in (("filtered", first), ("full", second))
        }
        by_type = {}
        for task_type in types:
            left = [row for row in first if row["task_type"] == task_type]
            right = [row for row in second if row["task_type"] == task_type]
            left_sr, right_sr = (
                statistics.mean(row["success"] for row in group) for group in (left, right)
            )
            by_type[task_type] = {
                "tasks": len(left),
                "filtered_sr": left_sr,
                "full_sr": right_sr,
                "delta_pp": 100 * (left_sr - right_sr),
            }
        per_seed.append(
            {
                "seed": seed,
                "tasks": len(first),
                "paired_counts": counts,
                **metrics,
                "delta_pp": 100
                * (metrics["filtered"]["success_rate"] - metrics["full"]["success_rate"]),
                "mean_decision_delta": metrics["filtered"]["mean_decisions"]
                - metrics["full"]["mean_decisions"],
                "mean_environment_step_delta": metrics["filtered"]["mean_environment_steps"]
                - metrics["full"]["mean_environment_steps"],
                "by_task_type": by_type,
                "task_pairs": pairs,
            }
        )
    sources = {
        name: {
            "root": str(arm.root),
            "fingerprint": arm.manifest["fingerprint"],
            "files": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in arm.evidence_files
            ],
        }
        for name, arm in (("filtered", filtered), ("full", full))
    }
    overall = {}
    for name in ("filtered", "full"):
        role_names = sorted({role for run in per_seed for role in run[name]["usage_by_role"]})
        roles = {}
        for role in role_names:
            usage = [run[name]["usage_by_role"][role] for run in per_seed]
            roles[role] = {
                "calls": sum(piece["calls"] for piece in usage),
                "prompt_tokens": nullable_total([piece["prompt_tokens"] for piece in usage]),
                "completion_tokens": nullable_total(
                    [piece["completion_tokens"] for piece in usage]
                ),
            }
        input_tokens = nullable_total([run[name]["prompt_tokens"] for run in per_seed])
        output_tokens = nullable_total([run[name]["completion_tokens"] for run in per_seed])
        overall[name] = {
            "episodes": expected_tasks * len(per_seed),
            "calls": sum(piece["calls"] for piece in roles.values()),
            "all_role_prompt_tokens": input_tokens,
            "all_role_completion_tokens": output_tokens,
            "all_role_input_output_tokens": nullable_total([input_tokens, output_tokens]),
            "usage_by_role": roles,
        }
    return {
        "kind": "matched_native_alfworld_storage_ablation",
        "benchmark": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "valid_seen",
        "tasks_per_seed": expected_tasks,
        "seeds": filtered.seeds,
        "executor_model": filtered.manifest["config"]["executor"]["model"],
        "curator_model": filtered.manifest["config"]["curator"]["model"],
        "variant": "API prompted curator with semantic paraphrases; no local training; not author RL-trained/model-equivalent results",
        "difference_definition": "filtered minus label-annotated full storage",
        "protocol": {
            "workers": expected_workers,
            "batch_size": expected_batch_size,
            "process_start_method": filtered.manifest["process_start_method"],
            "matched_config_except": ["experiment.store_policy", "experiment.output_dir"],
            "filtered_policy": "judge",
            "full_policy": "all",
            "cold_start": True,
            "filtered_labels_visible": False,
            "full_labels_visible": True,
            "matched_config": filtered.manifest["config"],
        },
        "sources": sources,
        "per_seed": per_seed,
        "overall": overall,
        "across_seed": {
            "filtered_sr_percent": mean_std(
                [100 * run["filtered"]["success_rate"] for run in per_seed]
            ),
            "full_sr_percent": mean_std([100 * run["full"]["success_rate"] for run in per_seed]),
            "delta_pp": mean_std([run["delta_pp"] for run in per_seed]),
            "filtered_mean_decisions": mean_std(
                [run["filtered"]["mean_decisions"] for run in per_seed]
            ),
            "full_mean_decisions": mean_std([run["full"]["mean_decisions"] for run in per_seed]),
            "filtered_mean_environment_steps": mean_std(
                [run["filtered"]["mean_environment_steps"] for run in per_seed]
            ),
            "full_mean_environment_steps": mean_std(
                [run["full"]["mean_environment_steps"] for run in per_seed]
            ),
            "mean_decision_delta": mean_std([run["mean_decision_delta"] for run in per_seed]),
            "mean_environment_step_delta": mean_std(
                [run["mean_environment_step_delta"] for run in per_seed]
            ),
            "by_task_type": {
                task_type: {
                    "tasks_per_seed": per_seed[0]["by_task_type"][task_type]["tasks"],
                    "filtered_sr_percent": mean_std(
                        [100 * run["by_task_type"][task_type]["filtered_sr"] for run in per_seed]
                    ),
                    "full_sr_percent": mean_std(
                        [100 * run["by_task_type"][task_type]["full_sr"] for run in per_seed]
                    ),
                    "delta_pp": mean_std(
                        [run["by_task_type"][task_type]["delta_pp"] for run in per_seed]
                    ),
                }
                for task_type in types
            },
        },
        "statistical_scope": {
            "std_definition": "sample standard deviation across seeds, ddof=1",
            "unique_task_count": expected_tasks,
            "repeated_episode_pairs": expected_tasks * len(filtered.seeds),
            "pooled_significance_test": None,
            "note": "Repeated tasks and memory-coupled streams are not independent observations; counts and mean/std are descriptive.",
        },
    }


def fraction(value):
    return "—（无分母）" if value is None else f"{100 * value:.2f}%"


def render_markdown(report: dict) -> str:
    aggregate = report["across_seed"]
    delta = aggregate["delta_pp"]
    lines = [
        "# ALFWorld 存储策略消融",
        "",
        f"成功过滤存储（filtered）相对带标签的全部存储（full）的原生 SR 差为 **{delta['mean']:+.2f} ± {delta['std']:.2f} 个百分点**（filtered-minus-full）。正值有利于 filtered，负值有利于 full；不预设论文结论在本配置下成立。",
        "",
        f"`valid_seen` 每轮 {report['tasks_per_seed']} 个任务，种子 {report['seeds']}；workers={report['protocol']['workers']}，batch={report['protocol']['batch_size']}，启动方式={report['protocol']['process_start_method'] or 'serial'}。两组仅改变 store_policy 和输出目录，均为 jitmem；已经核对相同配置、源代码、prompt、运行时、数据、任务顺序及整批冻结提交。",
        "",
        f"Executor：`{report['executor_model']}`；curator：`{report['curator_model']}`（API 配置模型名）。这是使用语义改写 prompts 的 API prompted variant，没有本地训练。[原论文](https://arxiv.org/pdf/2609.27334)的这项存储消融使用未训练的 JITMEM-base；本次复现采用所声明的 API 模型与实现，既不能视为原模型下消融数值的等价复现，也不能对应 RL-trained JITMEM 的 headline 性能。",
        "",
        "filtered 使用 executor judge gate，只保留 judge success 的原始轨迹，curator 看不到显式 outcome 标签；full 保存所有轨迹，并向 curator 显示同一 executor judge 的 success/failure 标签。标签不是原生 verifier 标签；原生 SR 独立于 judge 计算。",
        "",
        f"每组有 {report['statistical_scope']['repeated_episode_pairs']} 个重复 episode，覆盖 {report['statistical_scope']['unique_task_count']} 个唯一任务。种子间标准差为 ddof=1；记忆会关联同一轮任务，不把重复 episode 当作独立样本，不计算 pooled p-value。",
        "",
        f"Filtered 平均 SR：{aggregate['filtered_sr_percent']['mean']:.2f} ± {aggregate['filtered_sr_percent']['std']:.2f}%；full 平均 SR：{aggregate['full_sr_percent']['mean']:.2f} ± {aggregate['full_sr_percent']['std']:.2f}%。平均决策数差（filtered-minus-full）：{aggregate['mean_decision_delta']['mean']:+.2f} ± {aggregate['mean_decision_delta']['std']:.2f}；平均环境步数差：{aggregate['mean_environment_step_delta']['mean']:+.2f} ± {aggregate['mean_environment_step_delta']['std']:.2f}。",
        "",
        "## 每轮成对结果",
        "",
        "| Seed | Filtered SR | Full SR | Δ pp | Both success | Filtered only | Full only | Both failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["per_seed"]:
        counts = run["paired_counts"]
        lines.append(
            f"| {run['seed']} | {100 * run['filtered']['success_rate']:.2f}% | {100 * run['full']['success_rate']:.2f}% | {run['delta_pp']:+.2f} | {counts['both_success']} | {counts['filtered_only']} | {counts['full_only']} | {counts['both_failure']} |"
        )
    lines.extend(
        [
            "",
            "## 任务类型",
            "",
            "| Type | Tasks / seed | Filtered SR mean ± std | Full SR mean ± std | Δ pp mean ± std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_type, metrics in aggregate["by_task_type"].items():
        left, right, change = (
            metrics[key] for key in ("filtered_sr_percent", "full_sr_percent", "delta_pp")
        )
        lines.append(
            f"| {task_type} | {metrics['tasks_per_seed']} | {left['mean']:.2f} ± {left['std']:.2f} | {right['mean']:.2f} ± {right['std']:.2f} | {change['mean']:+.2f} ± {change['std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 交互与判定",
            "",
            "| Seed / arm | Mean decisions / env steps | Invalid / truncated | Incomplete calls | Judge TP / FP / FN / TN | Judge parse / generation errors |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for run in report["per_seed"]:
        for name in ("filtered", "full"):
            metrics = run[name]
            judge = metrics["native_vs_judge"]
            matrix = " / ".join(
                str(judge[key])
                for key in ("true_positive", "false_positive", "false_negative", "true_negative")
            )
            lines.append(
                f"| {run['seed']} / {name} | {metrics['mean_decisions']:.2f} / {metrics['mean_environment_steps']:.2f} | {metrics['invalid_actions']} / {metrics['truncated_episodes']} | {metrics['generation_failures']['calls']} | {matrix} | {judge['parse_errors']} / {judge['generation_errors']} |"
            )
    lines.extend(
        [
            "",
            "FP 是原生失败但 judge 判成功，FN 是原生成功但 judge 判失败。filtered 也可能存入 judge FP；full 中 judge failure 也可能是原生成功。下述失败比例分别用原生结果和 judge 标签计算，避免混淆。",
            "",
            "## 记忆增长与失败经验",
            "",
            "| Seed / arm | Final bank | Native-failed stored | Judge-failed stored | Retrieval references | Native-failed retrieved | Judge-failed retrieved |",
            "| --- | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for run in report["per_seed"]:
        for name in ("filtered", "full"):
            storage = run[name]["storage"]
            retrieval = storage["retrieval"]
            lines.append(
                f"| {run['seed']} / {name} | {storage['final_bank_size']} | {storage['native_failed_stored_entries']} ({fraction(storage['native_failed_stored_fraction'])}) | {storage['judge_failed_stored_entries']} ({fraction(storage['judge_failed_stored_fraction'])}) | {retrieval['references']} | {retrieval['native_failed_references']} ({fraction(retrieval['native_failed_fraction'])}) | {retrieval['judge_failed_references']} ({fraction(retrieval['judge_failed_fraction'])}) |"
            )
    lines.extend(
        [
            "",
            "存储比例分母为该轮 bank 条目数；检索比例分母为该轮检索引用次数，同一轨迹可被多次引用。无分母显示为 —。每批 bank before/after、新增失败条目、SR 与检索构成详见 JSON 的 batch_growth。curator 在空库时仍可提供一般指导，空库任务不证明使用了历史经验。",
            "",
            "## 按角色 token 用量",
            "",
            "缺失 usage 保留 null 并显示未知，不补成 0，不推算费用。统计包含 executor、curator 和 judge，避免将 executor-only 变化当作系统总成本变化。",
            "",
            "| Seed / arm | Role | Calls | Input tokens | Output tokens |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in report["per_seed"]:
        for name in ("filtered", "full"):
            for role, usage in run[name]["usage_by_role"].items():
                lines.append(
                    f"| {run['seed']} / {name} | {role} | {usage['calls']} | {shown(usage['prompt_tokens'])} | {shown(usage['completion_tokens'])} |"
                )
    lines.extend(
        [
            "",
            "## 三轮汇总",
            "",
            "| Arm | SR mean ± std | Decisions mean ± std | Env steps mean ± std | Calls | All-role input tokens | All-role output tokens | Input + output tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("filtered", "full"):
        total = report["overall"][name]
        sr = aggregate[f"{name}_sr_percent"]
        decisions = aggregate[f"{name}_mean_decisions"]
        env_steps = aggregate[f"{name}_mean_environment_steps"]
        lines.append(
            f"| {name} | {sr['mean']:.2f} ± {sr['std']:.2f}% | {decisions['mean']:.2f} ± {decisions['std']:.2f} | {env_steps['mean']:.2f} ± {env_steps['std']:.2f} | {total['calls']} | {shown(total['all_role_prompt_tokens'])} | {shown(total['all_role_completion_tokens'])} | {shown(total['all_role_input_output_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "SR 与步数统计的标准差按种子计算；calls/tokens 为全部重复 episode 的实际用量汇总。任一必要用量缺失时，相应汇总为未知。",
        ]
    )
    lines.extend(["", "## 证据", ""])
    for name, source in report["sources"].items():
        lines.extend(
            [
                f"{name}：{link(str(Path(source['root']) / 'manifest.json'), 'manifest')}，{link(str(Path(source['root']) / 'summary.json'), 'summary')}；fingerprint=`{source['fingerprint']}`。",
                "",
            ]
        )
    lines.extend(
        [
            "输入文件（含每个 episode、checkpoint 与最终 bank）的 SHA-256 和所有成对记录见 storage_ablation.json。下面列出原生结果不同的任务。",
            "",
            "| Seed | Task | Outcome | Filtered evidence | Full evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    disagreements = 0
    for run in report["per_seed"]:
        for pair in run["task_pairs"]:
            if pair["outcome"] not in {"filtered_only", "full_only"}:
                continue
            disagreements += 1
            left, right = pair["sources"]["filtered"], pair["sources"]["full"]
            task_id = pair["task_id"].replace("|", "\\|")
            lines.append(
                f"| {run['seed']} | {task_id} | {pair['outcome']} | {link(left['results_jsonl'], 'result', left['line'])} | {link(right['results_jsonl'], 'result', right['line'])} |"
            )
    if not disagreements:
        lines.append("| — | 无成对分歧 | — | — | — |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filtered", type=Path, help="Completed jitmem store_policy=judge directory")
    parser.add_argument("full", type=Path, help="Completed jitmem store_policy=all directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/storage_ablation_comparison")
    )
    parser.add_argument("--expected-tasks", type=int, default=140)
    parser.add_argument("--expected-workers", type=int, default=10)
    parser.add_argument("--expected-batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = build_comparison(
            args.filtered,
            args.full,
            expected_tasks=args.expected_tasks,
            expected_workers=args.expected_workers,
            expected_batch_size=args.expected_batch_size,
        )
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = output / "storage_ablation.json", output / "storage_ablation.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved {json_path}\nSaved {markdown_path}")
    except (AnalysisError, OSError) as error:
        parser.exit(1, f"Storage ablation analysis rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
