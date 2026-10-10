#!/usr/bin/env python3
"""Compare completed paper-v1 full-storage runs with a fixed curator and judge.

This reads local artifacts, reuses the existing episode/bank/prompt/usage validators,
and makes no model or environment calls. Raw evidence and resolved configs remain
in ignored outputs; this report is not a public-safe export.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_jitmem_executor_storage_helpers", Path(__file__).with_name("analyze_storage_ablation.py")
)
storage = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = storage
_SPEC.loader.exec_module(storage)
AnalysisError = storage.AnalysisError
require, integer, load_arm = storage.require, storage.integer, storage.load_arm
ARMS = ("baseline", "candidate")
SOURCE_FILES = {
    "__init__.py",
    "__main__.py",
    "api.py",
    "cli.py",
    "config.py",
    "environments.py",
    "evaluation.py",
    "memory.py",
    "paper_assets.py",
    "pipeline.py",
    "prompts.py",
    "usage.py",
}
SOURCE_CHANGES = ["cli.py", "config.py", "evaluation.py", "pipeline.py"]
CONFIG_CHANGES = ["executor.model", "experiment.output_dir"]
BRIDGE_VARIANTS = ["legacy_default", "compatible_default", "compatible_explicit_same_judge"]
DEFAULT_BRIDGE = PROJECT / "outputs/executor_gpt61_source_behavior_bridge.json"
DEFAULT_PREFLIGHT = PROJECT / "outputs/executor_gpt61_preflight.json"
EFFICIENCY_FIELDS = (
    "mean_executor_input_tokens_k",
    "mean_executor_output_tokens_k",
    "mean_executor_interaction_turns",
)


def effective_config(config):
    """Resolve an omitted judge while preserving all effective role parameters."""
    resolved = copy.deepcopy(config)
    if resolved.get("judge") is None:
        resolved["judge"] = copy.deepcopy(resolved["executor"])
    require(isinstance(resolved["judge"], dict), "Judge must resolve to a model configuration")
    return resolved


def differences(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        changed = []
        for key in sorted(set(first) | set(second)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                changed.append(field)
            else:
                changed.extend(differences(first[key], second[key], field))
        return changed
    return [prefix] if first != second or type(first) is not type(second) else []


def validate_protocol(baseline, candidate, expected_seeds, expected_workers, expected_batch_size):
    """Isolate effective executor.model; explicitly disclose source/representation changes."""
    require(baseline.root != candidate.root, "Baseline and candidate must be distinct runs")
    for arm in (baseline, candidate):
        config = arm.manifest["config"]
        experiment = config["experiment"]
        require(arm.summary["split"] == "valid_seen", f"{arm.root}: expected valid_seen")
        if len(arm.tasks) == 140:
            storage.validate_complete_seen_manifest(arm.tasks, config["environment"], str(arm.root))
        require(arm.seeds == list(expected_seeds), f"{arm.root}: unexpected order seeds")
        require(experiment.get("store_policy") == "all", f"{arm.root}: full storage required")
        require(experiment.get("warm_start") is None, f"{arm.root}: cold start required")
        require(experiment.get("task_adaptive") is True, f"{arm.root}: task-adaptive required")
        require(storage.prompt_profile(arm) == "paper-v1", f"{arm.root}: paper-v1 is required")
        for field, expected in (
            ("workers", expected_workers),
            ("batch_size", expected_batch_size),
            ("retrieval_k", 3),
            ("max_steps", 30),
            ("history_window", 3),
        ):
            require(experiment.get(field) == expected, f"{arm.root}: expected {field}={expected}")
        resolved = effective_config(config)
        for role in ("executor", "curator", "judge"):
            model = resolved.get(role, {}).get("model")
            require(isinstance(model, str) and bool(model), f"{arm.root}: {role} model missing")
        require(resolved["curator"]["model"] == "gpt-6.1-sol", "Curator must remain gpt-6.1-sol")
        require(resolved["judge"]["model"] == "gpt-5.5", "Effective judge must remain gpt-5.5")
        for field in (
            "source_hashes",
            "packages",
            "bm25",
            "python",
            "platform",
            "prompt_source",
            "fingerprint",
        ):
            require(bool(arm.manifest.get(field)), f"{arm.root}: {field} metadata missing")
        arm.paper_assets = storage.validate_prompt_assets(arm)
    require(
        baseline.manifest["config"]["executor"]["model"] == "gpt-5.5"
        and candidate.manifest["config"]["executor"]["model"] == "gpt-6.1-sol",
        "Executor models must be baseline gpt-5.5 and candidate gpt-6.1-sol",
    )
    require(
        baseline.manifest["config"].get("judge") is None,
        "Historical baseline must use the implicit executor judge",
    )
    require(
        isinstance(candidate.manifest["config"].get("judge"), dict),
        "Candidate must explicitly configure its independent judge",
    )
    resolved = [effective_config(arm.manifest["config"]) for arm in (baseline, candidate)]
    require(
        differences(*resolved) == CONFIG_CHANGES,
        "Effective config mismatch beyond executor.model and experiment.output_dir",
    )
    for arm in (baseline, candidate):
        require(
            set(arm.manifest["source_hashes"]) == SOURCE_FILES,
            "Core source inventory must contain exactly12 modules",
        )
        for digest in arm.manifest["source_hashes"].values():
            require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest),
                "Invalid source digest",
            )
    require(
        differences(baseline.manifest["source_hashes"], candidate.manifest["source_hashes"])
        == SOURCE_CHANGES,
        "Source differences must be exactly the four independent-judge routing modules",
    )
    normalized = []
    for arm, config in zip((baseline, candidate), resolved):
        manifest = copy.deepcopy(arm.manifest)
        manifest.pop("fingerprint")
        manifest.pop("source_hashes")
        manifest["config"] = config
        manifest["config"]["executor"].pop("model")
        manifest["config"]["experiment"].pop("output_dir", None)
        normalized.append(manifest)
    require(
        normalized[0] == normalized[1],
        "Matched manifests/runtime/task mismatch beyond effective executor.model/output_dir and declared source changes",
    )
    for seed in baseline.seeds:
        require(
            baseline.orders[seed] == candidate.orders[seed], f"Task order mismatch: seed {seed}"
        )


def validate_bridge(baseline, candidate, bridge_path):
    """Require the independently replayed old/new recorded behavior compatibility proof."""
    path = Path(bridge_path).expanduser().resolve()
    proof = storage.read_json(path)
    require(isinstance(proof, dict), "Source behavior bridge must be a JSON object")
    episodes = sum(len(rows) for rows in baseline.results.values())
    calls_by_role = {
        "curator": episodes,
        "executor": sum(row["steps"] for rows in baseline.results.values() for row in rows),
        "judge": episodes,
    }
    calls = sum(calls_by_role.values())
    batch_size = baseline.manifest["config"]["experiment"]["batch_size"]
    batches = sum((len(rows) + batch_size - 1) // batch_size for rows in baseline.results.values())
    complete420 = len(baseline.tasks) == 140 and baseline.seeds == [0, 1, 2] and episodes == 420
    expected = {
        "kind": "executor_gpt61_recorded_baseline_source_behavior_bridge",
        "passed": True,
        "errors": [],
        "benchmark": False,
        "replay_kind": "recorded_api_and_environment_fixture",
        "complete_420_baseline_episode_replay": complete420,
        "baseline_episodes_checked": episodes,
        "recorded_calls_checked": calls,
        "baseline_fingerprint": baseline.manifest["fingerprint"],
        "baseline_manifest_sha256": hashlib.sha256(
            (baseline.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "baseline_source_hashes": baseline.manifest["source_hashes"],
        "candidate_source_hashes": candidate.manifest["source_hashes"],
        "allowed_changed_sources": SOURCE_CHANGES,
        "source_difference_paths": SOURCE_CHANGES,
        "prompt_assets": baseline.manifest["prompt_assets"],
        "variants_checked": BRIDGE_VARIANTS,
        "model_api_calls": 0,
        "external_api_calls": 0,
        "native_environment_steps": 0,
    }
    for field, value in expected.items():
        require(
            proof.get(field) == value and type(proof.get(field)) is type(value),
            f"Source behavior bridge differs: {field}",
        )
    require(
        isinstance(proof.get("variants"), dict) and set(proof["variants"]) == set(BRIDGE_VARIANTS),
        "Incomplete source bridge variants",
    )
    for name in BRIDGE_VARIANTS:
        variant = proof["variants"][name]
        require(isinstance(variant, dict), "Invalid source bridge variant")
        for field, value in {
            "episodes_checked": episodes,
            "calls_checked": calls,
            "calls_by_role": calls_by_role,
            "batches_checked": batches,
            "effective_model_by_role": {
                "curator": "gpt-6.1-sol",
                "executor": "gpt-5.5",
                "judge": "gpt-5.5",
            },
            "effective_judge_model": "gpt-5.5",
            "all_messages_actions_storage_usage_batches_match": True,
        }.items():
            require(
                variant.get(field) == value and type(variant.get(field)) is type(value),
                f"Source bridge {name} differs: {field}",
            )
    actual = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted((PROJECT / "src/jitmem").glob("*.py"))
    }
    require(
        actual == candidate.manifest["source_hashes"],
        "Current core differs from bridged candidate source freeze",
    )
    return {**proof, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_preflight(baseline, candidate, preflight_path, bridge):
    path = Path(preflight_path).expanduser().resolve()
    proof = storage.read_json(path)
    require(
        isinstance(proof, dict) and proof.get("passed") is True, "Executor preflight did not pass"
    )
    freeze = proof.get("freeze")
    require(isinstance(freeze, dict), "Missing executor implementation freeze")
    for field, expected in {
        "baseline_manifest_sha256": hashlib.sha256(
            (baseline.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "baseline_fingerprint": baseline.manifest["fingerprint"],
        "config": candidate.manifest["config"],
        "effective_config_difference_paths": CONFIG_CHANGES,
        "source_hashes": candidate.manifest["source_hashes"],
        "baseline_source_hashes": baseline.manifest["source_hashes"],
        "source_difference_paths": SOURCE_CHANGES,
        "bridge_proof_sha256": bridge["sha256"],
        "prompt_assets": candidate.manifest["prompt_assets"],
        "tasks": candidate.manifest["tasks"],
        "packages": candidate.manifest["packages"],
        "python_version": list(sys.version_info[:3]),
        "expected_episodes": sum(len(rows) for rows in candidate.results.values()),
    }.items():
        require(
            freeze.get(field) == expected and type(freeze.get(field)) is type(expected),
            f"Executor preflight freeze differs: {field}",
        )
    for arm in (baseline, candidate):
        content = {
            "config": arm.manifest["config"],
            "tasks": arm.manifest["tasks"],
            "source_hashes": arm.manifest["source_hashes"],
            "packages": arm.manifest["packages"],
            "python_version": freeze["python_version"],
            "warm_start_sha256": None,
            "prompt_assets": arm.manifest["prompt_assets"],
        }
        expected = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        require(
            arm.manifest["fingerprint"] == expected,
            "Manifest fingerprint does not bind its exact effective/source configuration",
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "freeze": freeze,
    }


def seed_comparison(baseline, candidate, seed):
    banks, systems = {}, {}
    for name, arm in zip(ARMS, (baseline, candidate)):
        banks[name], systems[name] = storage.validate_storage(arm, seed, "all")
    require(systems["baseline"] == systems["candidate"], f"Curator system mismatch: seed {seed}")
    first, second = baseline.results[seed], candidate.results[seed]
    counts = dict.fromkeys(("both_success", "candidate_only", "baseline_only", "both_failure"), 0)
    pairs = []
    for index, (left, right) in enumerate(zip(first, second), 1):
        require(
            left["task_id"] == right["task_id"]
            and left["task_description"] == right["task_description"],
            f"Observed task mismatch: seed {seed}, position {index}",
        )
        outcome = (
            "both_success"
            if left["success"] and right["success"]
            else "candidate_only"
            if right["success"]
            else "baseline_only"
            if left["success"]
            else "both_failure"
        )
        counts[outcome] += 1
        pairs.append(
            {
                "task_id": left["task_id"],
                "task_type": left["task_type"],
                "outcome": outcome,
                **{
                    name + "_" + field: row[field]
                    for name, row in zip(ARMS, (left, right))
                    for field in ("success", "stored")
                },
                **{
                    name + "_judge_success": row["judge"]["success"]
                    for name, row in zip(ARMS, (left, right))
                },
                "sources": {
                    name: {
                        "results_jsonl": str(arm.root / f"seed_{seed}/results.jsonl"),
                        "line": index,
                        "episode_json": str(
                            arm.root / f"seed_{seed}/episodes/{index - 1:04d}.json"
                        ),
                    }
                    for name, arm in zip(ARMS, (baseline, candidate))
                },
            }
        )
    metrics = {}
    for name, rows in zip(ARMS, (first, second)):
        metrics[name] = {**storage.arm_metrics(rows), "storage": banks[name]}
        metrics[name]["paper_efficiency"] = storage.paper_efficiency(metrics[name], len(rows))
    by_type = {}
    for task_type in sorted({task["task_type"] for task in baseline.tasks.values()}):
        rates = {
            name: statistics.mean(row["success"] for row in rows if row["task_type"] == task_type)
            for name, rows in zip(ARMS, (first, second))
        }
        by_type[task_type] = {
            "tasks": sum(row["task_type"] == task_type for row in first),
            "baseline_sr": rates["baseline"],
            "candidate_sr": rates["candidate"],
            "delta_pp": 100 * (rates["candidate"] - rates["baseline"]),
        }
    return {
        "seed": seed,
        "tasks": len(first),
        "paired_counts": counts,
        **metrics,
        "delta_pp": 100
        * (metrics["candidate"]["success_rate"] - metrics["baseline"]["success_rate"]),
        "mean_decision_delta": metrics["candidate"]["mean_decisions"]
        - metrics["baseline"]["mean_decisions"],
        "mean_environment_step_delta": metrics["candidate"]["mean_environment_steps"]
        - metrics["baseline"]["mean_environment_steps"],
        "by_task_type": by_type,
        "task_pairs": pairs,
    }


def aggregate_metrics(per_seed):
    result = {
        field: storage.mean_std([run[field] for run in per_seed])
        for field in ("delta_pp", "mean_decision_delta", "mean_environment_step_delta")
    }
    for name in ARMS:
        result[name + "_sr_percent"] = storage.mean_std(
            [100 * run[name]["success_rate"] for run in per_seed]
        )
        for field in ("mean_decisions", "mean_environment_steps"):
            result[name + "_" + field] = storage.mean_std([run[name][field] for run in per_seed])
        for field in EFFICIENCY_FIELDS:
            result[name + "_" + field] = storage.nullable_mean_std(
                [run[name]["paper_efficiency"][field] for run in per_seed]
            )
    result["by_task_type"] = {
        task_type: {
            "tasks_per_seed": per_seed[0]["by_task_type"][task_type]["tasks"],
            **{
                name + "_sr_percent": storage.mean_std(
                    [100 * run["by_task_type"][task_type][name + "_sr"] for run in per_seed]
                )
                for name in ARMS
            },
            "delta_pp": storage.mean_std(
                [run["by_task_type"][task_type]["delta_pp"] for run in per_seed]
            ),
        }
        for task_type in per_seed[0]["by_task_type"]
    }
    return result


def overall_metrics(per_seed):
    result = {}
    for name in ARMS:
        runs = [run[name] for run in per_seed]
        roles = {}
        for role in ("curator", "executor", "judge"):
            pieces = [run["usage_by_role"][role] for run in runs]
            roles[role] = {
                "calls": sum(piece["calls"] for piece in pieces),
                **{
                    field: storage.nullable_total([piece[field] for piece in pieces])
                    for field in ("prompt_tokens", "completion_tokens")
                },
            }
        tokens = {
            field: storage.nullable_total([run[field] for run in runs])
            for field in ("prompt_tokens", "completion_tokens")
        }
        result[name] = {
            "episodes": sum(run["tasks"] for run in runs),
            "calls": sum(role["calls"] for role in roles.values()),
            "all_role_prompt_tokens": tokens["prompt_tokens"],
            "all_role_completion_tokens": tokens["completion_tokens"],
            "all_role_input_output_tokens": storage.nullable_total(list(tokens.values())),
            "usage_by_role": roles,
        }
        result[name]["paper_efficiency"] = storage.paper_efficiency(
            result[name], result[name]["episodes"]
        )
    return result


def build_comparison(
    baseline_dir,
    candidate_dir,
    *,
    expected_tasks=140,
    expected_seeds=(0, 1, 2),
    expected_workers=10,
    expected_batch_size=10,
    bridge_proof=DEFAULT_BRIDGE,
    preflight=DEFAULT_PREFLIGHT,
):
    for label, value in (
        ("expected_tasks", expected_tasks),
        ("expected_workers", expected_workers),
        ("expected_batch_size", expected_batch_size),
    ):
        integer(value, label, 1)
    require(bool(expected_seeds), "Expected order seeds must not be empty")
    for seed in expected_seeds:
        integer(seed, "expected_seed")
    require(
        list(expected_seeds) == sorted(set(expected_seeds)),
        "Expected order seeds must be unique and sorted",
    )
    baseline = load_arm(baseline_dir, "jitmem", expected_tasks)
    candidate = load_arm(candidate_dir, "jitmem", expected_tasks)
    validate_protocol(baseline, candidate, expected_seeds, expected_workers, expected_batch_size)
    bridge = validate_bridge(baseline, candidate, bridge_proof)
    launch = validate_preflight(baseline, candidate, preflight, bridge)
    per_seed = [seed_comparison(baseline, candidate, seed) for seed in baseline.seeds]
    overall = overall_metrics(per_seed)
    config = baseline.manifest["config"]
    complete_seen420 = (
        expected_tasks == 140
        and baseline.seeds == [0, 1, 2]
        and expected_workers == 10
        and expected_batch_size == 10
        and all(overall[name]["episodes"] == 420 for name in ARMS)
    )
    return {
        "kind": "matched_native_alfworld_executor_model_comparison_fixed_judge",
        "benchmark": complete_seen420,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "valid_seen",
        "tasks_per_seed": expected_tasks,
        "seeds": baseline.seeds,
        "total_episodes": sum(overall[name]["episodes"] for name in ARMS),
        "baseline_episodes": overall["baseline"]["episodes"],
        "candidate_episodes": overall["candidate"]["episodes"],
        "new_model_evaluated_episodes": overall["candidate"]["episodes"],
        "executor_models": {
            name: arm.manifest["config"]["executor"]["model"]
            for name, arm in zip(ARMS, (baseline, candidate))
        },
        "judge_model": effective_config(config)["judge"]["model"],
        "curator_model": config["curator"]["model"],
        "variant": "API prompted/untrained full-storage executor comparison under fixed independent judge; source routing changes require an offline recorded-behavior bridge; not author RL-trained/model-equivalent results",
        "prompt_profile": "paper-v1",
        "difference_definition": "candidate minus baseline",
        "protocol": {
            "workers": expected_workers,
            "batch_size": expected_batch_size,
            "process_start_method": baseline.manifest["process_start_method"],
            "matched_effective_config_except": CONFIG_CHANGES,
            "configuration_representation_difference": "baseline implicit judge reuses executor; candidate explicitly configures the same effective judge",
            "store_policy": "all",
            "cold_start": True,
            "labels_visible": True,
            "prompt_profile": "paper-v1",
            "prompt_assets": baseline.manifest["prompt_assets"],
            "primary_efficiency_scope": "executor-only tokens (K) and interaction turns per task; all evaluated tasks",
            "matched_config": config,
            "baseline_effective_config": effective_config(baseline.manifest["config"]),
            "candidate_effective_config": effective_config(candidate.manifest["config"]),
            "source_difference_paths": SOURCE_CHANGES,
            "same_core_sources": False,
            "source_behavior_bridge_sha256": bridge["sha256"],
            "preflight_sha256": launch["sha256"],
        },
        "source_behavior_bridge": bridge,
        "preflight": launch,
        "sources": {
            name: {
                "root": str(arm.root),
                "fingerprint": arm.manifest["fingerprint"],
                "files": [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in arm.evidence_files
                ],
            }
            for name, arm in zip(ARMS, (baseline, candidate))
        },
        "per_seed": per_seed,
        "overall": overall,
        "across_seed": aggregate_metrics(per_seed),
        "statistical_scope": {
            "std_definition": "sample standard deviation across seeds, ddof=1",
            "unique_task_count": expected_tasks,
            "repeated_episode_pairs": overall["candidate"]["episodes"],
            "pooled_significance_test": None,
            "note": "Repeated tasks and memory-coupled streams are not independent observations; results are descriptive.",
        },
        "usage_scope": "Committed episode API usage only; baseline usage is historical, candidate usage is new; transport retries and uncommitted billing are not inferred.",
        "analysis_model_api_calls": 0,
        "analysis_environment_steps": 0,
    }


def render_markdown(report):
    across = report["across_seed"]
    delta = storage.show_mean_std(across["delta_pp"])
    status = "完整 seen140×3 评测" if report["benchmark"] else "非 benchmark 的小规模验证"
    lines = [
        "# ALFWorld Executor 模型对照（paper-v1，full storage，固定judge）",
        "",
        f"{status}。Candidate 减 baseline 的 native SR 差为 **{delta} 个百分点**。仅报告描述性结果，不预设更换模型带来提升。",
        "",
        f"Baseline executor `{report['executor_models']['baseline']}`，candidate executor `{report['executor_models']['candidate']}`；"
        f"curator固定 `{report['curator_model']}`，judge固定 `{report['judge_model']}`。两组均为附judge标签的全量存储、task-adaptive、每轮冷启动；"
        "有效配置仅executor.model和输出目录不同，原文资产、runtime、数据、任务与顺序匹配。",
        "",
        "核心源码并非完全相同：独立judge路由只改变cli.py、config.py、evaluation.py、pipeline.py，旧/新12份指纹分别保留。"
        "离线兼容性证明以记录的API返回和环境反馈回放历史记录，对旧默认、新默认、新显式同模型judge三种配置核对messages/actions/storage/usage/batches；这不是新增模型评测，也不是原生环境重放。",
        "",
        f"复用 baseline 的{report['baseline_episodes']}条历史 episode，新 candidate 评测{report['candidate_episodes']}条；"
        f"比较共{report['total_episodes']}条记录，不代表本次新增{report['total_episodes']}条模型评测。分析脚本无模型调用或环境重放。",
        "",
        "这是模型API、未训练curator的executor对照，judge固定的独立路由是为控制本次变量而实施的扩展；原论文judge复用executor，因此本实验不代表原论文原封不动的变体或RL-trained结果。服务端权重未独立验证。",
        "",
        "## Native SR 与 Table4 executor-only 效率",
        "",
        "| Arm | SR mean ± sample std | 成功 / episodes | Executor input K / task | Executor output K / task | Executor turns / task |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ARMS:
        successes = sum(run[name]["successes"] for run in report["per_seed"])
        values = [
            storage.show_mean_std(across[name + "_" + field], 3 if index < 2 else 2)
            for index, field in enumerate(EFFICIENCY_FIELDS)
        ]
        lines.append(
            f"| {name} | {storage.show_mean_std(across[name + '_sr_percent'])}% | {successes} / {report['overall'][name]['episodes']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "K=1000，统计所有任务的executor全部调用，curator/judge不混入效率列。缺失usage保留null并显示未知，不补0；API token口径取决于提供方。",
        "",
        "| Seed | Baseline 成功 | Candidate 成功 | Δ pp | 两者成功 | 仅candidate | 仅baseline | 两者失败 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["per_seed"]:
        pair = run["paired_counts"]
        lines.append(
            f"| {run['seed']} | {run['baseline']['successes']} / {run['tasks']} | {run['candidate']['successes']} / {run['tasks']} | {run['delta_pp']:+.2f} | "
            + " | ".join(
                str(pair[key])
                for key in ("both_success", "candidate_only", "baseline_only", "both_failure")
            )
            + " |"
        )
    lines += [
        "",
        f"同一任务重复{len(report['seeds'])}轮且streaming记忆使轮内结果相关；order seed不固定服务端生成采样。std是种子间样本标准差，不是置信区间，不做pooled显著性检验。",
        "",
        "## Gate、存储及检索诊断",
        "",
        "| Seed / arm | Native vs judge TP / FP / FN / TN | Judge解析 / 生成异常 | 非法 / 预算耗尽 | 最终bank | Native失败存入 | 检索引用 / native失败引用 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for run in report["per_seed"]:
        for name in ARMS:
            arm, bank = run[name], run[name]["storage"]
            judge = arm["native_vs_judge"]
            matrix = " / ".join(
                str(judge[key])
                for key in ("true_positive", "false_positive", "false_negative", "true_negative")
            )
            lines.append(
                f"| {run['seed']} / {name} | {matrix} | {judge['parse_errors']} / {judge['generation_errors']} | {arm['invalid_actions']} / {arm['truncated_episodes']} | "
                f"{bank['final_bank_size']} | {bank['native_failed_stored_entries']} | {bank['retrieval']['references']} / {bank['retrieval']['native_failed_references']} |"
            )
    lines += [
        "",
        "Native verifier决定SR；固定gpt-5.5 judge自评每条轨迹，两组均保存全部轨迹并向curator展示judge标签，不用native标签修正judge。更换executor会改变轨迹、标签和后续检索/payload；这些反馈不能单独归因为某一个因素。judge输出不决定本次全量组是否入库。",
        "",
        "## 按角色API返回用量",
        "",
        "| Arm / role | Calls | Input tokens | Output tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ARMS:
        for role, measured in report["overall"][name]["usage_by_role"].items():
            lines.append(
                f"| {name} / {role} | {measured['calls']} | {storage.shown(measured['prompt_tokens'])} | {storage.shown(measured['completion_tokens'])} |"
            )
    lines += [
        "",
        "用量只统计已提交episode返回的usage：baseline为历史用量，candidate为本次新用量；不推算账单、重试或未提交请求的额外消耗。",
        "",
        "## 本机证据",
        "",
        "executor_comparison.json包含逐任务配对、分任务类指标、批次bank增长、原始文件哈希、resolved config、旧/新源码与兼容证明，仅保存于被忽略的outputs，不适合直接公开。公开结果须另行白名单导出。",
    ]
    for name, source in report["sources"].items():
        lines.append(
            f"{name}：[manifest](<{Path(source['root']) / 'manifest.json'}>)，fingerprint=`{source['fingerprint']}`。"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/executor_gpt61_comparison")
    )
    parser.add_argument("--bridge-proof", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--expected-tasks", type=int, default=140)
    parser.add_argument("--expected-workers", type=int, default=10)
    parser.add_argument("--expected-batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        require(
            output.is_relative_to(PROJECT / "outputs"),
            "Raw local analysis must remain under ignored project outputs",
        )
        require(
            not any(
                output.is_relative_to(root.expanduser().resolve())
                for root in (args.baseline, args.candidate)
            ),
            "Analysis output must not change either evaluation directory",
        )
        report = build_comparison(
            args.baseline,
            args.candidate,
            expected_tasks=args.expected_tasks,
            expected_workers=args.expected_workers,
            expected_batch_size=args.expected_batch_size,
            bridge_proof=args.bridge_proof,
            preflight=args.preflight,
        )
        output.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = (
            output / "executor_comparison.json",
            output / "executor_comparison.md",
        )
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved {json_path}\nSaved {markdown_path}")
    except (AnalysisError, OSError) as error:
        parser.exit(1, f"Executor comparison analysis rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
