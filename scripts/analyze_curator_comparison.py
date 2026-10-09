#!/usr/bin/env python3
"""Compare completed paper-v1 quality-filtered runs differing only in curator model.

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
    "_jitmem_curator_storage_helpers", Path(__file__).with_name("analyze_storage_ablation.py")
)
storage = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = storage
_SPEC.loader.exec_module(storage)
AnalysisError = storage.AnalysisError
require, integer, load_arm = storage.require, storage.integer, storage.load_arm
ARMS = ("baseline", "candidate")
EFFICIENCY_FIELDS = (
    "mean_executor_input_tokens_k",
    "mean_executor_output_tokens_k",
    "mean_executor_interaction_turns",
)


def validate_protocol(baseline, candidate, expected_seeds, expected_workers, expected_batch_size):
    """Require complete, matched filtered streams with exactly one model change."""
    require(baseline.root != candidate.root, "Baseline and candidate must be distinct runs")
    for arm in (baseline, candidate):
        config = arm.manifest["config"]
        experiment = config["experiment"]
        require(arm.summary["split"] == "valid_seen", f"{arm.root}: expected valid_seen")
        if len(arm.tasks) == 140:
            storage.validate_complete_seen_manifest(arm.tasks, config["environment"], str(arm.root))
        require(arm.seeds == list(expected_seeds), f"{arm.root}: unexpected order seeds")
        require(experiment.get("store_policy") == "judge", f"{arm.root}: judge filtering required")
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
        for role in ("executor", "curator"):
            model = config.get(role, {}).get("model")
            require(isinstance(model, str) and bool(model), f"{arm.root}: {role} model missing")
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
        baseline.manifest["config"]["curator"]["model"]
        != candidate.manifest["config"]["curator"]["model"],
        "Curator models must differ",
    )
    normalized = []
    for arm in (baseline, candidate):
        manifest = copy.deepcopy(arm.manifest)
        manifest.pop("fingerprint")
        manifest["config"]["curator"].pop("model")
        manifest["config"]["experiment"].pop("output_dir", None)
        normalized.append(manifest)
    require(
        normalized[0] == normalized[1],
        "Matched manifests/config mismatch beyond curator.model and experiment.output_dir",
    )
    for seed in baseline.seeds:
        require(
            baseline.orders[seed] == candidate.orders[seed], f"Task order mismatch: seed {seed}"
        )


def seed_comparison(baseline, candidate, seed):
    banks, systems = {}, {}
    for name, arm in zip(ARMS, (baseline, candidate)):
        banks[name], systems[name] = storage.validate_storage(arm, seed, "judge")
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
        "kind": "matched_native_alfworld_curator_model_comparison",
        "benchmark": complete_seen420,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "valid_seen",
        "tasks_per_seed": expected_tasks,
        "seeds": baseline.seeds,
        "total_episodes": sum(overall[name]["episodes"] for name in ARMS),
        "baseline_episodes": overall["baseline"]["episodes"],
        "candidate_episodes": overall["candidate"]["episodes"],
        "new_model_evaluated_episodes": overall["candidate"]["episodes"],
        "executor_model": config["executor"]["model"],
        "judge_model": config["executor"]["model"],
        "curator_models": {
            name: arm.manifest["config"]["curator"]["model"]
            for name, arm in zip(ARMS, (baseline, candidate))
        },
        "variant": "API prompted/untrained curator comparison with original templates; not author RL-trained/model-equivalent results",
        "prompt_profile": "paper-v1",
        "difference_definition": "candidate minus baseline",
        "protocol": {
            "workers": expected_workers,
            "batch_size": expected_batch_size,
            "process_start_method": baseline.manifest["process_start_method"],
            "matched_config_except": ["curator.model", "experiment.output_dir"],
            "store_policy": "judge",
            "cold_start": True,
            "labels_visible": False,
            "prompt_profile": "paper-v1",
            "prompt_assets": baseline.manifest["prompt_assets"],
            "primary_efficiency_scope": "executor-only tokens (K) and interaction turns per task; all evaluated tasks",
            "matched_config": config,
        },
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
        "# ALFWorld Curator 模型对照（paper-v1，quality-filtered）",
        "",
        f"{status}。Candidate 减 baseline 的 native SR 差为 **{delta} 个百分点**。仅报告描述性结果，不预设更换模型带来提升。",
        "",
        f"Baseline curator `{report['curator_models']['baseline']}`，candidate curator `{report['curator_models']['candidate']}`；"
        f"executor 与 judge 均固定 `{report['executor_model']}`。两组均为 judge-filtered、task-adaptive、每轮冷启动；"
        "仅 curator.model 和输出目录不同，配置、源码、原文资产、runtime、数据、任务与顺序经过匹配验证。",
        "",
        f"复用 baseline 的{report['baseline_episodes']}条历史 episode，新 candidate 评测{report['candidate_episodes']}条；"
        f"比较共{report['total_episodes']}条记录，不代表本次新增{report['total_episodes']}条模型评测。分析脚本无模型调用或环境重放。",
        "",
        "这是模型API、未训练curator的对照，不代表原论文模型或RL-trained结果；继承runtime及作者未披露细节仍见原文对齐说明。",
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
        "Native verifier决定SR，固定executor模型judge决定入库，不用native标签修正gate；两组均不展示judge结果标签。更换curator会改变后续轨迹、gate、bank及检索，不能将这些中介作用单独归因为某一个因素。",
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
        "curator_comparison.json包含逐任务配对、分任务类指标、批次bank增长、原始文件哈希和resolved config，仅保存于被忽略的outputs，不适合直接公开。公开结果须另行白名单导出。",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/curator_gpt61_comparison"))
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
        )
        output.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = (
            output / "curator_comparison.json",
            output / "curator_comparison.md",
        )
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved {json_path}\nSaved {markdown_path}")
    except (AnalysisError, OSError) as error:
        parser.exit(1, f"Curator comparison analysis rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
