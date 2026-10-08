"""Synthetic native-shaped artifacts verify analysis, never model capability."""

import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_results.py"
SPEC = importlib.util.spec_from_file_location("jitmem_result_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def worker_metadata(manifest, workers):
    manifest["config"]["experiment"]["workers"] = workers
    manifest["workers"] = workers
    manifest["process_start_method"] = "spawn" if workers > 1 else None
    manifest["batch_execution"] = (
        "spawn process parallel execution with a shared frozen bank per batch"
        if workers > 1
        else "sequential execution with a shared frozen bank per batch"
    )


def create_arm(root, method, outcomes, *, unknown_usage=False, workers=1, batch_size=10):
    tasks = [
        {
            "task_id": f"task-{i}",
            "task_type": "pick" if i < 2 else "clean",
            "split": "valid_seen",
            "description": f"objective {i}",
            "game_sha256": f"game-{i}",
            "game_file": f"/dataset/{i}.json",
        }
        for i in range(4)
    ]
    seeds = sorted(outcomes)
    executor = {
        "model": "gpt-5.5",
        "base_url": "https://gateway.example/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 1.0,
        "max_tokens": 4096,
    }
    manifest = {
        "benchmark": True,
        "fingerprint": f"{method}-fingerprint",
        "config": {
            "environment": {
                "backend": "alfworld",
                "split": "valid_seen",
                "limit": None,
                "task_types": [1, 2, 3, 4, 5, 6],
                "annotation_index": 0,
            },
            "experiment": {
                "method": method,
                "seeds": seeds,
                "batch_size": batch_size,
                "max_steps": 30,
                "history_window": 3,
            },
            "executor": executor,
            "curator": {**executor, "max_tokens": 8192},
        },
        "tasks": tasks,
        "source_hashes": {"pipeline.py": "same-source"},
        "packages": {"alfworld": "0.4.2"},
        "prompt_source": "paraphrased",
        "memory_commit_order": "predetermined task order after the complete batch succeeds",
    }
    worker_metadata(manifest, workers)
    runs = []
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        rows = []
        for index, success in enumerate(outcomes[seed]):
            judge = {"success": success, "rationale": "synthetic", "evidence_step": 1}
            if method == "no-memory":
                judge = None
            by_role = {"executor": {"calls": 3, "prompt_tokens": 30, "completion_tokens": 6}}
            if method == "jitmem":
                by_role.update(
                    {
                        "curator": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 2},
                        "judge": {"calls": 1, "prompt_tokens": 20, "completion_tokens": 3},
                    }
                )
            if unknown_usage and index == 0:
                by_role["executor"]["prompt_tokens"] = None
            rows.append(
                {
                    "task_id": f"task-{index}",
                    "task_type": tasks[index]["task_type"],
                    "split": "valid_seen",
                    "task_description": f"objective {index}",
                    "success": success,
                    "steps": 3,
                    "environment_steps": 3,
                    "invalid_actions": 0,
                    "truncated": False,
                    "judge": judge,
                    "worker_pid": 1000 + index,
                    "stored": method == "jitmem" and success,
                    "generation_failures": [],
                    "usage": {
                        "by_role": by_role,
                        "prompt_tokens": None if unknown_usage and index == 0 else 60,
                        "completion_tokens": 11,
                    },
                }
            )
        random.Random(seed).shuffle(rows)
        bank_size = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            for row in batch:
                row["batch_index"] = offset // batch_size
                row["memory_size_before"] = bank_size
            bank_size += sum(row["stored"] for row in batch)
        save(run_dir / "task_order.json", [row["task_id"] for row in rows])
        (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        run_summary = {
            "seed": seed,
            "benchmark": True,
            "tasks": 4,
            "successes": sum(outcomes[seed]),
            "success_rate": statistics.mean(outcomes[seed]),
        }
        save(run_dir / "summary.json", run_summary)
        runs.append(run_summary)
    rates = [run["success_rate"] for run in runs]
    save(root / "manifest.json", manifest)
    save(
        root / "summary.json",
        {
            "method": method,
            "split": "valid_seen",
            "benchmark": True,
            "runs": runs,
            "success_rate_mean": statistics.mean(rates),
            "success_rate_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        },
    )
    return root


@pytest.fixture
def arms(tmp_path):
    first = create_arm(
        tmp_path / "baseline",
        "no-memory",
        {0: [True, True, False, False], 1: [True, False, False, False]},
    )
    second = create_arm(
        tmp_path / "jitmem",
        "jitmem",
        {0: [True, False, True, False], 1: [True, True, True, False]},
        unknown_usage=True,
    )
    return first, second


def test_matched_comparison_preserves_pairs_and_unknown_usage(arms, tmp_path):
    report = analysis.build_comparison(*arms, expected_tasks=4)
    assert report["per_seed"][0]["paired_counts"] == {
        "both_success": 1,
        "baseline_only": 1,
        "jitmem_only": 1,
        "both_failure": 1,
    }
    assert report["per_seed"][1]["delta_pp"] == 50
    assert report["across_seed"]["delta_pp"]["mean"] == 25
    assert report["across_seed"]["delta_pp"]["std"] == pytest.approx(35.3553390593)
    assert report["statistical_scope"]["unique_task_count"] == 4
    assert report["statistical_scope"]["repeated_episode_pairs"] == 8
    assert report["statistical_scope"]["pooled_significance_test"] is None
    assert report["per_seed"][0]["jitmem"]["usage_by_role"]["executor"]["prompt_tokens"] is None
    assert report["per_seed"][0]["jitmem"]["native_vs_judge"]["true_positive"] == 2
    source = report["per_seed"][0]["task_pairs"][2]["sources"]["baseline"]
    assert Path(source["results_jsonl"]).is_absolute() and source["line"] == 3
    output = tmp_path / "comparison"
    assert (
        analysis.main(
            [str(arms[0]), str(arms[1]), "--expected-tasks", "4", "--output-dir", str(output)]
        )
        == 0
    )
    markdown = (output / "comparison.md").read_text()
    assert "gpt-5.5" in markdown and "RL-trained" in markdown and "未知" in markdown
    assert "独立任务样本" in markdown
    assert json.loads((output / "comparison.json").read_text())["benchmark"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("executor", "Executor config mismatch"),
        ("seed", "Seed mismatch"),
        ("source", "Implementation protocol mismatch"),
        ("batch", "Experiment protocol mismatch"),
        ("workers", "Experiment protocol mismatch: workers"),
        ("split", "Split mismatch"),
        ("task", "Task set mismatch"),
        ("mock", "not a native benchmark"),
        ("limited", "limited pilot"),
    ],
)
def test_rejects_unmatched_or_nonbenchmark_protocols(arms, mutation, match):
    root = arms[1]
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if mutation == "executor":
        manifest["config"]["executor"]["temperature"] = 0
    elif mutation == "source":
        manifest["source_hashes"]["pipeline.py"] = "changed"
    elif mutation == "batch":
        manifest["config"]["experiment"]["batch_size"] = 20
    elif mutation == "workers":
        worker_metadata(manifest, 10)
    elif mutation == "mock":
        manifest["benchmark"] = False
    elif mutation == "limited":
        manifest["config"]["environment"]["limit"] = 4
    elif mutation == "seed":
        manifest["config"]["experiment"]["seeds"] = [0]
        summary["runs"] = summary["runs"][:1]
        summary["success_rate_mean"] = summary["runs"][0]["success_rate"]
        summary["success_rate_std"] = 0
    elif mutation in {"split", "task"}:
        if mutation == "split":
            manifest["config"]["environment"]["split"] = summary["split"] = "valid_unseen"
        else:
            manifest["tasks"][0]["task_id"] = "changed-task"
        for task in manifest["tasks"]:
            if mutation == "split":
                task["split"] = "valid_unseen"
        for seed in manifest["config"]["experiment"]["seeds"]:
            result_path = root / f"seed_{seed}" / "results.jsonl"
            rows = [json.loads(line) for line in result_path.read_text().splitlines()]
            if mutation == "split":
                for row in rows:
                    row["split"] = "valid_unseen"
            else:
                next(row for row in rows if row["task_id"] == "task-0")["task_id"] = "changed-task"
            result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            save(root / f"seed_{seed}" / "task_order.json", [row["task_id"] for row in rows])
    save(root / "manifest.json", manifest)
    save(root / "summary.json", summary)
    with pytest.raises(analysis.AnalysisError, match=match):
        analysis.build_comparison(*arms, expected_tasks=4)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_file", "Cannot read complete results"),
        ("partial", "full manifest"),
        ("declared_count", "declared count is partial"),
        ("wrong_order", "task order disagrees with seed shuffle"),
        ("async_order", "result order disagrees"),
        ("stale_sr", "declared SR"),
        ("unresolved_error", "unresolved run error"),
    ],
)
def test_rejects_partial_or_inconsistent_results(arms, mutation, match):
    run_dir = arms[1] / "seed_0"
    result_path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if mutation == "missing_file":
        result_path.unlink()
    elif mutation == "partial":
        result_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    elif mutation in {"declared_count", "stale_sr"}:
        summary = json.loads((run_dir / "summary.json").read_text())
        summary["tasks" if mutation == "declared_count" else "success_rate"] = (
            3 if mutation == "declared_count" else 0.75
        )
        save(run_dir / "summary.json", summary)
    elif mutation in {"wrong_order", "async_order"}:
        rows.reverse()
        result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if mutation == "wrong_order":
            save(run_dir / "task_order.json", [row["task_id"] for row in rows])
    else:
        save(run_dir / "error.json", {"error": "unfinished"})
    with pytest.raises(analysis.AnalysisError, match=match):
        analysis.build_comparison(*arms, expected_tasks=4)


def test_default_full_seen_count_rejects_small_synthetic_arm(arms):
    with pytest.raises(analysis.AnalysisError, match="expected 140 manifest tasks"):
        analysis.build_comparison(*arms)


def test_reports_judge_disagreement_and_generation_failures(arms):
    result_path = arms[1] / "seed_0" / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0]["judge"]["success"] = False
    rows[0]["judge"]["generation_error"] = True
    rows[0]["stored"] = False
    rows[0]["generation_failures"] = [{"role": "judge", "finish_reason": "length"}]
    rows[3]["judge"]["success"] = True
    rows[3]["stored"] = True
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analysis.build_comparison(*arms, expected_tasks=4)
    metrics = report["per_seed"][0]["jitmem"]
    assert metrics["native_vs_judge"]["false_negative"] == 1
    assert metrics["native_vs_judge"]["false_positive"] == 1
    assert metrics["native_vs_judge"]["generation_errors"] == 1
    assert metrics["generation_failures"]["by_role"] == {"judge": 1}
    assert metrics["generation_failures"]["by_finish_reason"] == {"length": 1}
    assert metrics["native_failed_stored_episodes"] == 1
    assert metrics["stored_episodes"] == 2
    assert metrics["empty_bank_episodes"] == 4
    # The FP is a both-failure pair and must still receive a judge evidence link.
    pair = report["per_seed"][0]["task_pairs"][3]
    assert pair["outcome"] == "both_failure"
    assert pair["jitmem_judge_success"] is True and pair["jitmem_stored"] is True
    markdown = analysis.render_markdown(report)
    assert "judge FP=1、FN=1" in markdown
    assert "1 条原生失败轨迹" in markdown
    assert "| 0 | task-3 | FP | False | True | True |" in markdown
    assert f"{result_path}:4" in markdown


def test_missing_usage_is_unknown_for_all_called_roles(arms):
    result_path = arms[1] / "seed_0" / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[0].pop("usage")
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = analysis.build_comparison(*arms, expected_tasks=4)["per_seed"][0]["jitmem"]
    assert metrics["prompt_tokens"] is None and metrics["completion_tokens"] is None
    for role in ("executor", "curator", "judge"):
        assert metrics["usage_by_role"][role]["prompt_tokens"] is None
        assert metrics["usage_by_role"][role]["completion_tokens"] is None


def test_rejects_blank_lines_that_would_break_evidence_line_numbers(arms):
    path = arms[1] / "seed_0" / "results.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], "", *lines[1:]]) + "\n")
    with pytest.raises(analysis.AnalysisError, match="blank result lines"):
        analysis.build_comparison(*arms, expected_tasks=4)


def test_matches_ten_spawn_workers_and_reports_execution_protocol(arms):
    for root in arms:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        worker_metadata(manifest, 10)
        save(path, manifest)
    report = analysis.build_comparison(*arms, expected_tasks=4)
    assert report["protocol"]["workers"] == 10
    assert report["protocol"]["process_start_method"] == "spawn"
    assert "shared frozen bank" in report["protocol"]["batch_execution"]
    assert "complete batch" in report["protocol"]["memory_commit_order"]
    assert "10 个 worker" in analysis.render_markdown(report)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("workers", 10, "worker metadata disagrees"),
        ("process_start_method", "fork", "process start method disagrees"),
        ("batch_execution", "as completed", "frozen batch execution metadata"),
        ("memory_commit_order", "as completed", "memory commit order"),
    ],
)
def test_rejects_inconsistent_worker_metadata(arms, field, value, match):
    path = arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match=match):
        analysis.build_comparison(*arms, expected_tasks=4)


def test_rejects_matching_scrambled_task_orders_in_both_arms(arms):
    for root in arms:
        path = root / "seed_0" / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()][::-1]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        save(root / "seed_0" / "task_order.json", [row["task_id"] for row in rows])
    with pytest.raises(analysis.AnalysisError, match="task order disagrees with seed shuffle"):
        analysis.build_comparison(*arms, expected_tasks=4)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("within_batch", "memory changed within a frozen batch"),
        ("commit_size", "memory commit size disagrees"),
        ("batch_index", "batch index disagrees"),
        ("cold_start", "cold start bank is not empty"),
        ("over_cap", "decisions exceed declared cap"),
    ],
)
def test_rejects_invalid_batch_commits_or_decision_caps(tmp_path, mutation, match):
    first = create_arm(
        tmp_path / "baseline", "no-memory", {0: [True, True, False, False]}, batch_size=2
    )
    second = create_arm(
        tmp_path / "jitmem", "jitmem", {0: [True, False, True, False]}, batch_size=2
    )
    result_path = second / "seed_0" / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if mutation == "within_batch":
        rows[1]["memory_size_before"] += 1
    elif mutation == "commit_size":
        for row in rows[2:]:
            row["memory_size_before"] += 1
    elif mutation == "batch_index":
        rows[1]["batch_index"] = 1
    elif mutation == "cold_start":
        rows[0]["memory_size_before"] = 1
    else:
        rows[0]["steps"] = 31
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(analysis.AnalysisError, match=match):
        analysis.build_comparison(first, second, expected_tasks=4)


def test_full_multiple_batches_pass_commit_checks(tmp_path):
    arms = [
        create_arm(
            tmp_path / method,
            method,
            {0: [True, True, False, False], 1: [False, True, False, True]},
            workers=10,
            batch_size=2,
        )
        for method in ("no-memory", "jitmem")
    ]
    report = analysis.build_comparison(*arms, expected_tasks=4)
    assert report["protocol"]["batch_size"] == 2
    assert report["across_seed"]["delta_pp"]["mean"] == 0


def test_report_preserves_all_three_paired_seeds_and_unknown_provider_usage(tmp_path):
    first = create_arm(
        tmp_path / "baseline",
        "no-memory",
        {
            0: [True, True, False, False],
            1: [True, False, False, False],
            2: [True, True, True, False],
        },
        workers=10,
    )
    second = create_arm(
        tmp_path / "jitmem",
        "jitmem",
        {
            0: [True, False, True, False],
            1: [True, True, True, False],
            2: [True, False, False, False],
        },
        workers=10,
    )
    for root in (first, second):
        for seed in range(3):
            path = root / f"seed_{seed}" / "results.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["usage"]["prompt_tokens"] = row["usage"]["completion_tokens"] = None
                for role in row["usage"]["by_role"].values():
                    role["prompt_tokens"] = role["completion_tokens"] = None
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analysis.build_comparison(first, second, expected_tasks=4)
    assert report["seeds"] == [0, 1, 2]
    assert [run["delta_pp"] for run in report["per_seed"]] == [0, 50, -50]
    assert report["across_seed"]["delta_pp"] == {"mean": 0, "std": 50}
    assert report["statistical_scope"]["unique_task_count"] == 4
    assert report["statistical_scope"]["repeated_episode_pairs"] == 12
    for run in report["per_seed"]:
        for arm in ("baseline", "jitmem"):
            assert run[arm]["prompt_tokens"] is None
            assert run[arm]["completion_tokens"] is None
            for role in run[arm]["usage_by_role"].values():
                assert role["prompt_tokens"] is None and role["completion_tokens"] is None
    markdown = analysis.render_markdown(report)
    assert "| 2 | 75.00% | 25.00% | -50.00 |" in markdown
    assert "每组共 12 个 episode，覆盖 4 个唯一任务" in markdown
    assert "不补成 0" in markdown and "推算费用" in markdown
    assert "没有本地训练" in markdown and "不能视为作者训练后数字的等价复现" in markdown
