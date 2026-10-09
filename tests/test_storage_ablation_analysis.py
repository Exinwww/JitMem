"""Local synthetic artifact checks; these fixtures do not evaluate model capability."""

import copy
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_storage_ablation.py"
SPEC = importlib.util.spec_from_file_location("storage_ablation_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def save_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def entry_from_episode(episode):
    return {
        **{key: episode[key] for key in ("task_id", "task_description", "task_type", "split")},
        **episode["trajectory"],
        "judge_success": episode["judge"]["success"],
        "summary": None,
    }


def fixture_context(entries, policy):
    parts = []
    for index, entry in enumerate(entries, 1):
        label = ""
        if policy == "all":
            label = "\nExecutor judge label: " + (
                "success" if entry["judge_success"] else "failure"
            )
        trace = f"OBSERVATION 0: {entry['initial_observation']}"
        for number, turn in enumerate(entry["turns"], 1):
            trace += f"\nACTION {number}: {turn['action']}\nOBSERVATION {number}: {turn['next_observation']}"
        parts.append(
            f"Memory {index}:\nQuestion: {entry['task_description']}{label}\nTrajectory:\n{trace}"
        )
    return "\n\n".join(parts) or "No past episodes are available."


def create_arm(root, policy, outcomes, *, unknown_usage=False, all_rejected=False):
    tasks = [
        {
            "task_id": f"task-{i}",
            "task_type": "pick" if i < 3 else "clean",
            "split": "valid_seen",
            "description": f"objective {i}",
            "game_sha256": f"game-{i}",
            "game_file": f"/dataset/{i}.json",
        }
        for i in range(6)
    ]
    model = {
        "model": "gpt-5.5",
        "base_url": "https://gateway.example/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 1.0,
        "max_tokens": 4096,
        "extra_body": {},
    }
    manifest = {
        "benchmark": True,
        "fingerprint": f"{policy}-fingerprint",
        "tasks": tasks,
        "config": {
            "environment": {
                "backend": "alfworld",
                "data_root": "/dataset",
                "split": "valid_seen",
                "limit": None,
                "task_types": [1, 2, 3, 4, 5, 6],
                "annotation_index": 0,
            },
            "experiment": {
                "method": "jitmem",
                "seeds": [0, 1, 2],
                "workers": 10,
                "batch_size": 2,
                "max_steps": 30,
                "history_window": 3,
                "retrieval_k": 3,
                "task_adaptive": True,
                "store_policy": policy,
                "output_dir": str(root),
                "warm_start": None,
            },
            "executor": model,
            "curator": {**model, "max_tokens": 8192},
        },
        "source_hashes": {"pipeline.py": "same-source"},
        "packages": {"alfworld": "0.4.2"},
        "bm25": {"k1": 1.5, "b": 0.75},
        "python": "synthetic-runtime",
        "platform": "synthetic-platform",
        "prompt_source": "semantic paraphrases",
        "variant": "prompted curator; no local training",
        "warm_start_sha256": None,
        "workers": 10,
        "process_start_method": "spawn",
        "batch_execution": "spawn process parallel execution with a shared frozen bank per batch",
        "memory_commit_order": "predetermined task order after the complete batch succeeds",
    }
    runs = []
    for seed in (0, 1, 2):
        run_dir = root / f"seed_{seed}"
        order = list(range(6))
        random.Random(seed).shuffle(order)
        episodes, bank = [], []
        for offset in range(0, 6, 2):
            pending = []
            for task_index in order[offset : offset + 2]:
                task = tasks[task_index]
                success = outcomes[seed][task_index]
                judge = False if all_rejected else task_index % 2 == 0
                stored = policy == "all" or judge
                initial, feedback = f"room {task_index}", f"observation {task_index}"
                references = [{"task_id": entry["task_id"], "score": 1.0} for entry in bank[:3]]
                curator_user = (
                    f"Question: {task['description']}\n\nRetrieved memories:\n"
                    + fixture_context(bank[:3], policy)
                )
                calls = [
                    {
                        "role": "curator",
                        "messages": [
                            {"role": "system", "content": "Synthetic curator."},
                            {"role": "user", "content": curator_user},
                        ],
                    },
                    {
                        "role": "executor",
                        "messages": [{"role": "user", "content": "Synthetic executor fixture."}],
                    },
                    {
                        "role": "judge",
                        "messages": [{"role": "user", "content": "Synthetic judge fixture."}],
                    },
                ]
                for call in calls:
                    call.update(
                        {
                            "response": "synthetic",
                            "prompt_tokens": None if unknown_usage else 3,
                            "completion_tokens": None if unknown_usage else 2,
                            "incomplete": False,
                            "finish_reason": "stop",
                            "latency_seconds": 0.1,
                        }
                    )
                episode = {
                    **{key: task[key] for key in ("task_id", "task_type", "split")},
                    "task_description": task["description"],
                    "success": success,
                    "steps": 1,
                    "environment_steps": 1,
                    "invalid_actions": 0,
                    "truncated": False,
                    "worker_pid": 1000 + task_index,
                    "batch_index": offset // 2,
                    "memory_size_before": len(bank),
                    "retrieved": references,
                    "payload": "synthetic briefing",
                    "judge": {"success": judge, "rationale": "synthetic", "evidence_step": 1},
                    "stored": stored,
                    "generation_failures": [],
                    "calls": calls,
                    "usage": {
                        "prompt_tokens": None if unknown_usage else 9,
                        "completion_tokens": None if unknown_usage else 6,
                        "by_role": {
                            role: {
                                "calls": 1,
                                "prompt_tokens": None if unknown_usage else 3,
                                "completion_tokens": None if unknown_usage else 2,
                            }
                            for role in ("curator", "executor", "judge")
                        },
                    },
                    "trajectory": {
                        "initial_observation": initial,
                        "turns": [
                            {
                                "observation": initial,
                                "action": "look",
                                "response": "<action>look</action>",
                                "next_observation": feedback,
                                "admissible_actions": ["look"],
                                "executed": True,
                            }
                        ],
                    },
                }
                episodes.append(episode)
                if stored:
                    pending.append(entry_from_episode(episode))
            bank.extend(pending)
        for index, episode in enumerate(episodes):
            save(run_dir / "episodes" / f"{index:04d}.json", episode)
        save(run_dir / "task_order.json", [episode["task_id"] for episode in episodes])
        save_lines(
            run_dir / "results.jsonl",
            [
                {key: value for key, value in episode.items() if key not in {"calls", "trajectory"}}
                for episode in episodes
            ],
        )
        save_lines(run_dir / "memory.jsonl", bank)
        save(
            run_dir / "checkpoint.json",
            {"fingerprint": manifest["fingerprint"], "results": episodes, "bank": bank},
        )
        summary = {
            "benchmark": True,
            "seed": seed,
            "tasks": 6,
            "successes": sum(outcomes[seed]),
            "success_rate": statistics.mean(outcomes[seed]),
        }
        save(run_dir / "summary.json", summary)
        runs.append(summary)
    save(root / "manifest.json", manifest)
    rates = [run["success_rate"] for run in runs]
    save(
        root / "summary.json",
        {
            "benchmark": True,
            "method": "jitmem",
            "split": "valid_seen",
            "runs": runs,
            "success_rate_mean": statistics.mean(rates),
            "success_rate_std": statistics.stdev(rates),
        },
    )
    return root


@pytest.fixture
def arms(tmp_path):
    first = create_arm(
        tmp_path / "filtered",
        "judge",
        {
            0: [True, True, False, False, True, False],
            1: [True, False, False, False, True, False],
            2: [True, False, True, False, True, False],
        },
        unknown_usage=True,
    )
    second = create_arm(
        tmp_path / "full",
        "all",
        {
            0: [True, False, False, False, True, False],
            1: [True, True, True, False, True, False],
            2: [True, True, False, False, True, False],
        },
    )
    return first, second


def compare(arms):
    return analysis.build_comparison(*arms, expected_tasks=6, expected_batch_size=2)


def rewrite_episode(root, seed, index, change, *, rebuild_bank=False):
    run_dir = root / f"seed_{seed}"
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    episodes = checkpoint["results"]
    change(episodes[index])
    save(run_dir / "episodes" / f"{index:04d}.json", episodes[index])
    save_lines(
        run_dir / "results.jsonl",
        [
            {key: value for key, value in episode.items() if key not in {"calls", "trajectory"}}
            for episode in episodes
        ],
    )
    if rebuild_bank:
        checkpoint["bank"] = [
            entry_from_episode(episode) for episode in episodes if episode["stored"]
        ]
        save_lines(run_dir / "memory.jsonl", checkpoint["bank"])
    save(run_dir / "checkpoint.json", checkpoint)


def test_signed_three_seed_comparison_storage_and_nullable_usage(arms, tmp_path):
    report = compare(arms)
    assert report["seeds"] == [0, 1, 2]
    assert [run["delta_pp"] for run in report["per_seed"]] == pytest.approx([100 / 6, -100 / 3, 0])
    assert report["across_seed"]["delta_pp"]["mean"] == pytest.approx(-100 / 18)
    assert report["per_seed"][0]["paired_counts"]["filtered_only"] == 1
    for run in report["per_seed"]:
        filtered, full = (run[key] for key in ("filtered", "full"))
        assert filtered["storage"]["final_bank_size"] == 3
        assert full["storage"]["final_bank_size"] == 6
        assert filtered["storage"]["judge_failed_stored_entries"] == 0
        assert full["storage"]["judge_failed_stored_entries"] == 3
        assert filtered["storage"]["retrieval"]["judge_failed_references"] == 0
        assert full["storage"]["retrieval"]["judge_failed_references"] > 0
        assert len(full["storage"]["batch_growth"]) == 3
        assert full["storage"]["batch_growth"][-1]["bank_after"] == 6
        for role in filtered["usage_by_role"].values():
            assert role["prompt_tokens"] is None and role["completion_tokens"] is None
        assert full["usage_by_role"]["curator"]["prompt_tokens"] == 18
    assert report["per_seed"][0]["filtered"]["storage"]["native_failed_stored_entries"] == 1
    assert report["per_seed"][0]["filtered"]["native_vs_judge"]["false_positive"] == 1
    assert report["statistical_scope"]["unique_task_count"] == 6
    assert report["statistical_scope"]["repeated_episode_pairs"] == 18
    assert report["statistical_scope"]["pooled_significance_test"] is None
    assert report["overall"]["filtered"]["episodes"] == 18
    assert report["overall"]["filtered"]["calls"] == 54
    assert report["overall"]["filtered"]["all_role_prompt_tokens"] is None
    assert report["overall"]["filtered"]["all_role_completion_tokens"] is None
    assert report["overall"]["filtered"]["all_role_input_output_tokens"] is None
    assert report["overall"]["full"]["all_role_prompt_tokens"] == 162
    assert report["overall"]["full"]["all_role_completion_tokens"] == 108
    assert report["overall"]["full"]["all_role_input_output_tokens"] == 270
    assert report["overall"]["full"]["usage_by_role"]["judge"]["calls"] == 18
    filtered_rates = [300 / 6, 200 / 6, 300 / 6]
    assert report["across_seed"]["filtered_sr_percent"]["mean"] == pytest.approx(
        statistics.mean(filtered_rates)
    )
    assert report["across_seed"]["filtered_sr_percent"]["std"] == pytest.approx(
        statistics.stdev(filtered_rates)
    )
    assert report["across_seed"]["filtered_mean_decisions"] == {"mean": 1, "std": 0}
    assert report["across_seed"]["full_mean_environment_steps"] == {"mean": 1, "std": 0}
    assert report["sources"]["filtered"]["files"][-1]["path"].endswith("checkpoint.json")
    output = tmp_path / "comparison"
    assert (
        analysis.main(
            [
                str(arms[0]),
                str(arms[1]),
                "--expected-tasks",
                "6",
                "--expected-batch-size",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    markdown = (output / "storage_ablation.md").read_text()
    assert "filtered-minus-full" in markdown and "负值有利于 full" in markdown
    assert "未知" in markdown and "不补成 0" in markdown
    assert "没有本地训练" in markdown and "不计算 pooled p-value" in markdown
    assert "## 三轮汇总" in markdown
    assert "| filtered | 44.44 ± 9.62% | 1.00 ± 0.00 | 1.00 ± 0.00 | 54 |" in markdown
    assert (
        "| full | 50.00 ± 16.67% | 1.00 ± 0.00 | 1.00 ± 0.00 | 54 | 162 | 108 | 270 |" in markdown
    )
    assert json.loads((output / "storage_ablation.json").read_text())["benchmark"] is True


@pytest.mark.parametrize("field", ["source_hashes", "python", "platform", "prompt_source", "bm25"])
def test_rejects_implementation_or_runtime_mismatch(arms, field):
    path = arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = {"different": "value"} if isinstance(manifest[field], dict) else "different"
    save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="Matched manifests/config mismatch"):
        compare(arms)


@pytest.mark.parametrize("role", ["executor", "curator"])
def test_rejects_model_or_sampling_difference(arms, role):
    path = arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"][role]["temperature"] = 0
    save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="Matched manifests/config mismatch"):
        compare(arms)


def test_rejects_extra_config_difference_and_warm_start(arms):
    path = arms[1] / "manifest.json"
    original = json.loads(path.read_text())
    altered = copy.deepcopy(original)
    altered["config"]["executor"]["extra_body"] = {"reasoning_effort": "high"}
    save(path, altered)
    with pytest.raises(analysis.AnalysisError, match="Matched manifests/config mismatch"):
        compare(arms)
    original["config"]["experiment"]["warm_start"] = "/dataset/train-memory.jsonl"
    save(path, original)
    with pytest.raises(analysis.AnalysisError, match="warm start"):
        compare(arms)


@pytest.mark.parametrize("filename", ["memory.jsonl", "checkpoint.json", "episodes/0000.json"])
def test_rejects_missing_persisted_evidence(arms, filename):
    (arms[1] / "seed_0" / filename).unlink()
    with pytest.raises(analysis.AnalysisError, match="Cannot read"):
        compare(arms)


def test_rejects_partial_results_infrastructure_error_and_default_small_arm(arms):
    path = arms[1] / "seed_0" / "results.jsonl"
    original = path.read_text()
    path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    with pytest.raises(analysis.AnalysisError, match="full manifest"):
        compare(arms)
    path.write_text(original)
    save(arms[1] / "seed_0" / "error.json", {"error": "transport interrupted"})
    with pytest.raises(analysis.AnalysisError, match="unresolved run error"):
        compare(arms)
    (arms[1] / "seed_0" / "error.json").unlink()
    with pytest.raises(analysis.AnalysisError, match="expected 140 manifest tasks"):
        analysis.build_comparison(*arms)


@pytest.mark.parametrize("arm_index", [0, 1])
def test_rejects_gate_that_disagrees_with_judge_or_all_policy(arms, arm_index):
    root = arms[arm_index]
    rewrite_episode(
        root, 0, 5, lambda episode: episode.update(stored=not episode["stored"]), rebuild_bank=True
    )
    with pytest.raises(analysis.AnalysisError, match="storage gate"):
        compare(arms)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("label", "stored judge label"),
        ("raw", "stored raw trajectory"),
        ("order", "final bank order/count"),
        ("checkpoint", "final checkpoint"),
    ],
)
def test_rejects_corrupted_final_memory_or_checkpoint(arms, mutation, match):
    run_dir = arms[1] / "seed_0"
    memory_path = run_dir / "memory.jsonl"
    memory = [json.loads(line) for line in memory_path.read_text().splitlines()]
    if mutation == "label":
        memory[0]["judge_success"] = None
    elif mutation == "raw":
        memory[0]["initial_observation"] = "discarded raw content"
    elif mutation == "order":
        memory.reverse()
    else:
        checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
        checkpoint["results"].pop()
        save(run_dir / "checkpoint.json", checkpoint)
    if mutation != "checkpoint":
        save_lines(memory_path, memory)
    with pytest.raises(analysis.AnalysisError, match=match):
        compare(arms)


@pytest.mark.parametrize("task_id", ["task-1", "absent", "task-5"])
def test_rejects_retrieval_of_current_future_or_missing_memory(arms, task_id):
    # Seed0's second batch is task1/task0; task5 is in its future third batch.
    rewrite_episode(arms[1], 0, 2, lambda episode: episode["retrieved"][0].update(task_id=task_id))
    with pytest.raises(analysis.AnalysisError, match="unavailable/current/future/duplicate"):
        compare(arms)


@pytest.mark.parametrize("arm_index", [0, 1])
def test_rejects_missing_full_labels_or_filtered_label_leakage(arms, arm_index):
    def alter(episode):
        message = episode["calls"][0]["messages"][1]
        if arm_index:
            message["content"] = (
                message["content"]
                .replace("\nExecutor judge label: success", "")
                .replace("\nExecutor judge label: failure", "")
            )
        else:
            message["content"] = message["content"].replace(
                "\nTrajectory:", "\nExecutor judge label: success\nTrajectory:"
            )

    rewrite_episode(arms[arm_index], 0, 2, alter)
    with pytest.raises(analysis.AnalysisError, match="visible labels disagree with policy"):
        compare(arms)


def test_zero_retrieval_denominator_is_null_not_false_failure_rate(tmp_path):
    outcomes = {seed: [False] * 6 for seed in (0, 1, 2)}
    first = create_arm(tmp_path / "filtered", "judge", outcomes, all_rejected=True)
    second = create_arm(tmp_path / "full", "all", outcomes, all_rejected=True)
    report = compare((first, second))
    storage = report["per_seed"][0]["filtered"]["storage"]
    assert storage["stored_entries"] == storage["retrieval"]["references"] == 0
    assert storage["native_failed_stored_fraction"] is None
    assert storage["retrieval"]["native_failed_fraction"] is None
    assert report["per_seed"][0]["full"]["storage"]["retrieval"]["judge_failed_fraction"] == 1
    assert "—（无分母）" in analysis.render_markdown(report)


def test_rejects_declared_subset_of_three_required_seeds(arms):
    root = arms[1]
    manifest_path, summary_path = root / "manifest.json", root / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest["config"]["experiment"]["seeds"] = [0, 1]
    summary["runs"] = summary["runs"][:2]
    rates = [run["success_rate"] for run in summary["runs"]]
    summary["success_rate_mean"] = statistics.mean(rates)
    summary["success_rate_std"] = statistics.stdev(rates)
    save(manifest_path, manifest)
    save(summary_path, summary)
    with pytest.raises(analysis.AnalysisError, match="expected seeds"):
        compare(arms)


def test_default_batch_size_rejects_complete_alternate_batch_protocol(arms):
    with pytest.raises(analysis.AnalysisError, match="unexpected batch_size"):
        analysis.build_comparison(*arms, expected_tasks=6)


def test_default_workers_rejects_both_arms_matched_serial_protocol(arms):
    for root in arms:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"]["experiment"]["workers"] = manifest["workers"] = 1
        manifest["process_start_method"] = None
        manifest["batch_execution"] = "sequential execution with a shared frozen bank per batch"
        save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="unexpected workers"):
        compare(arms)


def test_rejects_episode_that_differs_from_committed_result(arms):
    path = arms[1] / "seed_0" / "episodes/0000.json"
    episode = json.loads(path.read_text())
    episode["success"] = not episode["success"]
    save(path, episode)
    with pytest.raises(analysis.AnalysisError, match="episode differs from committed result"):
        compare(arms)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "by_role"])
def test_rejects_usage_that_disagrees_with_recorded_calls(arms, field):
    def alter(episode):
        if field == "by_role":
            episode["usage"][field]["judge"]["calls"] = 99
        else:
            episode["usage"][field] += 1

    rewrite_episode(arms[1], 0, 0, alter)
    with pytest.raises(analysis.AnalysisError, match="aggregate usage disagrees"):
        compare(arms)


def test_one_missing_completion_count_preserves_known_inputs_but_null_total(arms):
    def omit(episode):
        episode["calls"][-1]["completion_tokens"] = None
        episode["usage"]["completion_tokens"] = None
        episode["usage"]["by_role"]["judge"]["completion_tokens"] = None

    rewrite_episode(arms[1], 0, 0, omit)
    report = compare(arms)
    totals = report["overall"]["full"]
    assert totals["all_role_prompt_tokens"] == 162
    assert totals["all_role_completion_tokens"] is None
    assert totals["all_role_input_output_tokens"] is None
    assert totals["usage_by_role"]["curator"]["completion_tokens"] == 36
    assert totals["usage_by_role"]["judge"]["completion_tokens"] is None


def test_rejects_claimed_complete_usage_when_call_count_is_unknown(arms):
    rewrite_episode(arms[0], 0, 0, lambda episode: episode["usage"].update(complete=True))
    with pytest.raises(analysis.AnalysisError, match="usage completeness disagrees"):
        compare(arms)


def full_seen_protocol():
    tasks = {
        f"{task_type}-{index}": {"task_type": task_type}
        for task_type, count in analysis.SEEN_TASK_COUNTS.items()
        for index in range(count)
    }
    return tasks, {"task_types": [1, 2, 3, 4, 5, 6], "annotation_index": 0}


def test_accepts_exact_full_seen140_task_distribution():
    tasks, environment = full_seen_protocol()
    assert len(tasks) == 140
    analysis.validate_complete_seen_manifest(tasks, environment, "seen140")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("types", "all six task_types"),
        ("annotation", "annotation_index=0"),
        ("boolean_annotation", "annotation_index=0"),
        ("counts", "task-type counts"),
    ],
)
def test_full_seen140_rejects_wrong_types_annotation_and_balanced_wrong_counts(mutation, match):
    tasks, environment = full_seen_protocol()
    if mutation == "types":
        environment["task_types"] = [1, 2, 3, 4, 5]
    elif mutation == "annotation":
        environment["annotation_index"] = 1
    elif mutation == "boolean_annotation":
        environment["annotation_index"] = False
    else:
        # Preserve the total 140 while moving one task to a wrong category.
        tasks["pick_and_place_simple-0"]["task_type"] = "look_at_obj_in_light"
    with pytest.raises(analysis.AnalysisError, match=match):
        analysis.validate_complete_seen_manifest(tasks, environment, "seen140")
