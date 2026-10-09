"""Paper-profile artifact audits use synthetic templates, never author prompt copies."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_storage_ablation_analysis import (
    analysis,
    create_arm,
    entry_from_episode,
    fixture_context,
    rewrite_episode,
    save,
    save_lines,
)


@pytest.fixture
def synthetic_assets(monkeypatch):
    templates = {
        "curator_system": "Synthetic curator system.",
        "curator_user": "Question: {query}\n### Retrieved Memories:\nMemory 1:\n...",
        "executor": (
            "Synthetic executor: {task_description}\nBriefing: {retrieved_context}\n"
            "Count: {step_count}; recent: {history_length}; step: {current_step}\n"
            "{action_history}\nCurrent: {current_observation}\nActions: [{admissible_actions}]."
        ),
        "judge_system": 'Synthetic judge format: {{"success": true}}.',
        "judge_user": "Goal: {task_description}\nRaw trace:\n{trajectory}",
        "distillation": "Synthetic distillation.",
    }
    assets = SimpleNamespace(
        **templates,
        provenance={
            "schema_version": 1,
            "profile": "paper-v1",
            "sources": {"synthetic": {"version": "test-only", "sha256": "source-digest"}},
            "templates": {
                name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in templates.items()
            },
        },
    )
    monkeypatch.setattr(analysis, "load_paper_assets", lambda directory: assets)
    return assets


def manual_executor_message(episode, turn, previous, assets):
    recent = previous[-3:]
    first = len(previous) - len(recent) + 1
    history = "\n".join(
        f"[Observation {number}: '{item['observation']}', Action {number}: '{item['action']}']"
        for number, item in enumerate(recent, first)
    )
    return [
        {
            "role": "user",
            "content": assets.executor.format(
                task_description=episode["task_description"],
                retrieved_context=episode["payload"],
                step_count=len(previous),
                history_length=len(recent),
                action_history=history,
                current_step=len(previous) + 1,
                current_observation=turn["observation"],
                admissible_actions="\n ".join(
                    f"'{command}'" for command in turn["admissible_actions"] if command != "help"
                ),
            ),
        }
    ]


def convert_to_paper(root, assets):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["config"]["experiment"].update(
        {"prompt_profile": "paper-v1", "prompt_assets": "/synthetic/local/assets"}
    )
    manifest["prompt_assets"] = copy.deepcopy(assets.provenance)
    manifest["prompt_source"] = "verified local paper templates"
    save(root / "manifest.json", manifest)
    policy = manifest["config"]["experiment"]["store_policy"]
    for seed in (0, 1, 2):
        directory = root / f"seed_{seed}"
        checkpoint = json.loads((directory / "checkpoint.json").read_text())
        episodes = checkpoint["results"]
        for index, episode in enumerate(episodes):
            initial = episode["trajectory"]["initial_observation"]
            count = 5 if index == 0 else 1
            turns = []
            for number in range(count):
                observation = initial if not number else f"progress {number}"
                turns.append(
                    {
                        "observation": observation,
                        "action": "look",
                        "response": "<action>look</action>",
                        "next_observation": f"progress {number + 1}",
                        "admissible_actions": ["help", "look", "inventory"],
                        "executed": True,
                    }
                )
            episode["trajectory"]["turns"] = turns
            episode["steps"] = episode["environment_steps"] = count
        bank = [entry_from_episode(item) for item in episodes if item["stored"]]
        entries = {item["task_id"]: item for item in bank}
        for index, episode in enumerate(episodes):
            references = [entries[item["task_id"]] for item in episode["retrieved"]]
            context = fixture_context(references, policy) if references else ""
            user = (
                assets.curator_user.partition("Memory 1:")[0].format(
                    query=episode["task_description"]
                )
                + context
            )
            curator = copy.deepcopy(episode["calls"][0])
            curator.update(
                messages=[
                    {"role": "system", "content": assets.curator_system},
                    {"role": "user", "content": user},
                ],
                response=episode["payload"],
                prompt_tokens=1000,
                completion_tokens=100,
            )
            executor = []
            turns = episode["trajectory"]["turns"]
            for number, turn in enumerate(turns):
                call = copy.deepcopy(episode["calls"][1])
                call.update(
                    messages=manual_executor_message(episode, turn, turns[:number], assets),
                    response=turn["response"],
                    prompt_tokens=30,
                    completion_tokens=2,
                )
                executor.append(call)
            judge = copy.deepcopy(episode["calls"][-1])
            judge.update(
                messages=[
                    {"role": "system", "content": assets.judge_system.format()},
                    {
                        "role": "user",
                        "content": assets.judge_user.format(
                            task_description=episode["task_description"],
                            trajectory=analysis.render_trajectory(episode["trajectory"]),
                        ),
                    },
                ],
                response=json.dumps(episode["judge"]),
                prompt_tokens=2000,
                completion_tokens=200,
            )
            episode["calls"] = [curator, *executor, judge]
            episode["usage"] = {
                "prompt_tokens": 3000 + 30 * len(turns),
                "completion_tokens": 300 + 2 * len(turns),
                "complete": True,
                "by_role": {
                    "curator": {"calls": 1, "prompt_tokens": 1000, "completion_tokens": 100},
                    "executor": {
                        "calls": len(turns),
                        "prompt_tokens": 30 * len(turns),
                        "completion_tokens": 2 * len(turns),
                    },
                    "judge": {"calls": 1, "prompt_tokens": 2000, "completion_tokens": 200},
                },
            }
            save(directory / "episodes" / f"{index:04d}.json", episode)
        save_lines(
            directory / "results.jsonl",
            [
                {k: v for k, v in item.items() if k not in {"calls", "trajectory"}}
                for item in episodes
            ],
        )
        save_lines(directory / "memory.jsonl", bank)
        save(directory / "checkpoint.json", {**checkpoint, "results": episodes, "bank": bank})


@pytest.fixture
def paper_arms(tmp_path, synthetic_assets):
    outcomes = {seed: [True, True, False, False, True, False] for seed in (0, 1, 2)}
    arms = tuple(
        create_arm(tmp_path / name, policy, outcomes)
        for name, policy in (("filtered", "judge"), ("full", "all"))
    )
    for root in arms:
        convert_to_paper(root, synthetic_assets)
    return arms


def compare(arms):
    return analysis.build_comparison(*arms, expected_tasks=6, expected_batch_size=2)


def test_original_profile_reconstructs_prompts_and_executor_only_metrics(paper_arms):
    report = compare(paper_arms)
    assert report["prompt_profile"] == "paper-v1"
    assert "semantic paraphrases" not in report["variant"]
    assert report["protocol"]["prompt_assets"]["profile"] == "paper-v1"
    for name in ("filtered", "full"):
        efficiency = report["overall"][name]["paper_efficiency"]
        assert efficiency["mean_executor_input_tokens_k"] == pytest.approx(0.05)
        assert efficiency["mean_executor_output_tokens_k"] == pytest.approx(1 / 300)
        assert efficiency["mean_executor_interaction_turns"] == pytest.approx(10 / 6)
        assert report["overall"][name]["all_role_prompt_tokens"] == 54900
        assert report["across_seed"][f"{name}_mean_executor_input_tokens_k"] == {
            "mean": 0.05,
            "std": 0.0,
        }
    markdown = analysis.render_markdown(report)
    assert "executor-only" in markdown and "Executor input K / task" in markdown
    assert "paper-v1" in markdown and "继承运行时选择" in markdown
    assert "全角色成本与交互诊断" in markdown
    first = json.loads((paper_arms[0] / "seed_0/episodes/0000.json").read_text())
    empty_user = first["calls"][0]["messages"][1]["content"]
    assert empty_user.endswith("### Retrieved Memories:\n")
    assert "No past episodes" not in empty_user
    last_executor = first["calls"][-2]["messages"][0]["content"]
    assert "[Observation 2:" in last_executor and "[Observation 1:" not in last_executor
    assert "Actions: ['look'\n 'inventory']." in last_executor


@pytest.mark.parametrize(
    ("role", "message_index", "change", "error"),
    [
        (
            0,
            1,
            lambda text: text.replace("### Retrieved Memories:", "Retrieved memories:"),
            "memory input",
        ),
        (0, 1, lambda text: text + "No past episodes are available.", "memory input"),
        (0, 0, lambda text: text + "Extra instruction.", "curator system"),
        (1, 0, lambda text: text.replace("Count: 0", "Count: 1"), "executor messages"),
        (1, 0, lambda text: text + "help", "executor messages"),
        (-1, 1, lambda text: text + "Omitted evidence.", "judge messages"),
    ],
)
def test_paper_prompt_tampering_is_rejected(paper_arms, role, message_index, change, error):
    def tamper(episode):
        message = episode["calls"][role]["messages"][message_index]
        message["content"] = change(message["content"])

    rewrite_episode(paper_arms[0], 0, 0, tamper)
    with pytest.raises(analysis.AnalysisError, match=error):
        compare(paper_arms)


@pytest.mark.parametrize("both", [False, True])
def test_asset_provenance_tampering_is_rejected_even_when_arms_match(paper_arms, both):
    roots = paper_arms if both else paper_arms[:1]
    for root in roots:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["prompt_assets"]["templates"]["executor"] = "changed-digest"
        save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="provenance/hashes mismatch"):
        compare(paper_arms)


def test_paper_profile_without_asset_directory_is_rejected(paper_arms):
    path = paper_arms[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["config"]["experiment"]["prompt_assets"]
    save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="requires a prompt_assets"):
        compare(paper_arms)


@pytest.mark.parametrize("kind", ["response", "action", "payload", "verdict", "unexecuted"])
def test_paper_raw_actions_and_consumed_outputs_cannot_be_rewritten(paper_arms, kind):
    def tamper(episode):
        if kind == "response":
            episode["calls"][1]["response"] = "<action>inventory</action>"
        elif kind == "action":
            episode["trajectory"]["turns"][0]["action"] = "inventory"
        elif kind == "payload":
            episode["payload"] = "Extra guidance."
        elif kind == "verdict":
            episode["calls"][-1]["response"] = "{}"
        else:
            episode["trajectory"]["turns"][0]["executed"] = False
            episode["environment_steps"] -= 1

    rewrite_episode(paper_arms[0], 0, 0, tamper, rebuild_bank=True)
    with pytest.raises(analysis.AnalysisError, match="paper"):
        compare(paper_arms)


def test_executor_usage_missing_remains_unknown_without_hiding_other_role_usage(paper_arms):
    def remove_usage(episode):
        episode["calls"][1]["prompt_tokens"] = None
        episode["usage"]["prompt_tokens"] = None
        episode["usage"]["by_role"]["executor"]["prompt_tokens"] = None
        episode["usage"]["complete"] = False

    rewrite_episode(paper_arms[0], 0, 0, remove_usage)
    report = compare(paper_arms)
    assert report["overall"]["filtered"]["paper_efficiency"]["mean_executor_input_tokens_k"] is None
    assert (
        report["overall"]["filtered"]["paper_efficiency"]["mean_executor_output_tokens_k"]
        is not None
    )
    assert report["overall"]["filtered"]["usage_by_role"]["curator"]["prompt_tokens"] == 18000
    assert report["across_seed"]["filtered_mean_executor_input_tokens_k"] == {
        "mean": None,
        "std": None,
    }
    assert "| filtered | 50.00 ± 0.00% | 未知 |" in analysis.render_markdown(report)


def test_bounded_generation_text_is_accepted_with_incomplete_diagnostics(paper_arms):
    def record_length(episode):
        episode["generation_failures"] = []
        for index in (0, 1, len(episode["calls"]) - 1):
            call = episode["calls"][index]
            call.update(incomplete=True, finish_reason="length")
            episode["generation_failures"].append(
                {"role": call["role"], "call_index": index, "finish_reason": "length"}
            )

    rewrite_episode(paper_arms[0], 0, 0, record_length)
    report = compare(paper_arms)
    failures = report["per_seed"][0]["filtered"]["generation_failures"]
    assert failures["calls"] == 3
    assert failures["by_role"] == {"curator": 1, "executor": 1, "judge": 1}
    assert report["per_seed"][0]["filtered"]["paper_efficiency"][
        "mean_executor_interaction_turns"
    ] == pytest.approx(10 / 6)


def test_unknown_prompt_profile_is_rejected(paper_arms):
    path = paper_arms[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["experiment"]["prompt_profile"] = "unverified-prompt"
    save(path, manifest)
    with pytest.raises(analysis.AnalysisError, match="unsupported prompt_profile"):
        compare(paper_arms)
