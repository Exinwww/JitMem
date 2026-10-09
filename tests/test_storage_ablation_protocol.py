"""Offline checks for the paired storage ablation, never benchmark results."""

import copy
import json
import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from jitmem.api import ChatResult
from jitmem.config import EnvironmentConfig, ExperimentConfig, RunConfig
from jitmem.environments import FakeHouseholdEnvironment, fake_tasks
from jitmem.evaluation import evaluate
from jitmem.memory import MemoryBank, Trajectory, Turn
from jitmem.pipeline import MockChatClient, Pipeline


def test_published_ablation_configs_keep_all_experimental_controls_matched():
    root = Path(__file__).resolve().parents[1]
    configs = []
    for arm, expected_policy in [("filtered", "judge"), ("all", "all")]:
        config = tomllib.loads((root / f"configs/storage_{arm}.example.toml").read_text())
        experiment = config["experiment"]
        assert experiment.pop("store_policy") == expected_policy
        assert experiment.pop("output_dir") == f"outputs/storage_{arm}_paper_v1"
        assert experiment["prompt_profile"] == "paper-v1"
        assert experiment.get("warm_start") is None
        assert config["environment"].get("limit") is None
        assert experiment["method"] == "jitmem" and experiment["seeds"] == [0, 1, 2]
        assert config["environment"]["split"] == "valid_seen"
        configs.append(config)
    assert configs[0] == configs[1]


class QueuedClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []

    def complete(self, messages):
        self.messages.append(copy.deepcopy(messages))
        return next(self.answers)


def answer(text, *, finish_reason=None):
    return ChatResult(text, 1, 1, 0.0, finish_reason=finish_reason)


def prior_success():
    return Trajectory(
        "prior-training-episode",
        "put a mug in a cabinet",
        "pick_and_place_simple",
        "train",
        "A mug is on the table.",
        [Turn("A mug is on the table.", "put mug in cabinet", "prior response", "Mug placed.")],
        judge_success=True,
    )


@pytest.mark.parametrize("judged_success", [False, True])
def test_paired_storage_policies_change_only_curator_labels_and_memory_gate(judged_success):
    results = {}
    messages = {}
    for policy in ["judge", "all"]:
        config = RunConfig(
            environment=EnvironmentConfig(backend="mock"),
            experiment=ExperimentConfig(store_policy=policy, max_steps=1),
        )
        executor = QueuedClient(
            [answer("<action>look</action>"), answer(json.dumps({"success": judged_success}))]
        )
        curator = QueuedClient([answer("Use the observed admissible actions.")])
        result, stored = Pipeline(config, executor, curator).run_episode(
            fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank([prior_success()])
        )
        results[policy] = (result, stored)
        messages[policy] = (curator.messages[0], executor.messages)

    filtered_curator, filtered_executor_and_judge = messages["judge"]
    full_curator, full_executor_and_judge = messages["all"]
    assert full_executor_and_judge == filtered_executor_and_judge
    assert full_curator[0] == filtered_curator[0]
    assert "Executor judge label: success" in full_curator[1]["content"]
    assert (
        re.sub(r"\nExecutor judge label: (?:success|failure)", "", full_curator[1]["content"])
        == filtered_curator[1]["content"]
    )
    for field in ["success", "reward", "trajectory", "judge", "payload", "retrieved", "usage"]:
        assert results["all"][0][field] == results["judge"][0][field]
    assert results["all"][1] is not None
    assert (results["judge"][1] is not None) is judged_success
    # Both prompts reached the same executor client; neither includes native scalars.
    assert len(full_executor_and_judge) == 2
    judge_trace = full_executor_and_judge[-1][-1]["content"]
    for forbidden in ['"reward":', '"success":', "native_success", "judge_success"]:
        assert forbidden not in judge_trace


@pytest.mark.parametrize(
    "verdict",
    [answer("malformed judgment"), answer('{"success": true}', finish_reason="length")],
    ids=["malformed-judge", "incomplete-judge"],
)
def test_full_storage_preserves_failed_trace_including_invalid_and_incomplete_decisions(verdict):
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(store_policy="all", max_steps=3),
    )
    executor = QueuedClient(
        [
            answer("<action>look</action>"),
            answer("This response contains no action tag."),
            answer("<action>look</action>", finish_reason="length"),
            verdict,
        ]
    )
    curator = QueuedClient([answer("Fixture briefing.")])
    result, stored = Pipeline(config, executor, curator).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank()
    )
    assert not result["success"] and result["truncated"]
    assert result["environment_steps"] == 1 and result["invalid_actions"] == 2
    assert result["judge"]["success"] is False
    assert stored is not None and stored.judge_success is False
    serialized = stored.to_dict()
    assert serialized["initial_observation"] == result["trajectory"]["initial_observation"]
    assert serialized["turns"] == result["trajectory"]["turns"]
    assert [turn["executed"] for turn in serialized["turns"]] == [True, False, False]
    assert serialized["turns"][1]["response"] == "This response contains no action tag."
    assert "Incomplete model output" in stored.render()
    assert all(turn["admissible_actions"] for turn in serialized["turns"])
    assert "reward" not in serialized and "success" not in serialized
    assert serialized["summary"] is None


class AlternatingJudgeClient(MockChatClient):
    def __init__(self):
        self.verdicts = iter([False, True, False, True, True])

    def complete(self, messages):
        if messages[0]["content"].startswith("Assess whether"):
            return answer(json.dumps({"success": next(self.verdicts)}))
        return super().complete(messages)


def test_failed_episodes_remain_frozen_within_batch_and_receive_labels_in_next_batch(tmp_path):
    runs = {}
    base = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(seeds=[0], batch_size=2, max_steps=1, retrieval_k=3),
    )
    for policy in ["judge", "all"]:
        output = tmp_path / policy
        config = replace(
            base,
            experiment=replace(base.experiment, store_policy=policy, output_dir=str(output)),
        )
        client = AlternatingJudgeClient()
        evaluate(config, fake_tasks(5), Pipeline(config, client, client), FakeHouseholdEnvironment)
        runs[policy] = json.loads((output / "seed_0" / "checkpoint.json").read_text())

    order = [result["task_id"] for result in runs["all"]["results"]]
    assert [result["task_id"] for result in runs["judge"]["results"]] == order
    for policy, checkpoint in runs.items():
        results = checkpoint["results"]
        assert [result["judge"]["success"] for result in results] == [
            False,
            True,
            False,
            True,
            True,
        ]
        assert not any(result["success"] for result in results)
        for index, result in enumerate(results):
            prior_batch_ids = set(order[: (index // 2) * 2])
            assert {item["task_id"] for item in result["retrieved"]} <= prior_batch_ids
        if policy == "all":
            assert [entry["task_id"] for entry in checkpoint["bank"]] == order
            assert [result["memory_size_before"] for result in results] == [0, 0, 2, 2, 4]
            assert [item["task_id"] for item in results[2]["retrieved"]] == order[:2]
            for result in results[2:4]:
                curator_prompt = result["calls"][0]["messages"][-1]["content"]
                assert curator_prompt.count("Executor judge label: failure") == 1
                assert curator_prompt.count("Executor judge label: success") == 1
            assert all(len(entry["turns"]) == 1 for entry in checkpoint["bank"])
        else:
            assert [entry["task_id"] for entry in checkpoint["bank"]] == [
                order[1],
                order[3],
                order[4],
            ]
            assert [result["memory_size_before"] for result in results] == [0, 0, 1, 1, 2]
            assert all(
                "Executor judge label:" not in result["calls"][0]["messages"][-1]["content"]
                for result in results
            )
    # A cold first batch has identical model-visible inputs before the gate can diverge.
    for filtered, full in zip(runs["judge"]["results"][:2], runs["all"]["results"][:2]):
        assert [call["messages"] for call in filtered["calls"]] == [
            call["messages"] for call in full["calls"]
        ]
