import json
from dataclasses import replace

import pytest

from jitmem.api import ChatResult
from jitmem.config import EnvironmentConfig, ExperimentConfig, RunConfig
from jitmem.environments import FakeHouseholdEnvironment, fake_tasks
from jitmem.evaluation import summarize
from jitmem.memory import MemoryBank, Trajectory, Turn
from jitmem.pipeline import MockChatClient, Pipeline, parse_action, parse_judgment


def test_retrieval_indexes_descriptions_only_and_preserves_raw_trace():
    first = Trajectory("a", "clean a bowl", "clean", "train", "apple apple apple", [])
    last = Trajectory(
        "b",
        "heat an apple",
        "heat",
        "train",
        "",
        [
            Turn("first", "heat apple", "response", "heated successfully"),
        ],
    )
    bank = MemoryBank([first, last])
    assert bank.retrieve("heat apple", 1)[0][0] is last
    assert bank.retrieve("anything", 0) == []
    recovered = Trajectory.from_dict(json.loads(json.dumps(last.to_dict())))
    assert "heated successfully" in recovered.render()
    assert recovered.turns[0].response == "response"
    assert MemoryBank().retrieve("task", 3) == []


@pytest.mark.parametrize(
    "response",
    ["go to desk 1", "<action>invented</action>", "<action>look</action><action>look</action>"],
)
def test_invalid_actions_are_never_silently_substituted(response):
    with pytest.raises(ValueError):
        parse_action(response, ["look"])


def test_judge_does_not_accept_truthy_strings():
    assert parse_judgment('{"success":"false"}')["success"] is False
    assert parse_judgment("not json")["parse_error"] is True
    assert parse_judgment('```json\n{"success":true}\n```')["success"] is True


def test_full_trajectory_and_payload_are_separated_from_persistent_bank():
    client = MockChatClient()
    pipeline = Pipeline(RunConfig(), client, client)
    environment = FakeHouseholdEnvironment()
    result, stored = pipeline.run_episode(fake_tasks(1)[0], environment, MemoryBank())
    assert result["success"] and stored.judge_success
    assert len(stored.turns) == 5
    assert "You put the mug" in stored.render()
    assert "payload" not in stored.to_dict()
    assert "success" not in stored.to_dict()
    assert "reward" not in stored.to_dict()
    assert result["usage"]["by_role"]["executor"]["calls"] == 5
    assert result["usage"]["by_role"]["curator"]["calls"] == 1
    assert result["usage"]["by_role"]["judge"]["calls"] == 1


def test_native_budget_end_is_reported_as_truncation():
    config = RunConfig(experiment=ExperimentConfig(max_steps=1))
    client = MockChatClient()
    result, stored = Pipeline(config, client, client).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(max_steps=1), MemoryBank()
    )
    assert result["environment_done"]
    assert not result["success"]
    assert result["truncated"]
    assert stored is None


def test_invalid_output_consumes_budget_without_environment_mutation():
    class InvalidClient(MockChatClient):
        def complete(self, messages):
            if "Admissible actions:" in messages[-1]["content"]:
                return ChatResult("<action>teleport</action>", 1, 1, 0.0)
            return super().complete(messages)

    client = InvalidClient()
    result, stored = Pipeline(
        RunConfig(experiment=ExperimentConfig(max_steps=2)), client, client
    ).run_episode(fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank())
    assert result["invalid_actions"] == 2
    assert result["environment_steps"] == 0
    assert result["truncated"]
    assert stored is None


def test_summary_ablation_discards_raw_memory_but_keeps_episode_audit():
    config = RunConfig(experiment=ExperimentConfig(method="write-summary"))
    client = MockChatClient()
    result, stored = Pipeline(config, client, client).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank()
    )
    assert stored.summary
    assert stored.turns == [] and stored.initial_observation == ""
    assert len(result["trajectory"]["turns"]) == 5
    assert result["usage"]["by_role"]["distiller"]["calls"] == 1


def test_no_memory_baseline_does_not_invoke_curator_or_judge():
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=replace(ExperimentConfig(), method="no-memory"),
    )
    client = MockChatClient()
    result, stored = Pipeline(config, client, client).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank()
    )
    assert result["success"]
    assert result["payload"] == "" and result["judge"] is None and stored is None
    assert set(result["usage"]["by_role"]) == {"executor"}


@pytest.mark.parametrize("role", ["curator", "executor", "judge", "distiller"])
def test_generation_limit_is_scored_without_restarting_episode(role):
    class BudgetLimitedClient(MockChatClient):
        def complete(self, messages):
            first = messages[0]["content"]
            current_role = (
                "executor"
                if "Admissible actions:" in first
                else "judge"
                if first.startswith("Assess whether")
                else "distiller"
                if first.startswith("Extract at most")
                else "curator"
            )
            answer = super().complete(messages)
            return replace(answer, finish_reason="length") if current_role == role else answer

    client = BudgetLimitedClient()
    config = RunConfig(
        experiment=ExperimentConfig(
            method="write-summary" if role == "distiller" else "jitmem",
            max_steps=2 if role == "executor" else 30,
        )
    )
    result, stored = Pipeline(config, client, client).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank()
    )
    failures = result["generation_failures"]
    assert failures and all(failure["role"] == role for failure in failures)
    if role == "executor":
        assert result["steps"] == result["invalid_actions"] == 2
        assert result["environment_steps"] == 0 and not result["success"]
    else:
        assert result["success"]
    if role == "curator":
        assert result["payload"] == "" and stored is not None
    else:
        assert stored is None
    if role == "judge":
        assert result["judge"]["success"] is False
        assert result["judge"]["generation_error"] is True
    summary = summarize([result])
    assert summary["tasks"] == 1
    assert summary["incomplete_generations"] == len(failures)


def test_missing_usage_stays_unknown_in_episode_and_summary():
    class NoUsageClient(MockChatClient):
        def complete(self, messages):
            return replace(super().complete(messages), prompt_tokens=None, completion_tokens=None)

    client = NoUsageClient()
    result, stored = Pipeline(RunConfig(), client, client).run_episode(
        fake_tasks(1)[0], FakeHouseholdEnvironment(), MemoryBank()
    )
    assert result["success"] and stored is not None
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["completion_tokens"] is None
    assert result["usage"]["complete"] is False
    summary = summarize([result])
    assert summary["prompt_tokens"] is None and summary["completion_tokens"] is None
    assert summary["mean_executor_prompt_tokens"] is None
    assert summary["usage_complete"] is False
