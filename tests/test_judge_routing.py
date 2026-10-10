"""Role routing at the HTTP, CLI, and spawned-worker boundaries; no model APIs."""

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from test_http_pipeline import scripted_endpoint

import jitmem.cli as cli_module
from jitmem.api import ChatClient
from jitmem.config import EnvironmentConfig, ExperimentConfig, ModelConfig, RunConfig
from jitmem.environments import FakeHouseholdEnvironment, fake_tasks
from jitmem.evaluation import _run_episode_worker, evaluate
from jitmem.memory import MemoryBank
from jitmem.pipeline import MockChatClient, Pipeline


def routed_config(url, *, independent=True, output_dir="outputs/fixture"):
    executor = ModelConfig(
        base_url=url,
        model="gpt-6.1-sol" if independent else "gpt-5.5",
        api_key_env="JITMEM_ROUTING_EXECUTOR_TEST_KEY",
        max_tokens=4096,
        retries=0,
    )
    return RunConfig(
        environment=EnvironmentConfig(backend="alfworld"),
        experiment=ExperimentConfig(seeds=[0], store_policy="all", output_dir=output_dir),
        executor=executor,
        curator=replace(executor, model="gpt-6.1-sol", max_tokens=8192),
        judge=(
            replace(executor, model="gpt-5.5", api_key_env="JITMEM_ROUTING_JUDGE_TEST_KEY")
            if independent
            else None
        ),
    )


def fixture_replies():
    actions = [
        "go to desk 1",
        "take mug 1 from desk 1",
        "go to cabinet 1",
        "open cabinet 1",
        "move mug 1 to cabinet 1",
    ]
    return (
        ["Fixture guidance."]
        + [f"<action>{action}</action>" for action in actions]
        + [
            json.dumps(
                {"success": False, "rationale": "Fixture negative label.", "evidence_step": -1}
            )
        ]
    )


def fixture_native_worker(config, task):
    # Exercise the actual worker's client construction in a fresh spawned process.
    # Replace only its household simulator: this is a software fixture, not a score.
    with patch("jitmem.environments.ALFWorldEnvironment", FakeHouseholdEnvironment):
        return _run_episode_worker(config, task, [])


@pytest.mark.parametrize("spawned", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_http_models_parameters_credentials_and_judge_labels(monkeypatch, spawned, independent):
    monkeypatch.setenv("JITMEM_ROUTING_EXECUTOR_TEST_KEY", "executor-fixture-credential")
    monkeypatch.setenv("JITMEM_ROUTING_JUDGE_TEST_KEY", "judge-fixture-credential")
    with scripted_endpoint(fixture_replies()) as (url, records, pending):
        config = routed_config(url, independent=independent)
        if spawned:
            with ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                result, stored = pool.submit(
                    fixture_native_worker, config, fake_tasks(1)[0]
                ).result(timeout=30)
            assert result["worker_pid"] != os.getpid()
            assert stored["judge_success"] is False
        else:
            executor = ChatClient(config.executor)
            judge = ChatClient(config.judge) if independent else None
            pipeline = Pipeline(config, executor, ChatClient(config.curator), judge)
            if not independent:
                assert pipeline.judge is executor
            env = FakeHouseholdEnvironment()
            try:
                result, stored = pipeline.run_episode(fake_tasks(1)[0], env, MemoryBank())
            finally:
                env.close()
            assert stored.judge_success is False
        assert not pending
    assert len(records) == 7
    assert [call["role"] for call in result["calls"]] == ["curator"] + ["executor"] * 5 + ["judge"]
    assert result["success"] is True and result["reward"] == 1.0
    assert result["judge"]["success"] is False and result["stored"] is True
    assert result["usage"]["by_role"]["judge"]["calls"] == 1
    for index, record in enumerate(records):
        payload = record["payload"]
        assert record["path"] == "/v1/chat/completions"
        assert payload["model"] == (
            "gpt-6.1-sol" if index == 0 else ("gpt-5.5" if index == 6 else config.executor.model)
        )
        assert payload["max_tokens"] == (8192 if index == 0 else 4096)
        assert payload["temperature"] == 1.0
        assert record["headers"]["Authorization"] == (
            "Bearer judge-fixture-credential"
            if index == 6 and independent
            else "Bearer executor-fixture-credential"
        )
        assert payload["messages"] == result["calls"][index]["messages"]
    trace = records[-1]["payload"]["messages"][-1]["content"]
    assert '"success":' not in trace and '"reward":' not in trace
    assert "judge_success" not in trace


def test_explicit_judge_cannot_silently_fall_back_to_executor():
    config = routed_config("http://127.0.0.1:1/v1")
    client = MockChatClient()
    with pytest.raises(ValueError, match="requires a judge client"):
        Pipeline(config, client, client)
    no_memory = replace(config, experiment=replace(config.experiment, method="no-memory"))
    assert Pipeline(no_memory, client, client).judge is client


def test_cli_constructs_independent_judge_and_default_reuses_executor(monkeypatch, tmp_path):
    for independent in [False, True]:
        config = routed_config("http://127.0.0.1:1/v1", independent=independent)
        created = []

        class Client(MockChatClient):
            def __init__(self, settings):
                self.settings = settings
                created.append(self)

        def fake_evaluate(settings, tasks, pipeline, factory, **kwargs):
            assert settings == config and len(tasks) == 1
            assert pipeline.executor.settings == config.executor
            assert pipeline.curator.settings == config.curator
            if independent:
                assert pipeline.judge.settings == config.judge
                assert pipeline.judge is not pipeline.executor
            else:
                assert pipeline.judge is pipeline.executor
            return {"benchmark": False, "success_rate_mean": 1.0, "success_rate_std": 0.0}

        monkeypatch.setattr(cli_module, "load_config", lambda _path: config)
        monkeypatch.setattr(cli_module, "_tasks", lambda _config: fake_tasks(1))
        monkeypatch.setattr(cli_module, "ChatClient", Client)
        monkeypatch.setattr(cli_module, "evaluate", fake_evaluate)
        monkeypatch.setattr(sys, "argv", ["jitmem", "evaluate"])
        cli_module.main()
        assert len(created) == (3 if independent else 2)


def test_resume_refuses_changed_judge_even_when_executor_unchanged(tmp_path):
    config = replace(
        routed_config("http://127.0.0.1:1/v1", output_dir=str(tmp_path)),
        environment=EnvironmentConfig(backend="mock"),
    )
    client = MockChatClient()
    evaluate(
        config, fake_tasks(1), Pipeline(config, client, client, client), FakeHouseholdEnvironment
    )
    manifest = json.loads((Path(tmp_path) / "manifest.json").read_text())
    assert manifest["config"]["judge"]["model"] == "gpt-5.5"
    changed = replace(config, judge=replace(config.judge, model="different-fixture-judge"))
    with pytest.raises(ValueError, match="different config"):
        evaluate(
            changed,
            fake_tasks(1),
            Pipeline(changed, client, client, client),
            FakeHouseholdEnvironment,
            resume=True,
        )
