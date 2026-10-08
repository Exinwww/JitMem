import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import pytest

import jitmem.evaluation as evaluation_module
from jitmem.config import EnvironmentConfig, ExperimentConfig, ModelConfig, RunConfig
from jitmem.environments import FakeHouseholdEnvironment, fake_tasks
from jitmem.evaluation import evaluate, initial_bank
from jitmem.memory import Trajectory
from jitmem.pipeline import MockChatClient, Pipeline


def controlled_process_worker(config, task, bank_entries):
    """Importable spawned-process fixture for latency and temporary outages."""
    control_path = Path(config.experiment.output_dir) / "worker_control.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    if control.get("fail_task") == task.task_id:
        raise RuntimeError("temporary provider outage in isolated worker")
    if control.get("delay_task") == task.task_id:
        time.sleep(0.4)
    # Spawn imports the unmodified module in each child; the parent's monkeypatch
    # only selects this importable fixture as the submitted worker function.
    result, stored = evaluation_module._run_episode_worker(config, task, bank_entries)
    result["credential_inherited"] = bool(os.environ.get("JITMEM_PARALLEL_TEST_CREDENTIAL"))
    return result, stored


def configuration(tmp_path):
    return RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(seeds=[0, 1], batch_size=2, output_dir=str(tmp_path)),
    )


def test_batches_share_bank_and_runs_cold_start(tmp_path):
    config = configuration(tmp_path)
    client = MockChatClient()
    summary = evaluate(
        config, fake_tasks(5), Pipeline(config, client, client), FakeHouseholdEnvironment
    )
    assert not summary["benchmark"]
    for seed in [0, 1]:
        checkpoint = json.loads((tmp_path / f"seed_{seed}" / "checkpoint.json").read_text())
        assert [r["memory_size_before"] for r in checkpoint["results"]] == [0, 0, 2, 2, 4]
        assert [len(r["retrieved"]) for r in checkpoint["results"]] == [0, 0, 2, 2, 3]
        assert len(checkpoint["bank"]) == 5
    resumed = evaluate(
        config,
        fake_tasks(5),
        Pipeline(config, client, client),
        FakeHouseholdEnvironment,
        resume=True,
    )
    assert resumed == summary
    with pytest.raises(ValueError, match="already contains"):
        evaluate(config, fake_tasks(5), Pipeline(config, client, client), FakeHouseholdEnvironment)
    changed = replace(config, experiment=replace(config.experiment, retrieval_k=0))
    with pytest.raises(ValueError, match="different config"):
        evaluate(
            changed,
            fake_tasks(5),
            Pipeline(changed, client, client),
            FakeHouseholdEnvironment,
            resume=True,
        )


def test_failed_batch_resumes_from_last_complete_batch(tmp_path):
    config = replace(
        configuration(tmp_path),
        experiment=ExperimentConfig(seeds=[0], batch_size=2, output_dir=str(tmp_path)),
    )
    client = MockChatClient()

    class FailingPipeline(Pipeline):
        calls = 0

        def run_episode(self, task, environment, bank):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("temporary provider outage")
            return super().run_episode(task, environment, bank)

    with pytest.raises(RuntimeError, match="outage"):
        evaluate(
            config, fake_tasks(5), FailingPipeline(config, client, client), FakeHouseholdEnvironment
        )
    checkpoint = json.loads((tmp_path / "seed_0" / "checkpoint.json").read_text())
    assert len(checkpoint["results"]) == len(checkpoint["bank"]) == 2
    assert (tmp_path / "seed_0" / "error.json").exists()
    summary = evaluate(
        config,
        fake_tasks(5),
        Pipeline(config, client, client),
        FakeHouseholdEnvironment,
        resume=True,
    )
    assert summary["runs"][0]["tasks"] == 5
    assert not (tmp_path / "seed_0" / "error.json").exists()


def test_warm_start_rejects_test_split_or_overlapping_tasks(tmp_path):
    path = tmp_path / "warm.jsonl"
    entry = Trajectory("overlap", "task", "pick", "valid_seen", "observation", [], True)
    path.write_text(json.dumps(entry.to_dict()) + "\n")
    config = replace(RunConfig(), experiment=ExperimentConfig(warm_start=str(path)))
    with pytest.raises(ValueError, match="train-split"):
        initial_bank(config, set())
    entry.split = "train"
    path.write_text(json.dumps(entry.to_dict()) + "\n")
    with pytest.raises(ValueError, match="overlaps"):
        initial_bank(config, {"overlap"})
    assert len(initial_bank(config, set()).entries) == 1


def test_resume_rejects_modified_warm_start_memory(tmp_path):
    path = tmp_path / "warm.jsonl"
    entry = Trajectory("train_task", "put a mug away", "pick", "train", "observation", [], True)
    path.write_text(json.dumps(entry.to_dict()) + "\n")
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(
            seeds=[0], batch_size=2, output_dir=str(tmp_path / "run"), warm_start=str(path)
        ),
    )
    client = MockChatClient()
    evaluate(config, fake_tasks(1), Pipeline(config, client, client), FakeHouseholdEnvironment)
    entry.task_description = "a modified training experience"
    path.write_text(json.dumps(entry.to_dict()) + "\n")
    with pytest.raises(ValueError, match="different config"):
        evaluate(
            config,
            fake_tasks(1),
            Pipeline(config, client, client),
            FakeHouseholdEnvironment,
            resume=True,
        )


def test_parallel_processes_freeze_memory_and_commit_in_task_order(tmp_path, monkeypatch, capsys):
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(
            seeds=[0, 1], batch_size=2, workers=2, output_dir=str(tmp_path)
        ),
        executor=ModelConfig(api_key_env="JITMEM_PARALLEL_TEST_CREDENTIAL"),
    )
    tasks = fake_tasks(5)
    ordered = list(tasks)
    random.Random(0).shuffle(ordered)
    (tmp_path / "worker_control.json").write_text(json.dumps({"delay_task": ordered[0].task_id}))
    monkeypatch.setenv("JITMEM_PARALLEL_TEST_CREDENTIAL", "private-test-key")
    monkeypatch.setattr(evaluation_module, "_run_episode_worker", controlled_process_worker)
    client = MockChatClient()

    def forbidden_parent_environment():
        raise AssertionError("Parallel evaluation must construct each environment in a child")

    summary = evaluate(
        config, tasks, Pipeline(config, client, client), forbidden_parent_environment
    )
    assert summary["success_rate_mean"] == 1.0
    manifest_text = (tmp_path / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["workers"] == 2
    assert manifest["process_start_method"] == "spawn"
    assert "parallel" in manifest["batch_execution"]
    assert "private-test-key" not in manifest_text
    completions = capsys.readouterr().out
    assert completions.index("seed=0 task=2/5") < completions.index("seed=0 task=1/5")
    for seed in [0, 1]:
        run_dir = tmp_path / f"seed_{seed}"
        checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
        order = json.loads((run_dir / "task_order.json").read_text())
        results = checkpoint["results"]
        assert [r["task_id"] for r in results] == order
        assert [entry["task_id"] for entry in checkpoint["bank"]] == order
        assert [r["memory_size_before"] for r in results] == [0, 0, 2, 2, 4]
        assert [r["retrieved"] for r in results[:2]] == [[], []]
        assert [r["task_id"] for r in results[2]["retrieved"]] == order[:2]
        assert all(r["worker_pid"] != os.getpid() for r in results)
        assert all(r["credential_inherited"] for r in results)
        for index, expected_id in enumerate(order):
            episode = json.loads((run_dir / "episodes" / f"{index:04d}.json").read_text())
            assert episode["task_id"] == expected_id


def test_parallel_failed_batch_keeps_checkpoint_and_resume_retries_batch(tmp_path, monkeypatch):
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(seeds=[0], batch_size=2, workers=2, output_dir=str(tmp_path)),
    )
    tasks = fake_tasks(5)
    ordered = list(tasks)
    random.Random(0).shuffle(ordered)
    control_file = tmp_path / "worker_control.json"
    control_file.write_text(json.dumps({"fail_task": ordered[3].task_id}))
    monkeypatch.setattr(evaluation_module, "_run_episode_worker", controlled_process_worker)
    client = MockChatClient()
    with pytest.raises(RuntimeError, match="provider outage"):
        evaluate(config, tasks, Pipeline(config, client, client), FakeHouseholdEnvironment)
    checkpoint_file = tmp_path / "seed_0" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_file.read_text())
    committed_ids = [r["task_id"] for r in checkpoint["results"]]
    assert committed_ids == [task.task_id for task in ordered[:2]]
    assert len(checkpoint["bank"]) == 2
    error = json.loads((tmp_path / "seed_0" / "error.json").read_text())
    assert error["committed_tasks"] == 2
    assert error["task_id"] == ordered[3].task_id
    assert error["checkpoint"] == str(checkpoint_file)
    assert not (tmp_path / "summary.json").exists()
    control_file.unlink()
    summary = evaluate(
        config, tasks, Pipeline(config, client, client), FakeHouseholdEnvironment, resume=True
    )
    assert summary["runs"][0]["tasks"] == 5
    resumed = json.loads(checkpoint_file.read_text())
    assert resumed["results"][:2] == checkpoint["results"]
    assert [r["memory_size_before"] for r in resumed["results"]] == [0, 0, 2, 2, 4]
    assert [entry["task_id"] for entry in resumed["bank"]] == [task.task_id for task in ordered]
    assert not (tmp_path / "seed_0" / "error.json").exists()


def test_parallel_rejects_custom_pipeline_instead_of_ignoring_it(tmp_path):
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(seeds=[0], workers=2, output_dir=str(tmp_path)),
    )

    class CustomPipeline(Pipeline):
        pass

    client = MockChatClient()
    with pytest.raises(ValueError, match="standard Pipeline"):
        evaluate(
            config, fake_tasks(1), CustomPipeline(config, client, client), FakeHouseholdEnvironment
        )
