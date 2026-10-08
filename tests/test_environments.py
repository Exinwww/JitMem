"""Check episode selection, observation boundaries, and terminal semantics."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jitmem.environments import (
    ALFWorldEnvironment,
    FakeHouseholdEnvironment,
    discover_tasks,
    extract_goal,
    fake_tasks,
)


def _episode(root: Path, name: str, *, task_type="pick_and_place_simple", solvable=True):
    directory = root / "json_2.1.1" / "valid_seen" / name / "trial_1"
    directory.mkdir(parents=True)
    (directory / "traj_data.json").write_text(
        json.dumps(
            {
                "task_id": f"task-{name}",
                "task_type": task_type,
                "turk_annotations": {"anns": [{"task_desc": "Human paraphrase."}]},
            }
        )
    )
    (directory / "game.tw-pddl").write_text(
        json.dumps(
            {
                "solvable": solvable,
                "grammar": 'grammar :: """ {"task": [{"rhs": "Your task is to: put a mug in cabinet."}]} """',
                "walkthrough": ["EXPERT ACTION MUST NOT ESCAPE"],
            }
        )
    )
    return directory


def test_discovery_matches_official_filters_and_sorted_coverage(tmp_path):
    _episode(tmp_path, "z-task")
    _episode(tmp_path, "a-task")
    _episode(tmp_path, "b-unsolvable", solvable=False)
    _episode(tmp_path, "c-movable")
    _episode(tmp_path, "d-Sliced")
    missing = _episode(tmp_path, "e-missing")
    (missing / "game.tw-pddl").unlink()
    _episode(tmp_path, "f-other", task_type="pick_and_place_with_movable_recep")
    tasks = discover_tasks(tmp_path, "valid_seen")
    assert [task.task_id for task in tasks] == ["task-a-task", "task-z-task"]
    assert tasks[0].description == "put a mug in cabinet."
    assert "EXPERT" not in repr(tasks)
    assert discover_tasks(tmp_path, "valid_seen", limit=1) == tasks[:1]
    assert len(discover_tasks(tmp_path / "json_2.1.1", "valid_seen", [1])) == 2
    assert discover_tasks(tmp_path, "valid_seen", ["look_at_obj_in_light"]) == []


def test_discovery_preserves_every_trial(tmp_path):
    directory = _episode(tmp_path, "task-config")
    second_trial = directory.parent / "trial_2"
    second_trial.mkdir()
    for path in directory.iterdir():
        (second_trial / path.name).write_bytes(path.read_bytes())
    assert len(discover_tasks(tmp_path, "valid_seen")) == 2


def test_discovery_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        discover_tasks(tmp_path, "test")
    with pytest.raises(ValueError):
        discover_tasks(tmp_path, "train", limit=-1)
    with pytest.raises(ValueError):
        discover_tasks(tmp_path, "train", task_types=[99])
    with pytest.raises(FileNotFoundError):
        discover_tasks(tmp_path, "train")


def test_fake_environment_success_requires_actual_state_transitions():
    env = FakeHouseholdEnvironment()
    state = env.reset(fake_tasks(1)[0])
    assert not env.benchmark
    assert not state.success
    assert extract_goal(state.observation) == "put a mug in cabinet."
    for _ in range(5):
        state = env.step(state.admissible_actions[0])
    assert state.done and state.success
    assert state.reward == 1.0
    with pytest.raises(RuntimeError):
        env.step("look")
    env.close()
    with pytest.raises(RuntimeError):
        env.step("look")


def test_invalid_commands_consume_budget_without_claiming_success():
    env = FakeHouseholdEnvironment(max_steps=2)
    env.reset(fake_tasks(1)[0])
    state = env.step("invented command")
    assert state.observation == "Nothing happens."
    assert not state.done
    state = env.step("invented command")
    assert state.done and not state.success
    assert state.reward == 0.0


def test_native_adapter_requests_public_info_and_loads_exact_game(tmp_path, monkeypatch):
    directory = _episode(tmp_path, "fixture")
    task = discover_tasks(tmp_path, "valid_seen")[0]
    requests = []
    starts = []

    class StubEnvironment:
        def __init__(self):
            self.closed = False

        def reset(self):
            return {"feedback": "observable", "admissible_commands": ["look"], "won": False}

        def step(self, action):
            return self.reset() | {"feedback": "Nothing happens."}, 0, False

        def close(self):
            self.closed = True

    runtime = StubEnvironment()

    def request_infos(**kwargs):
        requests.append(kwargs)
        return kwargs

    def start(path, infos, wrappers):
        starts.append(path)
        assert wrappers[0](runtime) is runtime
        return runtime

    monkeypatch.setitem(
        sys.modules, "textworld", SimpleNamespace(EnvInfos=request_infos, start=start)
    )
    monkeypatch.setitem(
        sys.modules,
        "alfworld.agents.environment.alfred_tw_env",
        SimpleNamespace(AlfredDemangler=lambda env, shuffle: env),
    )
    env = ALFWorldEnvironment(max_steps=1)
    state = env.reset(task)
    assert starts == [str(directory / "game.tw-pddl")]
    assert requests == [{"won": True, "admissible_commands": True, "extras": ["gamefile"]}]
    assert state.observation == "observable"
    state = env.step("invalid")
    assert state.done and not state.success
    env.close()
    assert runtime.closed
