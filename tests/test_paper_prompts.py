"""Paper-profile protocol checks using fictional, non-paper template fixtures."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import jitmem.pipeline as pipeline_module
from jitmem.api import ChatResult
from jitmem.config import ConfigError, EnvironmentConfig, ExperimentConfig, RunConfig, load_config
from jitmem.environments import EnvState, TaskSpec
from jitmem.memory import MemoryBank, Trajectory, Turn
from jitmem.pipeline import Pipeline, parse_action
from jitmem.prompts import curator_messages, executor_messages, judge_messages


@pytest.fixture
def assets():
    # The synthetic strings deliberately contain no published prompt wording.
    return SimpleNamespace(
        curator_system="FICTIONAL CURATOR SYSTEM",
        curator_user="CQUERY:{query}\n\nCMEMORIES:\nMemory 1: fixture remainder",
        executor=(
            "ETASK:{task_description}\nEPAYLOAD:<{retrieved_context}>\n"
            "ESTEPS:{step_count};EHISTORYLEN:{history_length};ENEXT:{current_step}\n"
            "EHISTORY:<{action_history}>\nEOBS:<{current_observation}>\n"
            "EACTIONS:[{admissible_actions}]"
        ),
        judge_system='FICTIONAL JUDGE SYSTEM {{"sample": true}}',
        judge_user="JTASK:{task_description}\nJTRACE:\n{trajectory}",
        distillation="FICTIONAL DISTILLER SYSTEM",
        provenance={"profile": "paper-v1", "fixture": True},
    )


def episode_task():
    return TaskSpec("fictional", "pick_and_place_simple", "", "fallback goal", "valid_seen")


def answer(text, *, finish_reason=None):
    return ChatResult(text, 1, 1, 0.0, finish_reason=finish_reason)


class QueuedClient:
    def __init__(self, answers):
        self.answers = deque(answers)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        assert self.answers, "Unexpected model call or role-routing error"
        return self.answers.popleft()


class RecordingEnvironment:
    """Unconditional step recording exposes caller-side action suppression."""

    def __init__(self, states):
        self.states = deque(states)
        self.actions = []

    def reset(self, task):
        return EnvState("Your task is to: fictional goal", ("look", "help"))

    def step(self, action):
        self.actions.append(action)
        assert self.states, "Unexpected environment step"
        return self.states.popleft()

    def close(self):
        pass


def paper_pipeline(monkeypatch, assets, executor_answers, curator_answers, **experiment):
    monkeypatch.setattr(pipeline_module, "load_assets", lambda folder: assets)
    config = RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(prompt_profile="paper-v1", **experiment),
    )
    executor = QueuedClient(executor_answers)
    curator = QueuedClient(curator_answers)
    return Pipeline(config, executor, curator), executor, curator


def test_paper_cold_start_preserves_empty_context_without_extra_guidance(assets):
    messages = curator_messages("fictional goal", [], profile="paper-v1", assets=assets)
    assert messages == [
        {"role": "system", "content": assets.curator_system},
        {"role": "user", "content": "CQUERY:fictional goal\n\nCMEMORIES:\n"},
    ]
    prompt = executor_messages(
        "fictional goal",
        "",
        "visible room",
        ["look", "help"],
        [],
        3,
        profile="paper-v1",
        assets=assets,
    )[0]["content"]
    assert "EPAYLOAD:<>" in prompt and "EHISTORY:<>" in prompt
    assert "ESTEPS:0;EHISTORYLEN:0;ENEXT:1" in prompt
    assert "EACTIONS:['look']" in prompt
    assert "No memory" not in prompt and "No previous" not in prompt


def test_paper_curator_expands_ranked_raw_memories_and_only_full_storage_adds_labels(assets):
    first = Trajectory(
        "past-a",
        "earlier goal a",
        "pick",
        "valid_seen",
        "initial a",
        [Turn("initial a", "look", "hidden model reasoning", "final a")],
        False,
    )
    second = Trajectory("past-b", "earlier goal b", "pick", "valid_seen", "initial b", [], True)
    memories = [(first, 2.0), (second, 1.0)]
    filtered = curator_messages("new goal", memories, profile="paper-v1", assets=assets)
    full = curator_messages(
        "new goal",
        memories,
        label_outcomes=True,
        profile="paper-v1",
        assets=assets,
    )
    text = filtered[1]["content"]
    assert text.startswith("CQUERY:new goal\n\nCMEMORIES:\nMemory 1:")
    assert text.index("earlier goal a") < text.index("Memory 2:") < text.index("earlier goal b")
    assert first.render() in text and second.render() in text
    assert "hidden model reasoning" not in text and "Executor judge label" not in text
    assert filtered[0] == full[0] == {"role": "system", "content": assets.curator_system}
    assert "Executor judge label: failure" in full[1]["content"]
    assert "Executor judge label: success" in full[1]["content"]
    for label in ("\nExecutor judge label: failure", "\nExecutor judge label: success"):
        full[1]["content"] = full[1]["content"].replace(label, "")
    assert full == filtered


def test_task_adaptive_ablation_removes_current_query_without_inventing_instructions(assets):
    messages = curator_messages(
        "query must be hidden",
        [],
        task_adaptive=False,
        profile="paper-v1",
        assets=assets,
    )
    assert messages == [
        {"role": "system", "content": assets.curator_system},
        {"role": "user", "content": "CMEMORIES:\n"},
    ]


def test_executor_uses_numbered_recent_observations_and_parsed_actions(assets):
    turns = [
        Turn(f"visible-{i}", f"command-{i}", f"private-reasoning-{i}", f"next-{i}")
        for i in range(1, 5)
    ]
    prompt = executor_messages(
        "goal",
        "payload",
        "now",
        ["look", "help", "open box 1"],
        turns,
        3,
        profile="paper-v1",
        assets=assets,
    )[0]["content"]
    assert "ESTEPS:4;EHISTORYLEN:3;ENEXT:5" in prompt
    assert "visible-1" not in prompt and "command-1" not in prompt
    assert "private-reasoning" not in prompt
    expected_history = "\n".join(
        f"[Observation {i}: 'visible-{i}', Action {i}: 'command-{i}']" for i in (2, 3, 4)
    )
    assert f"EHISTORY:<{expected_history}>" in prompt
    assert "EACTIONS:['look'\n 'open box 1']" in prompt and "'help'" not in prompt
    assert "EPAYLOAD:<payload>" in prompt and "EOBS:<now>" in prompt
    without_history = executor_messages(
        "goal",
        "payload",
        "now",
        ["look"],
        turns,
        0,
        profile="paper-v1",
        assets=assets,
    )[0]["content"]
    assert "ESTEPS:4;EHISTORYLEN:0;ENEXT:5" in without_history
    assert "EHISTORY:<>" in without_history


def test_paper_judge_uses_original_asset_only_and_observation_action_trace(assets):
    trajectory = Trajectory(
        "past",
        "goal",
        "pick",
        "valid_seen",
        "initial",
        [Turn("initial", "look", "private model reasoning", "confirmed simulator effect")],
        True,
    )
    messages = judge_messages(trajectory, profile="paper-v1", assets=assets)
    assert messages == [
        {"role": "system", "content": 'FICTIONAL JUDGE SYSTEM {"sample": true}'},
        {"role": "user", "content": "JTASK:goal\nJTRACE:\n" + trajectory.render()},
    ]
    assert "private model reasoning" not in str(messages)
    assert "judge_success" not in str(messages)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<ACTION> LOOK </ACTION>", "look"),
        ("<action>INVENTED COMMAND</action><action>look</action>", "invented command"),
        ("An invalid long preamble " + "A" * 30, "a" * 30),
        ("", ""),
    ],
)
def test_paper_action_projection_matches_inherited_first_block_or_tail(response, expected):
    assert parse_action(response, ["look"], profile="paper-v1") == expected


def test_invalid_actions_reach_environment_without_added_feedback_or_retries(monkeypatch, assets):
    pipeline, executor, curator = paper_pipeline(
        monkeypatch,
        assets,
        [answer("<action>INVENTED</action>"), answer("bad syntax"), answer('{"success":false}')],
        [answer("fixture payload")],
        max_steps=2,
    )
    environment = RecordingEnvironment(
        [
            EnvState("native feedback 1", ("look",)),
            EnvState("native feedback 2", ("look",)),
        ]
    )
    result, stored = pipeline.run_episode(episode_task(), environment, MemoryBank())
    assert environment.actions == ["invented", "bad syntax"]
    assert result["steps"] == result["environment_steps"] == result["invalid_actions"] == 2
    assert result["truncated"] and stored is None
    assert all(turn["executed"] for turn in result["trajectory"]["turns"])
    second_prompt = executor.messages[1][0]["content"]
    assert "EOBS:<native feedback 1>" in second_prompt
    assert "Action 1: 'invented'" in second_prompt
    assert "Invalid decision" not in str(result["calls"])
    assert [call["role"] for call in result["calls"]] == [
        "curator",
        "executor",
        "executor",
        "judge",
    ]
    assert len(curator.messages) == 1 and not curator.answers and not executor.answers


def test_length_limited_text_is_used_by_curator_executor_and_closed_json_judge(monkeypatch, assets):
    pipeline, executor, curator = paper_pipeline(
        monkeypatch,
        assets,
        [
            answer("<ACTION> LOOK </ACTION>", finish_reason="length"),
            answer('{"success":true,"evidence_step":1}', finish_reason="length"),
        ],
        [answer("use this returned partial payload", finish_reason="length")],
        max_steps=1,
    )
    environment = RecordingEnvironment([EnvState("simulator confirmed", ("look",), done=True)])
    result, stored = pipeline.run_episode(episode_task(), environment, MemoryBank())
    assert environment.actions == ["look"]
    assert result["payload"] == "use this returned partial payload"
    assert "EPAYLOAD:<use this returned partial payload>" in executor.messages[0][0]["content"]
    assert result["judge"]["success"] is True and stored.judge_success is True
    assert result["success"] is False and result["reward"] == 0.0
    assert result["steps"] == result["environment_steps"] == 1
    assert result["invalid_actions"] == 0
    assert [item["role"] for item in result["generation_failures"]] == [
        "curator",
        "executor",
        "judge",
    ]
    assert not curator.answers and not executor.answers


@pytest.mark.parametrize("store_policy", ["judge", "all"])
def test_malformed_judge_json_does_not_change_native_score(monkeypatch, assets, store_policy):
    pipeline, executor, curator = paper_pipeline(
        monkeypatch,
        assets,
        [answer("<action>look</action>"), answer('{"success":tru', finish_reason="length")],
        [answer("fixture payload")],
        max_steps=1,
        store_policy=store_policy,
    )
    environment = RecordingEnvironment(
        [
            EnvState("native success confirmed", ("look",), done=True, success=True, reward=1.0),
        ]
    )
    result, stored = pipeline.run_episode(episode_task(), environment, MemoryBank())
    assert result["success"] is True and result["reward"] == 1.0
    assert result["judge"]["success"] is False and result["judge"]["parse_error"] is True
    if store_policy == "all":
        assert stored is not None and stored.judge_success is False
        assert stored.turns[0].next_observation == "native success confirmed"
    else:
        assert stored is None
    assert not curator.answers and not executor.answers


def test_paper_no_memory_uses_same_executor_and_does_not_call_curator_or_judge(monkeypatch, assets):
    pipeline, executor, curator = paper_pipeline(
        monkeypatch,
        assets,
        [answer("<action>look</action>")],
        [],
        method="no-memory",
        max_steps=1,
    )
    environment = RecordingEnvironment(
        [
            EnvState("native success", ("look",), done=True, success=True, reward=1.0),
        ]
    )
    result, stored = pipeline.run_episode(episode_task(), environment, MemoryBank())
    assert result["success"] and result["judge"] is None and stored is None
    assert result["payload"] == "" and "EPAYLOAD:<>" in executor.messages[0][0]["content"]
    assert not curator.messages and not executor.answers
    assert set(result["usage"]["by_role"]) == {"executor"}


def test_toml_defaults_to_paper_but_legacy_is_explicitly_selectable(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("")
    config = load_config(path)
    assert config.experiment.prompt_profile == "paper-v1"
    path.write_text('[experiment]\nprompt_profile = "legacy-paraphrase"\n')
    assert load_config(path).experiment.prompt_profile == "legacy-paraphrase"
    assert RunConfig().experiment.prompt_profile == "legacy-paraphrase"


@pytest.mark.parametrize("profile", ["paper", "", None, ["paper-v1"]])
def test_invalid_prompt_profile_is_a_configuration_error(profile):
    with pytest.raises(ConfigError, match="prompt_profile"):
        ExperimentConfig(prompt_profile=profile)


def test_missing_paper_assets_fail_before_any_model_call(tmp_path):
    config = RunConfig(
        experiment=ExperimentConfig(
            prompt_profile="paper-v1",
            prompt_assets=str(tmp_path / "missing"),
        )
    )
    client = QueuedClient([])
    with pytest.raises(
        (FileNotFoundError, ValueError), match="(?i)(asset|prepare|manifest|prompt)"
    ):
        Pipeline(config, client, client)
    assert client.messages == []


def test_real_local_assets_validate_and_render_without_public_template_copies():
    from jitmem.paper_assets import load_assets

    folder = Path(__file__).resolve().parents[1] / "outputs/paper_prompts/v1"
    if not (folder / "manifest.json").exists():
        pytest.skip("Original paper assets are prepared locally, outside version control")
    real = load_assets(folder)
    messages = executor_messages(
        "fictional goal",
        "fictional payload",
        "fictional observation",
        ["look"],
        [],
        3,
        profile="paper-v1",
        assets=real,
    )
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert all(
        text in messages[0]["content"]
        for text in (
            "fictional goal",
            "fictional payload",
            "fictional observation",
        )
    )
    assert real.provenance
