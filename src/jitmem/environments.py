"""Deterministic ALFWorld task selection and isolated interactive environments.

Only observable feedback and optionally admissible commands reach the agent.
Expert plans, PDDL facts, and ground-truth object locations are never returned.
The household fixture below verifies software wiring; it is not an ALFWorld benchmark.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
SPLITS = ("train", "valid_seen", "valid_unseen", "valid_train")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    game_file: str
    description: str
    split: str


@dataclass(frozen=True)
class EnvState:
    observation: str
    admissible_actions: tuple[str, ...]
    done: bool = False
    success: bool = False
    reward: float = 0.0


@runtime_checkable
class Environment(Protocol):
    def reset(self, task: TaskSpec) -> EnvState: ...

    def step(self, action: str) -> EnvState: ...

    def close(self) -> None: ...


def resolve_data_root(data_root: str | Path) -> Path:
    """Accept the existing data workspace, ALFWorld root, or JSON dataset root."""
    root = Path(data_root).expanduser().resolve()
    for candidate in (root, root / "json_2.1.1", root / "data/alfworld/json_2.1.1"):
        if any((candidate / split).is_dir() for split in SPLITS):
            return candidate
    raise FileNotFoundError(f"No ALFWorld split directories found beneath {root}")


def extract_goal(observation: str) -> str:
    """The environment's observed instruction is authoritative during evaluation."""
    match = re.search(r"Your task is to:\s*([^\n]+)", observation, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _grammar_goal(grammar: str) -> str:
    # ALFWorld stores a TWL2 string with a JSON-like task production. Parse only
    # this production: no walkthrough, internal facts, or planner inputs escape.
    match = re.search(r'"task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*("(?:\\.|[^"\\])*")', grammar)
    if match:
        return extract_goal(json.loads(match.group(1)))
    return ""


def discover_tasks(
    data_root: str | Path,
    split: str,
    task_types: Iterable[str | int] | None = None,
    annotation_index: int = 0,
    limit: int | None = None,
) -> list[TaskSpec]:
    """Select the official text-eligible trials, without requiring a runtime.

    Every trial remains a distinct episode, including trials sharing a task
    configuration directory. Human annotations are used only when a game's
    own goal production cannot be read. Initial state PDDL is embedded in the
    game and need not be opened. Selection matches AlfredTWEnv's six-type,
    non-movable, non-sliced, present-and-solvable-game filters.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose one of {SPLITS}")
    if annotation_index < 0:
        raise ValueError("annotation_index must be nonnegative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None")
    allowed = set(TASK_TYPES.values())
    if task_types is not None:
        selected: set[str] = set()
        for value in task_types:
            name = TASK_TYPES.get(value) if isinstance(value, int) else str(value)
            if name not in allowed:
                raise ValueError(f"Unsupported ALFWorld task type: {value!r}")
            selected.add(name)
        allowed = selected

    split_root = resolve_data_root(data_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ALFWorld split not found: {split_root}")
    tasks = []
    for trajectory_file in sorted(split_root.rglob("traj_data.json")):
        directory = trajectory_file.parent
        if "movable" in str(directory) or "Sliced" in str(directory):
            continue
        trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
        if trajectory.get("task_type") not in allowed:
            continue
        game_file = directory / "game.tw-pddl"
        if not game_file.is_file():
            continue
        game = json.loads(game_file.read_text(encoding="utf-8"))
        if not game.get("solvable", False):
            continue
        description = _grammar_goal(game.get("grammar", ""))
        if not description:
            annotations = trajectory.get("turk_annotations", {}).get("anns", [])
            if annotations:
                if annotation_index >= len(annotations):
                    raise ValueError(
                        f"Annotation {annotation_index} unavailable in {trajectory_file}"
                    )
                description = annotations[annotation_index].get("task_desc", "")
        tasks.append(
            TaskSpec(
                task_id=trajectory.get("task_id", directory.name),
                task_type=trajectory["task_type"],
                game_file=str(game_file),
                description=description,
                split=split,
            )
        )
    return tasks[:limit] if limit is not None else tasks


class ALFWorldEnvironment:
    """Single-game TextWorld adapter with the official ALFWorld name wrapper.

    Opening each selected game explicitly avoids Gym's shuffled episode cycle,
    and permits deterministic manifests and exact task coverage. No expert
    wrapper is used, including when collecting training experiences.
    """

    benchmark = True
    backend = "alfworld"

    def __init__(self, max_steps: int = 30):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._env: Any = None
        self._steps = 0
        self._state: EnvState | None = None

    def reset(self, task: TaskSpec) -> EnvState:
        self.close()
        game_file = Path(task.game_file).expanduser().resolve()
        if not game_file.is_file():
            raise FileNotFoundError(game_file)
        # alfworld.info creates its data root on import. Set this first so an
        # import cannot attempt to create ~/.cache/alfworld. Existing data is read.
        dataset_root = next(
            (parent for parent in game_file.parents if parent.name == "json_2.1.1"), None
        )
        if dataset_root is not None:
            os.environ["ALFWORLD_DATA"] = str(dataset_root.parent)
        try:
            import textworld
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        except ImportError as error:
            raise RuntimeError(
                "ALFWorld runtime is unavailable. Run scripts/setup_alfworld.sh "
                "and use .venv/bin/python."
            ) from error
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        try:
            self._env = textworld.start(
                str(game_file), infos, wrappers=[lambda env: AlfredDemangler(env, shuffle=False)]
            )
            raw_state = self._env.reset()
        except Exception:
            self.close()
            raise
        self._steps = 0
        self._state = self._convert(raw_state)
        return self._state

    def _convert(self, raw_state: Any, reward: float = 0.0, done: bool = False) -> EnvState:
        success = bool(raw_state.get("won", False))
        return EnvState(
            observation=str(raw_state.get("feedback", "")),
            admissible_actions=tuple(raw_state.get("admissible_commands", ()) or ()),
            done=bool(done or success or self._steps >= self.max_steps),
            success=success,
            reward=float(reward),
        )

    def step(self, action: str) -> EnvState:
        if self._env is None or self._state is None:
            raise RuntimeError("Call reset(task) before step(action)")
        if self._state.done:
            raise RuntimeError("Episode already ended; call reset(task)")
        # Count every submitted command, including invalid commands and look.
        raw_state, score, done = self._env.step(action)
        self._steps += 1
        self._state = self._convert(raw_state, score, done)
        return self._state

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._state = None


def fake_tasks(count: int = 3) -> list[TaskSpec]:
    """Small deterministic fixtures for offline tests; never benchmark data."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    return [
        TaskSpec(
            task_id=f"fake-household-{index + 1}",
            task_type="pick_and_place_simple",
            game_file="",
            description="put a mug in cabinet.",
            split="fake",
        )
        for index in range(count)
    ]


class FakeHouseholdEnvironment:
    """Minimal observable household state machine, exclusively for wiring tests."""

    benchmark = False
    backend = "mock"

    def __init__(self, max_steps: int = 30):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._state: EnvState | None = None
        self._steps = 0
        self._location = "room"
        self._held = False
        self._open = False
        self._won = False

    def _actions(self) -> tuple[str, ...]:
        # Productive commands precede observation-only commands for the
        # deterministic mock policy used in offline pipeline tests.
        productive = []
        if self._location == "room":
            productive.append("go to desk 1")
        elif self._location == "desk 1" and not self._held:
            productive.append("take mug 1 from desk 1")
        elif self._location == "desk 1":
            productive.append("go to cabinet 1")
        elif self._location == "cabinet 1" and not self._open:
            productive.append("open cabinet 1")
        elif self._held:
            productive.append("move mug 1 to cabinet 1")
        return tuple(productive + ["look", "inventory"])

    def _make_state(self, observation: str) -> EnvState:
        self._state = EnvState(
            observation=observation,
            admissible_actions=self._actions(),
            done=self._won or self._steps >= self.max_steps,
            success=self._won,
            reward=float(self._won),
        )
        return self._state

    def reset(self, task: TaskSpec) -> EnvState:
        if task.split != "fake":
            raise ValueError("Fake environment accepts only split='fake' wiring fixtures")
        self._steps = 0
        self._location = "room"
        self._held = self._open = self._won = False
        return self._make_state(
            "[OFFLINE WIRING FIXTURE]\nYou are in a room with a desk 1 and a cabinet 1.\n"
            "Your task is to: put a mug in cabinet."
        )

    def step(self, action: str) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset(task) before step(action)")
        if self._state.done:
            raise RuntimeError("Episode already ended; call reset(task)")
        self._steps += 1
        observation = "Nothing happens."
        if action == "look":
            observation = f"You are at {self._location}."
            if self._location == "desk 1" and not self._held:
                observation += " On the desk 1, you see a mug 1."
        elif action == "inventory":
            observation = "You are carrying a mug 1." if self._held else "You are empty-handed."
        elif action == "go to desk 1":
            self._location = "desk 1"
            observation = "You arrive at desk 1. On the desk 1, you see a mug 1."
        elif action == "take mug 1 from desk 1" and self._location == "desk 1" and not self._held:
            self._held = True
            observation = "You pick up the mug 1 from the desk 1."
        elif action == "go to cabinet 1":
            self._location = "cabinet 1"
            observation = "You arrive at cabinet 1. The cabinet 1 is closed."
        elif action == "open cabinet 1" and self._location == "cabinet 1":
            self._open = True
            observation = "You open the cabinet 1. The cabinet 1 is empty."
        elif (
            action == "move mug 1 to cabinet 1"
            and self._location == "cabinet 1"
            and self._open
            and self._held
        ):
            self._held = False
            self._won = True
            observation = "You put the mug 1 in the cabinet 1. You won!"
        return self._make_state(observation)

    def close(self) -> None:
        self._state = None
