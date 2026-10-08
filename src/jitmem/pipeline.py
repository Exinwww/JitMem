from __future__ import annotations

import json
import re
from dataclasses import asdict
from time import perf_counter
from typing import Any

from .api import ChatResult
from .memory import MemoryBank, Trajectory, Turn
from .prompts import (
    DISTILL_SYSTEM,
    curator_messages,
    executor_messages,
    judge_messages,
    memory_context,
)
from .usage import aggregate_usage


class MockChatClient:
    """Deterministic test double. Results from this client are never benchmark scores."""

    def complete(self, messages: list[dict[str, str]]) -> ChatResult:
        content = messages[-1]["content"]
        if "Admissible actions:" in content:
            import ast

            line = content.split("Admissible actions:", 1)[1].split("\n", 1)[0].strip()
            actions = ast.literal_eval(line)
            action = next((a for a in actions if a != "look"), "look")
            text = f"<action>{action}</action>"
        elif messages[0]["content"].startswith("Assess whether"):
            success = "you put the mug 1 in the cabinet 1" in content.lower()
            text = json.dumps(
                {
                    "success": success,
                    "rationale": "Fixture observation checked.",
                    "evidence_step": 5 if success else -1,
                }
            )
        else:
            text = "Find the target object, take it, then place it in the requested receptacle."
        return ChatResult(text=text, prompt_tokens=0, completion_tokens=0, latency_seconds=0.0)


def parse_action(response: str, admissible: list[str]) -> str:
    matches = re.findall(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL)
    if len(matches) != 1:
        raise ValueError("Expected exactly one <action>...</action> command.")
    command = matches[0].strip()
    if command not in admissible:
        raise ValueError(f"Command is not admissible: {command!r}.")
    return command


def parse_judgment(response: str) -> dict[str, Any]:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {
            "success": False,
            "rationale": "Malformed judge JSON; rejected.",
            "evidence_step": -1,
            "parse_error": True,
        }
    if not isinstance(value, dict) or type(value.get("success")) is not bool:
        return {
            "success": False,
            "rationale": "Judge success must be a JSON boolean; rejected.",
            "evidence_step": -1,
            "parse_error": True,
        }
    return value


def goal_from_observation(observation: str, fallback: str) -> str:
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation, re.IGNORECASE)
    return match.group(1).strip() if match else fallback


class Pipeline:
    def __init__(self, config, executor, curator):
        self.config = config
        self.executor = executor
        self.curator = curator

    def run_episode(self, task, environment, bank: MemoryBank) -> tuple[dict, Trajectory | None]:
        started = perf_counter()
        state = environment.reset(task)
        description = goal_from_observation(state.observation, task.description)
        initial_observation = state.observation
        exp = self.config.experiment
        memories = bank.retrieve(description, exp.retrieval_k) if exp.method != "no-memory" else []
        calls: list[dict] = []
        generation_failures: list[dict] = []

        def complete(role, client, messages):
            answer = client.complete(messages)
            calls.append(
                {
                    "role": role,
                    "messages": messages,
                    "response": answer.text,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "latency_seconds": answer.latency_seconds,
                    "finish_reason": answer.finish_reason,
                    "incomplete": answer.incomplete,
                }
            )
            if answer.incomplete:
                generation_failures.append(
                    {
                        "role": role,
                        "call_index": len(calls) - 1,
                        "finish_reason": answer.finish_reason,
                    }
                )
            return answer

        payload = ""
        if exp.method in {"jitmem", "write-summary"}:
            briefing = complete(
                "curator",
                self.curator,
                curator_messages(
                    description,
                    memories,
                    task_adaptive=exp.task_adaptive,
                    raw=exp.method != "write-summary",
                    label_outcomes=exp.store_policy == "all",
                ),
            )
            payload = "" if briefing.incomplete else briefing.text
        elif exp.method == "raw-memory":
            payload = memory_context(memories, label_outcomes=exp.store_policy == "all")

        turns: list[Turn] = []
        invalid_actions = 0
        while len(turns) < exp.max_steps and not state.done:
            observation = state.observation
            available = list(state.admissible_actions)
            answer = complete(
                "executor",
                self.executor,
                executor_messages(
                    description, payload, observation, available, turns, exp.history_window
                ),
            )
            response = answer.text
            try:
                if answer.incomplete:
                    raise ValueError(
                        f"Incomplete model output ({answer.finish_reason}); no command executed."
                    )
                action = parse_action(response, available)
            except ValueError as error:
                invalid_actions += 1
                # Invalid outputs consume the decision budget but cannot mutate the game.
                feedback = f"{observation}\nInvalid decision: {error}"
                turns.append(Turn(observation, "<invalid>", response, feedback, available, False))
                from dataclasses import replace

                state = replace(state, observation=feedback)
                continue
            state = environment.step(action)
            turns.append(Turn(observation, action, response, state.observation, available))

        trajectory = Trajectory(
            task.task_id, description, task.task_type, task.split, initial_observation, turns
        )
        judgment = None
        stored = None
        if exp.method != "no-memory":
            verdict = complete("judge", self.executor, judge_messages(trajectory))
            judgment = (
                {
                    "success": False,
                    "rationale": f"Incomplete judge output ({verdict.finish_reason}); rejected.",
                    "evidence_step": -1,
                    "generation_error": True,
                }
                if verdict.incomplete
                else parse_judgment(verdict.text)
            )
            trajectory.judge_success = judgment["success"]
            if exp.store_policy == "all" or judgment["success"]:
                if exp.method == "write-summary":
                    distilled = complete(
                        "distiller",
                        self.curator,
                        [
                            {"role": "system", "content": DISTILL_SYSTEM},
                            {
                                "role": "user",
                                "content": f"Task: {description}\n{trajectory.render()}",
                            },
                        ],
                    )
                    if not distilled.incomplete:
                        trajectory.summary = distilled.text
                        # This ablation intentionally discards raw stored information.
                        trajectory.turns = []
                        trajectory.initial_observation = ""
                        stored = trajectory
                else:
                    stored = trajectory

        result = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "split": task.split,
            "task_description": description,
            "success": bool(state.success),
            "reward": float(state.reward),
            "environment_done": bool(state.done),
            "steps": len(turns),
            "environment_steps": sum(turn.executed for turn in turns),
            "invalid_actions": invalid_actions,
            "generation_failures": generation_failures,
            "truncated": not state.success and len(turns) >= exp.max_steps,
            "memory_size_before": len(bank.entries),
            "retrieved": [{"task_id": item.task_id, "score": score} for item, score in memories],
            "payload": payload,
            "judge": judgment,
            "stored": stored is not None,
            "latency_seconds": perf_counter() - started,
            "usage": aggregate_usage(calls),
            "calls": calls,
            "trajectory": {
                "initial_observation": initial_observation,
                "turns": [asdict(turn) for turn in turns],
            },
        }
        return result, stored
