"""Lossless episodic storage and description-only BM25 retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Turn:
    observation: str
    action: str
    response: str
    next_observation: str
    admissible_actions: list[str] = field(default_factory=list)
    executed: bool = True


@dataclass
class Trajectory:
    task_id: str
    task_description: str
    task_type: str
    split: str
    initial_observation: str
    turns: list[Turn]
    judge_success: bool | None = None
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # Native rewards and curator payload deliberately live only in evaluation artifacts.
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Trajectory:
        value = dict(value)
        value["turns"] = [Turn(**turn) for turn in value["turns"]]
        return cls(**value)

    def render(self, *, raw: bool = True) -> str:
        if not raw and self.summary is not None:
            return self.summary
        lines = [f"OBSERVATION 0: {self.initial_observation}"]
        for i, turn in enumerate(self.turns, 1):
            # Keep the complete trace, including invalid decisions and their feedback.
            lines.extend(
                [
                    f"ACTION {i}: {turn.action}",
                    f"OBSERVATION {i}: {turn.next_observation}",
                ]
            )
        return "\n".join(lines)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class MemoryBank:
    def __init__(self, entries: list[Trajectory] | None = None):
        self.entries = list(entries or [])

    def retrieve(self, description: str, k: int) -> list[tuple[Trajectory, float]]:
        if not self.entries or k <= 0:
            return []
        # BM25 with positive Robertson IDF. Stable insertion order breaks score ties.
        corpus = [Counter(tokenize(item.task_description)) for item in self.entries]
        lengths = [sum(doc.values()) for doc in corpus]
        average_length = sum(lengths) / len(lengths) or 1.0
        query = tokenize(description)
        frequencies = {term: sum(term in doc for doc in corpus) for term in set(query)}
        scores = []
        for index, doc in enumerate(corpus):
            score = 0.0
            for term in query:
                tf = doc[term]
                df = frequencies[term]
                idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * lengths[index] / average_length)
                score += idf * tf * 2.5 / denominator
            scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [(self.entries[index], score) for index, score in scores[:k]]

    def append_batch(self, trajectories: list[Trajectory]) -> None:
        existing = {item.task_id for item in self.entries}
        for trajectory in trajectories:
            if trajectory.task_id not in existing:
                self.entries.append(trajectory)
                existing.add(trajectory.task_id)
