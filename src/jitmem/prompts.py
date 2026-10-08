"""Semantically faithful paraphrases of JITMEM / SkillOS ALFWorld prompts.

Prompt wording is an explicit reproduction choice, not an author-provided code release.
"""

from .memory import Trajectory

CURATOR_SYSTEM = """You curate episodic memory for an agent solving household tasks in ALFWorld.
Read the current task and the retrieved experience. Produce a short, actionable briefing:
identify relevant episodes, extract useful object-search strategies and action ordering,
and explain how these lessons apply to the current goal. Treat episode-specific locations
and object numbers as clues that require checking in the current environment.
When experience is absent or irrelevant, say so and give only general procedural guidance.
Keep the briefing concise. The output is guidance, not actions executed in the environment."""

JUDGE_SYSTEM = """Assess whether a household agent completed the supplied task using only
the recorded observations and actions. Check every required final-state condition,
including the correct objects, destination, quantity, and any requested cleaning, heating,
cooling or lighting. Any required transformation must occur before final placement.
ALFWorld's look/examine-an-object-with-or-under-a-lamp tasks are satisfied by holding
the requested object at the receptacle where the requested lamp has been activated.
Infer this from the recorded pickup, lamp-use and navigation observations. No separate
examine action or placing the object under/on the lamp is required; activating the lamp
before picking up the object or returning to it while holding the object is valid.
An agent's intention or claim is not evidence: effects must be supported by observations.
Partial completion, an unresolved loop, repeated invalid outputs, an exhausted budget
without completion, or ambiguous evidence is failure.
Return only JSON with success (boolean), rationale (short explanation), and evidence_step
(integer observation index, or -1 when no conclusive evidence exists)."""

DISTILL_SYSTEM = """Extract at most three reusable household-planning lessons from this
episode. Describe search strategies, action prerequisites, useful operation order,
or failure avoidance. Generalize beyond this episode: omit specific object names, numbers
and locations. Return a JSON array of nonoverlapping items. Each item must have a short
title, a one-sentence description, and content containing one to three actionable sentences."""


def memory_context(
    memories: list[tuple[Trajectory, float]], *, raw: bool = True, label_outcomes: bool = False
) -> str:
    parts = []
    for index, (memory, _) in enumerate(memories, 1):
        outcome = ""
        if label_outcomes:
            outcome = f"\nExecutor judge label: {'success' if memory.judge_success else 'failure'}"
        parts.append(
            f"Memory {index}:\nQuestion: {memory.task_description}{outcome}\n"
            f"Trajectory:\n{memory.render(raw=raw)}"
        )
    return "\n\n".join(parts) or "No past episodes are available."


def curator_messages(
    description: str,
    memories: list[tuple[Trajectory, float]],
    *,
    task_adaptive: bool = True,
    raw: bool = True,
    label_outcomes: bool = False,
) -> list[dict[str, str]]:
    question = f"Question: {description}\n\n" if task_adaptive else ""
    system = CURATOR_SYSTEM
    if not task_adaptive:
        system += "\nNo current question is supplied in this ablation; give reusable guidance."
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": question
            + "Retrieved memories:\n"
            + memory_context(memories, raw=raw, label_outcomes=label_outcomes),
        },
    ]


def executor_messages(
    description: str,
    payload: str,
    observation: str,
    admissible: list[str],
    turns: list,
    history_window: int,
) -> list[dict[str, str]]:
    recent = turns[-history_window:] if history_window else []
    history = (
        "\n".join(f"Observation: {turn.observation}\nAction: {turn.action}" for turn in recent)
        or "No previous actions."
    )
    prompt = (
        f"Solve this task in the ALFWorld household environment:\n{description}\n\n"
        f"Past-experience guidance:\n{payload or 'No memory guidance.'}\n\n"
        f"You have made {len(turns)} previous decisions. Recent {len(recent)} turns:\n"
        f"{history}\n\nCurrent step: {len(turns) + 1}\n"
        f"Current observation: {observation}\n"
        f"Admissible actions: {admissible}\n\n"
        "Reason about what advances the goal, then choose exactly one admissible action. "
        "Write the chosen command inside <action>...</action> tags."
    )
    return [{"role": "user", "content": prompt}]


def judge_messages(trajectory: Trajectory) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Task: {trajectory.task_description}\n\n"
            f"Recorded episode:\n{trajectory.render()}",
        },
    ]
