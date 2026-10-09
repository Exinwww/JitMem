"""Original-paper templates from verified local assets, plus historical prompts.

Only the legacy profile uses the paraphrases below. Paper assets are obtained
by scripts/prepare_paper_prompts.py, never fetched during model evaluation.
"""

from __future__ import annotations

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
    memories: list[tuple[Trajectory, float]],
    *,
    raw: bool = True,
    label_outcomes: bool = False,
    empty: str = "No past episodes are available.",
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
    return "\n\n".join(parts) or empty


def _paper_assets(profile, assets):
    if profile == "legacy-paraphrase":
        return None
    if profile != "paper-v1":
        raise ValueError("Unsupported prompt profile")
    if assets is None:
        from .paper_assets import load_assets

        assets = load_assets("outputs/paper_prompts/v1")
    return assets


def curator_messages(
    description: str,
    memories: list[tuple[Trajectory, float]],
    *,
    task_adaptive: bool = True,
    raw: bool = True,
    label_outcomes: bool = False,
    profile: str = "legacy-paraphrase",
    assets=None,
) -> list[dict[str, str]]:
    paper = _paper_assets(profile, assets)
    if paper is not None:
        # The listing's ellipsis denotes repetition of its numbered Memory block.
        # It supplies no special empty-bank instructions: retain an empty context.
        header = paper.curator_user.partition("Memory 1:")[0]
        if not task_adaptive:
            header = header.partition("\n\n")[2]
        return [
            {"role": "system", "content": paper.curator_system},
            {
                "role": "user",
                "content": header.format(query=description)
                + memory_context(memories, raw=raw, label_outcomes=label_outcomes, empty=""),
            },
        ]
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
    *,
    profile: str = "legacy-paraphrase",
    assets=None,
) -> list[dict[str, str]]:
    recent = turns[-history_window:] if history_window else []
    paper = _paper_assets(profile, assets)
    if paper is not None:
        # GiGPO SimpleMemory/build_text_obs, pinned in docs/paper_fidelity.md.
        # JITMEM provides the outer template; these field encodings are inherited.
        first = len(turns) - len(recent) + 1
        history = "\n".join(
            f"[Observation {index}: '{turn.observation}', Action {index}: '{turn.action}']"
            for index, turn in enumerate(recent, first)
        )
        return [
            {
                "role": "user",
                "content": paper.executor.format(
                    task_description=description,
                    retrieved_context=payload,
                    step_count=len(turns),
                    history_length=len(recent),
                    action_history=history,
                    current_step=len(turns) + 1,
                    current_observation=observation,
                    admissible_actions="\n ".join(
                        f"'{action}'" for action in admissible if action != "help"
                    ),
                ),
            }
        ]
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


def judge_messages(
    trajectory: Trajectory, *, profile: str = "legacy-paraphrase", assets=None
) -> list[dict[str, str]]:
    paper = _paper_assets(profile, assets)
    if paper is not None:
        return [
            {"role": "system", "content": paper.judge_system.format()},
            {
                "role": "user",
                "content": paper.judge_user.format(
                    task_description=trajectory.task_description, trajectory=trajectory.render()
                ),
            },
        ]
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Task: {trajectory.task_description}\n\n"
            f"Recorded episode:\n{trajectory.render()}",
        },
    ]
