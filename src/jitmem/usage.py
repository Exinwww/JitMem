"""Token counts remain unknown when a provider omits any required usage value."""

from collections.abc import Iterable


def token_total(values: Iterable[int | None]) -> int | None:
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


def aggregate_usage(calls: list[dict]) -> dict:
    prompt_tokens = token_total(call["prompt_tokens"] for call in calls)
    completion_tokens = token_total(call["completion_tokens"] for call in calls)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "complete": prompt_tokens is not None and completion_tokens is not None,
        "by_role": {
            role: {
                "calls": sum(call["role"] == role for call in calls),
                "prompt_tokens": token_total(
                    call["prompt_tokens"] for call in calls if call["role"] == role
                ),
                "completion_tokens": token_total(
                    call["completion_tokens"] for call in calls if call["role"] == role
                ),
            }
            for role in sorted({call["role"] for call in calls})
        },
    }
