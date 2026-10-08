"""Small, strict client for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import ModelConfig


class _RejectRedirects(HTTPRedirectHandler):
    """Fail before inspecting Location or constructing a redirected request."""

    def http_error_302(self, request, response, code, _message, headers):
        raise HTTPError(request.full_url, code, "HTTP redirects are disabled", headers, response)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


_CHAT_OPENER = build_opener(_RejectRedirects())


def urlopen(request: Request, timeout: float):
    """Patchable transport entry point that never follows HTTP redirects."""
    return _CHAT_OPENER.open(request, timeout=timeout)


class ChatAPIError(RuntimeError):
    """An API request failed without exposing response bodies or credentials."""


class AuthenticationError(ChatAPIError):
    """Credentials are absent or the provider rejected authentication."""


class InvalidResponseError(ChatAPIError):
    """The endpoint did not return a well-formed assistant text response."""


class IncompleteResponseError(InvalidResponseError):
    """Legacy export; normal generation limits are represented by ChatResult."""


_INCOMPLETE_REASONS = frozenset({"length", "max_tokens", "max_completion_tokens", "content_filter"})


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float
    finish_reason: str | None = None

    @property
    def incomplete(self) -> bool:
        return self.finish_reason in _INCOMPLETE_REASONS


def _token_count(usage: dict[str, Any], field: str) -> int | None:
    value = usage.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidResponseError(f"Response usage.{field} must be a nonnegative integer")
    return value


def _response_text(content: Any, *, allow_empty: bool = False) -> str:
    if content is None and allow_empty:
        result = ""
    elif isinstance(content, str):
        result = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise InvalidResponseError("Response content parts must be objects with a type")
            if part["type"] == "text":
                if not isinstance(part.get("text"), str):
                    raise InvalidResponseError("Response text parts must contain a text string")
                parts.append(part["text"])
        result = "".join(parts)
    else:
        raise InvalidResponseError(
            "Response message.content must be text or a list of content parts"
        )
    if not result.strip() and not allow_empty:
        raise InvalidResponseError("Provider returned no assistant text")
    return result


def _parse_response(body: bytes, elapsed: float) -> ChatResult:
    try:
        response = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidResponseError("Provider response is not valid UTF-8 JSON") from exc
    if not isinstance(response, dict):
        raise InvalidResponseError("Provider response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise InvalidResponseError("Provider returned no valid response choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason is not None and not isinstance(reason, str):
        raise InvalidResponseError("Response finish_reason must be a string or null")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise InvalidResponseError("Response choice must contain an assistant message")
    if "content" not in message:
        raise InvalidResponseError("Response message must contain a content field")
    result = _response_text(message["content"], allow_empty=reason in _INCOMPLETE_REASONS)
    usage = response.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise InvalidResponseError("Response usage must be an object")
    return ChatResult(
        text=result,
        prompt_tokens=_token_count(usage, "prompt_tokens"),
        completion_tokens=_token_count(usage, "completion_tokens"),
        latency_seconds=elapsed,
        finish_reason=reason,
    )


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Use capped provider backoff; malformed headers fall back to exponential delay."""
    fallback = min(0.5 * 2**attempt, 8.0)
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                seconds = (retry_time - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return fallback
        if seconds == seconds:
            return max(0.0, min(seconds, 30.0))
    return fallback


class ChatClient:
    """Call a chat endpoint with bounded retries and environment-only credentials.

    ``retries`` counts extra attempts, so the total is at most ``retries + 1``.
    Authentication, other 4xx errors, malformed responses, and truncation are never
    retried. Only transport failures, HTTP 429, and HTTP 5xx are retried.
    """

    def __init__(self, config: ModelConfig):
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if not config.model.strip():
            raise ChatAPIError(
                "Set OPENAI_MODEL or JITMEM_EXECUTOR_MODEL/JITMEM_CURATOR_MODEL before making API calls"
            )
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if config.api_key_env and not self._api_key.strip():
            raise AuthenticationError(
                f"Set the {config.api_key_env} environment variable before making API calls"
            )
        if self._api_key and any(
            ord(character) < 33 or ord(character) > 126 for character in self._api_key
        ):
            # http.client includes invalid header values in its exception text.
            # Reject those credentials before any Request/header construction.
            raise AuthenticationError(
                f"The {config.api_key_env} environment variable must contain an ASCII API key without whitespace or control characters"
            )
        base = config.base_url.rstrip("/")
        self._url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def complete(self, messages: list[dict[str, str]]) -> ChatResult:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        validated_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Each message must contain exactly role and content")
            if not isinstance(message["role"], str) or message["role"] not in {
                "system",
                "developer",
                "user",
                "assistant",
            }:
                raise ValueError("Message role must be system, developer, user, or assistant")
            if not isinstance(message["content"], str):
                raise ValueError("Message content must be a string")
            validated_messages.append(dict(message))
        payload = copy.deepcopy(self.config.extra_body)
        payload.update(
            {
                "model": self.config.model,
                "messages": validated_messages,
                self.config.token_limit_parameter: self.config.max_tokens,
                "stream": False,
            }
        )
        if not self.config.omit_temperature and self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        started = time.perf_counter()
        for attempt in range(self.config.retries + 1):
            request = Request(self._url, data=encoded, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read()
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status in {401, 403}:
                    raise AuthenticationError(
                        f"Provider rejected authentication (HTTP {status})"
                    ) from None
                if (status == 429 or 500 <= status <= 599) and attempt < self.config.retries:
                    time.sleep(_retry_delay(attempt, retry_after))
                    continue
                raise ChatAPIError(
                    f"Chat request failed (HTTP {status}) after {attempt + 1} attempt(s)"
                ) from None
            except (URLError, TimeoutError, OSError, IncompleteRead) as exc:
                if attempt < self.config.retries:
                    time.sleep(_retry_delay(attempt, None))
                    continue
                # Do not include an exception string: it can contain endpoint
                # credentials or arbitrary provider content.
                raise ChatAPIError(
                    f"Chat transport failed after {attempt + 1} attempt(s) ({type(exc).__name__})"
                ) from None
            return _parse_response(body, time.perf_counter() - started)
        raise AssertionError("Unreachable retry state")
