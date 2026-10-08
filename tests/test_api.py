"""Protocol and failure tests use only a stub transport, never a paid API."""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from jitmem.api import (
    AuthenticationError,
    ChatAPIError,
    ChatClient,
    IncompleteResponseError,
    InvalidResponseError,
)
from jitmem.config import ModelConfig


def response(content: object = "take apple 1 from table 1", **overrides: object) -> dict:
    result = {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 23, "completion_tokens": 9},
    }
    result.update(overrides)
    return result


def body(value: object) -> BytesIO:
    return BytesIO(json.dumps(value).encode("utf-8"))


def http_error(status: int, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        "https://example.test/v1/chat/completions",
        status,
        "provider error",
        headers,
        BytesIO(b"secret-error-body"),
    )


class ChatClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {"TEST_JITMEM_KEY": "test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = ModelConfig(
            base_url="https://example.test/v1/",
            model="test-model",
            api_key_env="TEST_JITMEM_KEY",
            retries=2,
        )
        self.messages = [{"role": "user", "content": "找到苹果"}]

    def test_payload_headers_usage_and_text_parts(self) -> None:
        config = replace(
            self.config,
            token_limit_parameter="max_completion_tokens",
            max_tokens=123,
            extra_body={"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}},
        )
        parts = [{"type": "text", "text": "take "}, {"type": "text", "text": "apple 1"}]
        with patch("jitmem.api.urlopen", return_value=body(response(parts))) as transport:
            result = ChatClient(config).complete(self.messages)
        request = transport.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(transport.call_args.kwargs["timeout"], config.timeout_seconds)
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["messages"], self.messages)
        self.assertEqual(sent["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", sent)
        self.assertEqual(sent["temperature"], 1.0)
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["top_p"], 0.95)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("test-secret", json.dumps(sent))
        self.assertEqual(result.text, "take apple 1")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (23, 9))
        self.assertEqual(result.finish_reason, "stop")
        self.assertFalse(result.incomplete)
        self.assertGreaterEqual(result.latency_seconds, 0)

    def test_retry_only_rate_limits_server_errors_and_network(self) -> None:
        for failure in (
            http_error(429, "0"),
            http_error(503),
            URLError("network unavailable"),
            TimeoutError(),
            IncompleteRead(b"partial"),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch("jitmem.api.time.sleep") as sleep,
                patch("jitmem.api.urlopen", side_effect=[failure, body(response())]) as transport,
            ):
                result = ChatClient(self.config).complete(self.messages)
                self.assertEqual(transport.call_count, 2)
                self.assertEqual(sleep.call_count, 1)
                self.assertEqual(result.text, "take apple 1 from table 1")

    def test_retry_exhaustion_is_bounded(self) -> None:
        with (
            patch("jitmem.api.time.sleep") as sleep,
            patch(
                "jitmem.api.urlopen", side_effect=[http_error(503) for _ in range(3)]
            ) as transport,
        ):
            with self.assertRaisesRegex(ChatAPIError, "HTTP 503.*3 attempt"):
                ChatClient(self.config).complete(self.messages)
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_authentication_and_other_client_errors_are_not_retried(self) -> None:
        for status in (400, 401, 403, 404):
            with (
                self.subTest(status=status),
                patch("jitmem.api.time.sleep") as sleep,
                patch("jitmem.api.urlopen", side_effect=http_error(status)) as transport,
            ):
                expected = AuthenticationError if status in (401, 403) else ChatAPIError
                with self.assertRaises(expected) as caught:
                    ChatClient(self.config).complete(self.messages)
                self.assertEqual(transport.call_count, 1)
                sleep.assert_not_called()
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertNotIn("secret-error-body", str(caught.exception))

    def test_malformed_and_normal_empty_responses_fail_without_retry(self) -> None:
        invalid = [
            None,
            [],
            {},
            {"choices": []},
            response(None),
            response("   "),
            response([{"type": "text", "text": 4}]),
            response([{"text": "missing type"}]),
            response("ok", usage="invalid"),
            response("ok", usage={"prompt_tokens": True}),
            response("ok", usage={"completion_tokens": -1}),
            {"choices": [{"message": {"content": 7}, "finish_reason": "length"}]},
            {"choices": [{"message": {}, "finish_reason": "length"}]},
        ]
        for value in invalid:
            with (
                self.subTest(value=value),
                patch("jitmem.api.urlopen", return_value=body(value)) as transport,
            ):
                with self.assertRaises(InvalidResponseError):
                    ChatClient(self.config).complete(self.messages)
                self.assertEqual(transport.call_count, 1)
        with patch("jitmem.api.urlopen", return_value=BytesIO(b"not JSON")) as transport:
            with self.assertRaises(InvalidResponseError):
                ChatClient(self.config).complete(self.messages)
            self.assertEqual(transport.call_count, 1)
        self.assertTrue(issubclass(IncompleteResponseError, InvalidResponseError))

    def test_generation_limits_preserve_partial_text_usage_and_reason_without_retry(self) -> None:
        for reason in ("length", "max_tokens", "max_completion_tokens", "content_filter"):
            for content in (
                "partial response",
                "",
                None,
                [{"type": "text", "text": "partial list"}],
            ):
                value = response(content)
                value["choices"][0]["finish_reason"] = reason
                with (
                    self.subTest(reason=reason, content=content),
                    patch("jitmem.api.time.sleep") as sleep,
                    patch("jitmem.api.urlopen", return_value=body(value)) as transport,
                ):
                    result = ChatClient(self.config).complete(self.messages)
                expected = "partial list" if isinstance(content, list) else content or ""
                self.assertEqual(result.text, expected)
                self.assertEqual(result.finish_reason, reason)
                self.assertTrue(result.incomplete)
                self.assertEqual((result.prompt_tokens, result.completion_tokens), (23, 9))
                self.assertEqual(transport.call_count, 1)
                sleep.assert_not_called()

    def test_missing_usage_is_unknown_and_unauthed_endpoint_is_supported(self) -> None:
        config = replace(
            self.config,
            api_key_env="",
            temperature=None,
            base_url="https://example.test/v1/chat/completions",
        )
        with patch(
            "jitmem.api.urlopen", return_value=body({"choices": [{"message": {"content": "look"}}]})
        ) as transport:
            result = ChatClient(config).complete(self.messages)
        request = transport.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertNotIn("temperature", json.loads(request.data))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (None, None))
        self.assertIsNone(result.finish_reason)
        self.assertFalse(result.incomplete)

    def test_partial_or_null_usage_counts_remain_unknown(self) -> None:
        for usage, expected in (
            ({}, (None, None)),
            (None, (None, None)),
            ({"prompt_tokens": 23}, (23, None)),
            ({"completion_tokens": 0}, (None, 0)),
            ({"prompt_tokens": None, "completion_tokens": 9}, (None, 9)),
        ):
            with (
                self.subTest(usage=usage),
                patch("jitmem.api.urlopen", return_value=body(response(usage=usage))),
            ):
                result = ChatClient(self.config).complete(self.messages)
            self.assertEqual((result.prompt_tokens, result.completion_tokens), expected)

    def test_omit_temperature_removes_field_without_changing_other_settings(self) -> None:
        config = replace(self.config, temperature=0.6, omit_temperature=True)
        with patch("jitmem.api.urlopen", return_value=body(response())) as transport:
            ChatClient(config).complete(self.messages)
        payload = json.loads(transport.call_args.args[0].data)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], config.max_tokens)
        self.assertEqual(
            transport.call_args.args[0].get_header("Authorization"), "Bearer test-secret"
        )

    def test_missing_model_or_key_fails_before_network(self) -> None:
        with patch("jitmem.api.urlopen") as transport:
            with self.assertRaises(ChatAPIError):
                ChatClient(replace(self.config, model=""))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(AuthenticationError):
                    ChatClient(self.config)
            transport.assert_not_called()

    def test_invalid_messages_fail_before_network(self) -> None:
        invalid = [
            [],
            [{"role": "user"}],
            [{"role": "tool", "content": "x"}],
            [{"role": ["user"], "content": "x"}],
            [{"role": "user", "content": None}],
        ]
        for messages in invalid:
            with self.subTest(messages=messages), patch("jitmem.api.urlopen") as transport:
                with self.assertRaises(ValueError):
                    ChatClient(self.config).complete(messages)
                transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
