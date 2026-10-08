"""Exercise the complete API/pipeline boundary using a local HTTP endpoint.

The server's scripted replies and household fixture are software integration
tests. They are never ALFWorld benchmark results or paid model calls.
"""

import json
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from jitmem.api import ChatAPIError, ChatClient
from jitmem.config import EnvironmentConfig, ExperimentConfig, ModelConfig, RunConfig
from jitmem.environments import FakeHouseholdEnvironment, fake_tasks
from jitmem.memory import MemoryBank, Trajectory, Turn
from jitmem.pipeline import Pipeline


@contextmanager
def scripted_endpoint(replies):
    """Serve queued completions and retain the actual serialized HTTP requests."""
    pending = deque(replies)
    records = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            records.append({"path": self.path, "headers": dict(self.headers), "payload": payload})
            if not pending:
                self.send_error(500, "Unexpected extra API call")
                return
            reply = pending.popleft()
            call_number = len(records)
            result = (
                reply
                if isinstance(reply, dict)
                else {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100 + call_number,
                        "completion_tokens": 10 + call_number,
                    },
                }
            )
            encoded = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", records, pending
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def redirect_endpoints(*, same_origin: bool, status: int):
    """Two loopback origins record whether any redirected request reaches a sink."""
    gateway_requests = []
    sink_requests = []
    sink_url = ""

    class SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            sink_requests.append({"method": self.command, "path": self.path})
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *_args):
            pass

    class GatewayHandler(SinkHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return super().do_POST()
            gateway_requests.append(
                {"authorization_present": self.headers.get("Authorization") is not None}
            )
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Location", "/sink" if same_origin else sink_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

    sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    sink_url = f"http://127.0.0.1:{sink.server_port}/sink"
    servers = [sink, gateway]
    threads = []
    for server in servers:
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        yield f"http://127.0.0.1:{gateway.server_port}/v1", gateway_requests, sink_requests
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


@pytest.mark.parametrize("same_origin", [False, True])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_http_redirects_are_rejected_without_retry_or_credential_forwarding(
    monkeypatch, same_origin, status
):
    monkeypatch.setenv("JITMEM_REDIRECT_FIXTURE_KEY", "redirect-fixture-credential")
    with redirect_endpoints(same_origin=same_origin, status=status) as (
        url,
        gateway_requests,
        sink_requests,
    ):
        client = ChatClient(
            ModelConfig(
                base_url=url,
                model="fixture-model",
                api_key_env="JITMEM_REDIRECT_FIXTURE_KEY",
                retries=3,
            )
        )
        with pytest.raises(ChatAPIError, match=f"HTTP {status}.*1 attempt") as caught:
            client.complete([{"role": "user", "content": "Local transport security fixture."}])
    assert gateway_requests == [{"authorization_present": True}]
    assert sink_requests == []
    assert "redirect-fixture-credential" not in str(caught.value)
    assert "Authorization" not in str(caught.value)


def run_config(base_url, *, max_steps=30, store_policy="judge"):
    return RunConfig(
        environment=EnvironmentConfig(backend="mock"),
        experiment=ExperimentConfig(max_steps=max_steps, store_policy=store_policy, seeds=[0]),
        executor=ModelConfig(
            base_url=base_url,
            model="executor-fixture",
            api_key_env="",
            temperature=1.0,
            max_tokens=4096,
            retries=0,
            token_limit_parameter="max_completion_tokens",
            extra_body={"top_p": 0.9},
        ),
        curator=ModelConfig(
            base_url=base_url,
            model="curator-fixture",
            api_key_env="",
            temperature=0.6,
            max_tokens=8192,
            retries=0,
            extra_body={"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}},
        ),
    )


def past_memory():
    return Trajectory(
        task_id="train-mug-example",
        task_description="put a mug in cabinet.",
        task_type="pick_and_place_simple",
        split="train",
        initial_observation="RAW_TRACE_SENTINEL: A mug is on a table.",
        turns=[
            Turn(
                "A mug is on a table.",
                "take mug 9 from table 7",
                "prior response",
                "You pick up the mug 9 from the table 7.",
            )
        ],
        judge_success=True,
    )


def assert_usage(result, records):
    count = len(records)
    assert result["usage"]["prompt_tokens"] == sum(100 + index for index in range(1, count + 1))
    assert result["usage"]["completion_tokens"] == sum(10 + index for index in range(1, count + 1))
    assert result["usage"]["complete"] is True
    assert len(result["calls"]) == count
    by_role = {}
    for index, (call, record) in enumerate(zip(result["calls"], records), 1):
        assert call["messages"] == record["payload"]["messages"]
        assert call["prompt_tokens"] == 100 + index
        assert call["completion_tokens"] == 10 + index
        assert call["latency_seconds"] >= 0
        assert call["finish_reason"] == "stop"
        assert call["incomplete"] is False
        usage = by_role.setdefault(
            call["role"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        usage["calls"] += 1
        usage["prompt_tokens"] += 100 + index
        usage["completion_tokens"] += 10 + index
    assert result["usage"]["by_role"] == by_role


def assert_judge_boundary(record):
    request = record["payload"]
    assert request["model"] == "executor-fixture"
    assert request["messages"][0]["content"].startswith("Assess whether")
    # The judge's output schema contains the word 'success'; its input trajectory
    # must contain only observations/actions, never native evaluation scalars.
    trace = request["messages"][-1]["content"]
    assert "Recorded episode:" in trace
    assert '"success":' not in trace
    assert '"reward":' not in trace
    assert "reward=" not in trace
    assert "environment_done" not in trace
    assert "judge_success" not in trace
    assert "RAW_TRACE_SENTINEL" not in trace


def test_local_http_pipeline_serializes_roles_memory_and_usage():
    actions = [
        "go to desk 1",
        "take mug 1 from desk 1",
        "go to cabinet 1",
        "open cabinet 1",
        "move mug 1 to cabinet 1",
    ]
    replies = ["CURATOR_PAYLOAD: Search, take, open, then place."]
    replies += [f"<action>{action}</action>" for action in actions]
    replies += [
        json.dumps({"success": True, "rationale": "Observed final placement.", "evidence_step": 5})
    ]
    bank = MemoryBank([past_memory()])
    with scripted_endpoint(replies) as (url, records, pending):
        config = run_config(url)
        pipeline = Pipeline(config, ChatClient(config.executor), ChatClient(config.curator))
        env = FakeHouseholdEnvironment()
        result, stored = pipeline.run_episode(fake_tasks(1)[0], env, bank)
        env.close()
        assert not pending
    assert result["success"] is True
    assert result["reward"] == 1.0
    assert result["steps"] == result["environment_steps"] == 5
    assert result["stored"] is True and stored is not None
    assert result["memory_size_before"] == 1
    assert result["retrieved"][0]["task_id"] == "train-mug-example"
    assert [call["role"] for call in result["calls"]] == ["curator"] + ["executor"] * 5 + ["judge"]
    assert len(records) == 7
    for record in records:
        assert record["path"] == "/v1/chat/completions"
        assert record["headers"]["Content-Type"] == "application/json"
        assert record["headers"]["Accept"] == "application/json"
        assert "Authorization" not in record["headers"]
        assert record["payload"]["stream"] is False
    curator = records[0]["payload"]
    assert curator["model"] == "curator-fixture"
    assert curator["temperature"] == 0.6
    assert curator["max_tokens"] == 8192
    assert curator["top_p"] == 0.95
    assert curator["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Question: put a mug in cabinet." in curator["messages"][-1]["content"]
    assert "RAW_TRACE_SENTINEL" in curator["messages"][-1]["content"]
    for record in records[1:]:
        executor = record["payload"]
        assert executor["model"] == "executor-fixture"
        assert executor["temperature"] == 1.0
        assert executor["max_completion_tokens"] == 4096
        assert "max_tokens" not in executor
        assert executor["top_p"] == 0.9
    for record in records[1:-1]:
        prompt = record["payload"]["messages"][-1]["content"]
        assert "CURATOR_PAYLOAD" in prompt
        assert "RAW_TRACE_SENTINEL" not in prompt
    assert_judge_boundary(records[-1])
    assert_usage(result, records)
    # Episode collection returns pending memory; batch commit is explicit.
    assert len(bank.entries) == 1
    bank.append_batch([stored])
    assert len(bank.entries) == 2
    assert bank.entries[-1].judge_success is True


@pytest.mark.parametrize("native_success,judge_success", [(True, False), (False, True)])
def test_local_http_memory_gate_is_independent_of_native_success(native_success, judge_success):
    actions = (
        [
            "go to desk 1",
            "take mug 1 from desk 1",
            "go to cabinet 1",
            "open cabinet 1",
            "move mug 1 to cabinet 1",
        ]
        if native_success
        else ["look"]
    )
    max_steps = 5 if native_success else 1
    replies = ["Proceed using observed prerequisites."]
    replies += [f"<action>{action}</action>" for action in actions]
    replies += [
        json.dumps(
            {"success": judge_success, "rationale": "Scripted disagreement.", "evidence_step": -1}
        )
    ]
    bank = MemoryBank()
    with scripted_endpoint(replies) as (url, records, pending):
        config = run_config(url, max_steps=max_steps)
        pipeline = Pipeline(config, ChatClient(config.executor), ChatClient(config.curator))
        env = FakeHouseholdEnvironment(max_steps=max_steps)
        result, stored = pipeline.run_episode(fake_tasks(1)[0], env, bank)
        env.close()
        assert not pending
    assert result["success"] is native_success
    assert result["reward"] == float(native_success)
    assert result["judge"]["success"] is judge_success
    assert result["stored"] is judge_success
    assert (stored is not None) is judge_success
    if stored is not None:
        assert stored.judge_success is True
        assert "reward" not in stored.to_dict()
        assert "success" not in stored.to_dict()
        bank.append_batch([stored])
    assert len(bank.entries) == int(judge_success)
    assert_judge_boundary(records[-1])
    assert_usage(result, records)


def test_local_http_store_all_retains_judge_rejected_episode():
    replies = [
        "Use the observed room.",
        "<action>look</action>",
        json.dumps({"success": False, "rationale": "Object was not placed.", "evidence_step": -1}),
    ]
    with scripted_endpoint(replies) as (url, records, pending):
        config = run_config(url, max_steps=1, store_policy="all")
        pipeline = Pipeline(config, ChatClient(config.executor), ChatClient(config.curator))
        env = FakeHouseholdEnvironment(max_steps=1)
        result, stored = pipeline.run_episode(fake_tasks(1)[0], env, MemoryBank())
        env.close()
        assert not pending
    assert result["success"] is False
    assert result["stored"] is True
    assert stored is not None and stored.judge_success is False
    assert_judge_boundary(records[-1])
    assert_usage(result, records)


def test_local_http_generation_limits_consume_budget_and_unknown_usage_stays_unknown():
    replies = [
        {
            "choices": [
                {"message": {"content": "PARTIAL_CURATOR_BRIEFING"}, "finish_reason": "length"}
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": "<action>go to desk 1</action>"},
                    "finish_reason": "max_completion_tokens",
                }
            ],
            "usage": {"prompt_tokens": 102, "completion_tokens": 12},
        },
        "<action>go to desk 1</action>",
        {
            "choices": [{"message": {"content": '{"success": true}'}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 104, "completion_tokens": 14},
        },
    ]
    with scripted_endpoint(replies) as (url, records, pending):
        config = run_config(url, max_steps=2)
        pipeline = Pipeline(config, ChatClient(config.executor), ChatClient(config.curator))
        env = FakeHouseholdEnvironment(max_steps=2)
        result, stored = pipeline.run_episode(fake_tasks(1)[0], env, MemoryBank())
        env.close()
        assert not pending
    # A length-limited but syntactically valid action must not mutate the game.
    assert len(records) == 4
    assert result["steps"] == 2
    assert result["environment_steps"] == 1
    assert result["invalid_actions"] == 1
    assert result["success"] is False
    assert result["truncated"] is True
    assert result["payload"] == ""
    assert result["judge"]["success"] is False
    assert result["stored"] is False and stored is None
    assert [failure["role"] for failure in result["generation_failures"]] == [
        "curator",
        "executor",
        "judge",
    ]
    assert result["usage"]["complete"] is False
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["completion_tokens"] is None
    assert result["usage"]["by_role"]["curator"] == {
        "calls": 1,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    assert result["usage"]["by_role"]["executor"] == {
        "calls": 2,
        "prompt_tokens": 205,
        "completion_tokens": 25,
    }
    assert result["usage"]["by_role"]["judge"] == {
        "calls": 1,
        "prompt_tokens": 104,
        "completion_tokens": 14,
    }
    assert result["calls"][0]["prompt_tokens"] is None
    assert result["calls"][0]["completion_tokens"] is None
    assert [call["finish_reason"] for call in result["calls"]] == [
        "length",
        "max_completion_tokens",
        "stop",
        "length",
    ]
    assert [call["incomplete"] for call in result["calls"]] == [True, True, False, True]
    for record in records[1:3]:
        assert "PARTIAL_CURATOR_BRIEFING" not in record["payload"]["messages"][-1]["content"]
    assert_judge_boundary(records[-1])
