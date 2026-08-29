"""Unit tests for the Unchained framework.

These run fully offline: a FakeLLM stands in for any real provider, so no API
keys or network access are required.

    pytest
"""

import asyncio
import enum
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Literal, Optional

import pytest
import requests

# Make the top-level unchained.py importable regardless of how pytest is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import BaseModel, ValidationError

import unchained
from unchained import LLM, RAG, Agent, Memory, MockLLM, Router, Tool, tool


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------
class FakeLLM:
    """A scripted stand-in for LLM. Returns queued responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, response_format=None):
        self.calls.append(
            {"messages": messages, "tools": tools, "response_format": response_format}
        )
        resp = self.responses.pop(0) if self.responses else {}
        if callable(resp):
            resp = resp(messages, tools, response_format)
        return {
            "content": resp.get("content", ""),
            "tool_calls": resp.get("tool_calls", []),
            "usage": resp.get("usage", {}),
        }


# ---------------------------------------------------------------------------
# Tool system
# ---------------------------------------------------------------------------
@tool
def forecast(city: str, days: int = 3, tags: Optional[List[str]] = None) -> str:
    """Look up a forecast."""
    return f"{city}:{days}"


def test_tool_is_tool_instance_and_callable():
    assert isinstance(forecast, Tool)
    assert forecast("Paris", 2) == "Paris:2"  # still callable
    assert forecast.run({"city": "Rome"}) == "Rome:3"


def test_tool_schema_types_and_required():
    fn = forecast.schema["function"]
    assert forecast.schema["type"] == "function"
    assert fn["name"] == "forecast"
    assert fn["description"] == "Look up a forecast."

    props = fn["parameters"]["properties"]
    assert props["city"] == {"type": "string"}
    assert props["days"] == {"type": "integer"}
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"] == {"type": "string"}

    # only parameters without a default are required
    assert fn["parameters"]["required"] == ["city"]


def test_tool_error_is_caught_by_agent():
    @tool
    def boom(x: int) -> int:
        """Always explodes."""
        raise ValueError("nope")

    agent = Agent(FakeLLM([{"content": "done"}]), tools=[boom])
    # unknown tool
    assert "unknown tool" in agent._execute({"name": "ghost", "arguments": {}})
    # raising tool
    assert "Error executing 'boom'" in agent._execute({"name": "boom", "arguments": {"x": 1}})


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
def test_rag_ranks_relevant_document_first():
    rag = RAG()
    rag.add_many(
        [
            "Python is a programming language popular for data science.",
            "Cats are small domesticated animals kept as pets.",
            "JavaScript runs in the browser for web development.",
        ]
    )
    results = rag.search("data science with python", top_k=2)
    assert len(results) == 2
    assert results[0]["text"].startswith("Python")
    assert results[0]["score"] > 0
    assert results[0]["score"] >= results[1]["score"]


def test_rag_smoothed_idf_stays_positive():
    rag = RAG()
    # 'shared' appears in every document; naive idf would be 0.
    rag.add_many(["shared alpha term", "shared beta term", "shared gamma term"])
    assert rag._idf, "index should be built"
    assert all(value > 0 for value in rag._idf.values())


def test_rag_empty_search_returns_empty():
    assert RAG().search("anything") == []


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def test_memory_sliding_window_compresses_without_llm():
    memory = Memory(max_messages=4)  # no llm -> truncation strategy
    for i in range(5):
        memory.add("user", f"message number {i} with some content")

    assert len(memory.get()) == 2  # keeps recent half
    assert memory.summary  # overflow was summarised
    assert "message number 0" in memory.summary


def test_memory_add_preserves_extra_fields():
    memory = Memory()
    memory.add("assistant", "hi", tool_calls=[{"name": "x", "arguments": {}}])
    assert memory.get()[0]["tool_calls"][0]["name"] == "x"


# ---------------------------------------------------------------------------
# Agent (ReAct loop)
# ---------------------------------------------------------------------------
def test_agent_calls_tool_then_answers():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    llm = FakeLLM(
        [
            {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}, "id": "c1"}]},
            {"content": "The sum is 5."},
        ]
    )
    agent = Agent(llm, tools=[add])
    answer = agent.run("What is 2 + 3?")

    assert answer == "The sum is 5."
    tool_messages = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["content"] == "5"


def test_agent_uses_rag_context():
    rag = RAG()
    rag.add_many(["Unchained supports OpenAI, Anthropic and Ollama providers."])
    llm = FakeLLM([{"content": "It supports three providers."}])
    agent = Agent(llm, rag=rag)
    agent.run("Which providers are supported?")

    # the user message should have been augmented with retrieved context
    user_msg = agent.memory.get()[0]["content"]
    assert "Context:" in user_msg and "providers" in user_msg


def test_agent_structured_output():
    class Answer(BaseModel):
        value: int
        label: str

    llm = FakeLLM([{"content": '{"value": 42, "label": "answer"}'}])
    result = Agent(llm).run("give me the answer", response_format=Answer)

    assert isinstance(result, Answer)
    assert result.value == 42 and result.label == "answer"
    # no tools -> JSON mode requested from the provider
    assert llm.calls[0]["response_format"] is Answer


def test_agent_structured_output_strips_code_fences():
    class Item(BaseModel):
        name: str

    llm = FakeLLM([{"content": '```json\n{"name": "widget"}\n```'}])
    result = Agent(llm).run("name it", response_format=Item)
    assert result.name == "widget"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def _named_agent(name: str, answer: str) -> Agent:
    return Agent(FakeLLM([{"content": answer}]), name=name, description=f"{name} specialist")


def test_router_routes_to_named_agent():
    cost = _named_agent("cost", "cost answer")
    fit = _named_agent("fit", "fit answer")
    router = Router(FakeLLM([{"content": "the best choice is the cost agent"}]), agents=[cost, fit])
    assert router.route("How much will it cost?").name == "cost"


def test_router_run_all_collects_every_agent():
    router = Router(
        FakeLLM([]),
        agents=[_named_agent("a", "A"), _named_agent("b", "B"), _named_agent("c", "C")],
    )
    results = router.run_all("hello")
    assert results == {"a": "A", "b": "B", "c": "C"}


def test_router_synthesize_uses_synthesizer():
    agents = [_named_agent("a", "finding-a"), _named_agent("b", "finding-b")]
    synth = _named_agent("synth", "FINAL ANSWER")
    router = Router(FakeLLM([]), agents=agents, synthesizer=synth)
    assert router.synthesize("question") == "FINAL ANSWER"


def test_router_requires_agents():
    try:
        Router(FakeLLM([]), agents=[])
    except ValueError:
        return
    raise AssertionError("Router should reject an empty agent list")


# ---------------------------------------------------------------------------
# LLM message shaping (offline)
# ---------------------------------------------------------------------------
def test_openai_message_conversion():
    llm = LLM(provider="openai", api_key="test-key")
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "f", "arguments": {"x": 1}, "id": "c1"}],
        },
        {"role": "tool", "content": "42", "tool_call_id": "c1", "name": "f"},
    ]
    out = llm._to_openai_messages(messages)
    call = out[1]["tool_calls"][0]
    assert call["function"]["name"] == "f"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}
    assert out[2]["role"] == "tool" and out[2]["tool_call_id"] == "c1"


def test_anthropic_splits_system_and_converts_tools():
    llm = LLM(provider="anthropic", api_key="test-key")
    messages = [
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"name": "g", "arguments": {"y": 2}, "id": "u1"}],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "u1", "name": "g"},
    ]
    converted = llm._to_anthropic_messages(messages)
    assert converted[0]["content"][-1]["type"] == "tool_use"
    assert converted[1]["content"][0]["type"] == "tool_result"


def test_unknown_provider_rejected():
    try:
        LLM(provider="not-a-provider")
    except ValueError:
        return
    raise AssertionError("Unknown provider should raise ValueError")


# ---------------------------------------------------------------------------
# Provider request/response handling (offline via a monkeypatched Session.post)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None, lines=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"simulated HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def _patch_post(
    monkeypatch, payload=None, status_code=200, headers=None, lines=None, responses=None
):
    """Patch requests.Session.post; capture calls and optionally script responses.

    LLM makes requests through a persistent ``requests.Session`` (for
    connection reuse), so the fake is installed on the ``Session`` class
    rather than on the ``requests.post`` module function.
    """
    captured = {"calls": 0}
    queue = list(responses) if responses is not None else None

    def fake_post(self, url, **kwargs):
        captured["calls"] += 1
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        captured["stream"] = kwargs.get("stream", False)
        if queue is not None:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return _FakeResponse(payload, status_code, headers, lines)

    monkeypatch.setattr(unchained.requests.Session, "post", fake_post)
    # Make retry backoff instant in tests.
    monkeypatch.setattr(unchained.time, "sleep", lambda *_: None)
    return captured


class _BoomLLM:
    """An LLM stand-in whose chat always fails."""

    def chat(self, *args, **kwargs):
        raise RuntimeError("provider down")


def test_openai_chat_parses_content_and_tool_calls(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": '{"x": 1}'}}
                    ],
                }
            }
        ],
        "usage": {"total_tokens": 10},
    }
    captured = _patch_post(monkeypatch, payload)
    out = LLM(provider="openai", api_key="k").chat([{"role": "user", "content": "hi"}])
    assert out["content"] == ""  # null content normalised to ""
    assert out["tool_calls"] == [{"name": "f", "arguments": {"x": 1}, "id": "c1"}]
    assert out["usage"]["total_tokens"] == 10
    assert captured["url"].endswith("/v1/chat/completions")


def test_openai_chat_raises_on_non_retryable_http_error(monkeypatch):
    _patch_post(monkeypatch, {}, status_code=400)
    with pytest.raises(requests.HTTPError):
        LLM(provider="openai", api_key="k", max_retries=2).chat([{"role": "user", "content": "hi"}])


def test_anthropic_chat_splits_system_and_parses_tools(monkeypatch):
    payload = {
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "tool_use", "id": "u1", "name": "g", "input": {"y": 2}},
        ],
        "usage": {"input_tokens": 3},
    }
    captured = _patch_post(monkeypatch, payload)
    out = LLM(provider="anthropic", api_key="k").chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )
    assert out["content"] == "hello "
    assert out["tool_calls"][0] == {"name": "g", "arguments": {"y": 2}, "id": "u1"}
    assert captured["json"]["system"] == "sys"  # system message split out
    assert captured["url"].endswith("/v1/messages")


def test_ollama_chat_parses_response(monkeypatch):
    payload = {
        "message": {
            "content": "hi there",
            "tool_calls": [{"function": {"name": "h", "arguments": {"z": 3}}}],
        },
        "prompt_eval_count": 5,
        "eval_count": 7,
    }
    captured = _patch_post(monkeypatch, payload)
    out = LLM(provider="ollama").chat([{"role": "user", "content": "hi"}])
    assert out["content"] == "hi there"
    assert out["tool_calls"][0]["name"] == "h"
    assert out["tool_calls"][0]["arguments"] == {"z": 3}
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert captured["url"].endswith("/api/chat")


def test_llm_loads_handles_various_inputs():
    assert LLM._loads('{"a": 1}') == {"a": 1}
    assert LLM._loads({"b": 2}) == {"b": 2}
    assert LLM._loads("not json") == {}
    assert LLM._loads(None) == {}
    assert LLM._loads("") == {}


# ---------------------------------------------------------------------------
# Memory compression paths
# ---------------------------------------------------------------------------
def test_memory_llm_summarization_path():
    llm = FakeLLM([{"content": "SUMMARY TEXT"}])
    mem = Memory(max_messages=4, llm=llm)
    for i in range(5):
        mem.add("user", f"m{i}")
    assert mem.summary == "SUMMARY TEXT"
    assert len(mem.get()) == 2


def test_memory_summarizer_failure_falls_back_to_truncation():
    mem = Memory(max_messages=4, llm=_BoomLLM())
    for i in range(5):
        mem.add("user", f"message {i}")
    assert "message 0" in mem.summary  # fell back to truncation


# ---------------------------------------------------------------------------
# Agent iteration limits and structured output with tools
# ---------------------------------------------------------------------------
def test_agent_forces_final_answer_after_max_iterations():
    @tool
    def noop() -> str:
        """Do nothing."""
        return "ok"

    class LoopLLM:
        # Always asks for a tool while tools are offered; gives a final answer
        # only on the post-loop call made without tools.
        def chat(self, messages, tools=None, response_format=None):
            if tools:
                return {
                    "content": "",
                    "tool_calls": [{"name": "noop", "arguments": {}, "id": "x"}],
                    "usage": {},
                }
            return {"content": "final", "tool_calls": [], "usage": {}}

    agent = Agent(LoopLLM(), tools=[noop], max_iterations=3)
    assert agent.run("go") == "final"


def test_agent_structured_output_with_tools_runs_format_pass():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    class Ans(BaseModel):
        ok: bool

    llm = FakeLLM([{"content": "here is the answer"}, {"content": '{"ok": true}'}])
    result = Agent(llm, tools=[ping]).run("q", response_format=Ans)
    assert isinstance(result, Ans) and result.ok is True


# ---------------------------------------------------------------------------
# Router robustness
# ---------------------------------------------------------------------------
def test_router_run_all_isolates_a_failing_agent():
    good = _named_agent("good", "OK")
    bad = Agent(_BoomLLM(), name="bad", description="bad")
    results = Router(FakeLLM([]), agents=[good, bad]).run_all("q")
    assert results["good"] == "OK"
    assert results["bad"].startswith("Error:")


def test_router_run_all_sequential():
    router = Router(FakeLLM([]), agents=[_named_agent("a", "A"), _named_agent("b", "B")])
    assert router.run_all("q", parallel=False) == {"a": "A", "b": "B"}


def test_loads_object_extracts_and_handles_garbage():
    assert Agent._loads_object('prefix {"a": 1} suffix') == {"a": 1}
    assert Agent._loads_object('```json\n{"b": 2}\n```') == {"b": 2}
    assert Agent._loads_object("no json here") == {}


# ---------------------------------------------------------------------------
# Tier 2: retry / backoff
# ---------------------------------------------------------------------------
def test_request_retries_then_succeeds(monkeypatch):
    ok_payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(
        monkeypatch,
        responses=[
            _FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            _FakeResponse(status_code=503),
            _FakeResponse(ok_payload, status_code=200),
        ],
    )
    out = LLM(provider="openai", api_key="k", max_retries=3).chat(
        [{"role": "user", "content": "hi"}]
    )
    assert out["content"] == "hi"
    assert captured["calls"] == 3  # two retries then success


def test_request_raises_after_exhausting_retries(monkeypatch):
    _patch_post(monkeypatch, responses=[_FakeResponse(status_code=503) for _ in range(3)])
    with pytest.raises(requests.HTTPError):
        LLM(provider="openai", api_key="k", max_retries=2).chat([{"role": "user", "content": "hi"}])


def test_request_retries_on_connection_error(monkeypatch):
    ok_payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(
        monkeypatch,
        responses=[requests.ConnectionError("network down"), _FakeResponse(ok_payload)],
    )
    out = LLM(provider="openai", api_key="k", max_retries=2).chat(
        [{"role": "user", "content": "hi"}]
    )
    assert out["content"] == "hi"
    assert captured["calls"] == 2


def test_retry_delay_honours_retry_after():
    llm = LLM(provider="openai", api_key="k")
    exc = unchained._RetryableStatus(_FakeResponse(status_code=429, headers={"Retry-After": "7"}))
    assert llm._retry_delay(exc, attempt=0) == 7.0


# ---------------------------------------------------------------------------
# Tier 2: token usage normalisation + accumulation
# ---------------------------------------------------------------------------
def test_normalize_usage_across_provider_shapes():
    assert LLM._normalize_usage(
        {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    ) == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    # Anthropic shape
    assert LLM._normalize_usage({"input_tokens": 5, "output_tokens": 2}) == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    # Ollama shape
    assert LLM._normalize_usage({"prompt_eval_count": 8, "eval_count": 1}) == {
        "prompt_tokens": 8,
        "completion_tokens": 1,
        "total_tokens": 9,
    }
    assert LLM._normalize_usage(None)["total_tokens"] == 0


def test_agent_accumulates_usage():
    llm = FakeLLM(
        [
            {
                "content": "",
                "tool_calls": [{"name": "noop", "arguments": {}, "id": "1"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "content": "done",
                "tool_calls": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            },
        ]
    )

    @tool
    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent(llm, tools=[noop])
    agent.run("go")
    assert agent.usage == {"prompt_tokens": 14, "completion_tokens": 11, "total_tokens": 25}


# ---------------------------------------------------------------------------
# Tier 2: streaming
# ---------------------------------------------------------------------------
def test_openai_stream_parses_sse(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]
    _patch_post(monkeypatch, lines=lines)
    chunks = list(LLM(provider="openai", api_key="k").stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello"


def test_anthropic_stream_parses_sse(monkeypatch):
    lines = [
        'data: {"type":"content_block_delta","delta":{"text":"Hel"}}',
        'data: {"type":"content_block_delta","delta":{"text":"lo"}}',
        'data: {"type":"message_stop"}',
    ]
    _patch_post(monkeypatch, lines=lines)
    chunks = list(
        LLM(provider="anthropic", api_key="k").stream(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        )
    )
    assert "".join(chunks) == "Hello"


def test_ollama_stream_parses_jsonl(monkeypatch):
    lines = [
        '{"message":{"content":"Hel"}}',
        '{"message":{"content":"lo"},"done":true}',
    ]
    captured = _patch_post(monkeypatch, lines=lines)
    chunks = list(LLM(provider="ollama").stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hello"
    assert captured["stream"] is True


def test_agent_stream_without_tools(monkeypatch):
    class StreamLLM:
        def stream(self, messages):
            yield from ["Hel", "lo"]

    agent = Agent(StreamLLM())
    out = "".join(agent.stream("hi"))
    assert out == "Hello"
    assert agent.memory.get()[-1] == {"role": "assistant", "content": "Hello"}


def test_agent_stream_resolves_tools_then_streams():
    class ToolThenStreamLLM:
        def __init__(self):
            self.chat_calls = 0

        def chat(self, messages, tools=None, response_format=None):
            self.chat_calls += 1
            return {
                "content": "",
                "tool_calls": [{"name": "ping", "arguments": {}, "id": "1"}]
                if self.chat_calls == 1
                else [],
                "usage": {},
            }

        def stream(self, messages):
            yield from ["fin", "al"]

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(ToolThenStreamLLM(), tools=[ping])
    assert "".join(agent.stream("go")) == "final"
    tool_msgs = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "pong"


# ---------------------------------------------------------------------------
# Tier 2: observability callbacks
# ---------------------------------------------------------------------------
def test_callbacks_fire_across_the_loop():
    events = []

    class Recorder(unchained.Callback):
        def on_iteration(self, index):
            events.append(("iteration", index))

        def on_llm_call(self, messages, response):
            events.append(("llm", len(response["tool_calls"])))

        def on_tool_call(self, name, arguments, result):
            events.append(("tool", name, result))

        def on_finish(self, answer):
            events.append(("finish", answer))

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    llm = FakeLLM(
        [
            {"content": "", "tool_calls": [{"name": "ping", "arguments": {}, "id": "1"}]},
            {"content": "done"},
        ]
    )
    agent = Agent(llm, tools=[ping], callbacks=[Recorder()])
    agent.run("go")
    kinds = [e[0] for e in events]
    assert kinds.count("iteration") == 2
    assert ("tool", "ping", "pong") in events
    assert ("finish", "done") in events


def test_callback_errors_do_not_break_run():
    class BadCallback(unchained.Callback):
        def on_finish(self, answer):
            raise RuntimeError("callback boom")

    agent = Agent(FakeLLM([{"content": "ok"}]), callbacks=[BadCallback()])
    assert agent.run("hi") == "ok"  # run survives a failing callback


def test_logging_callback_is_a_callback():
    assert isinstance(unchained.LoggingCallback(), unchained.Callback)


# ---------------------------------------------------------------------------
# Tier 2: structured-output repair
# ---------------------------------------------------------------------------
def test_structured_output_repairs_invalid_json():
    class Person(BaseModel):
        name: str
        age: int

    # First response is missing `age`; the repair response is valid.
    llm = FakeLLM(
        [
            {"content": '{"name": "Ada"}'},
            {"content": '{"name": "Ada", "age": 36}'},
        ]
    )
    result = Agent(llm, structured_retries=1).run("extract", response_format=Person)
    assert result.name == "Ada" and result.age == 36


def test_structured_output_raises_when_repair_budget_exhausted():
    class Person(BaseModel):
        name: str
        age: int

    llm = FakeLLM([{"content": '{"name": "Ada"}'}, {"content": '{"name": "Ada"}'}])
    with pytest.raises(ValidationError):
        Agent(llm, structured_retries=1).run("extract", response_format=Person)


# ---------------------------------------------------------------------------
# Tier 2: SQLiteMemory example
# ---------------------------------------------------------------------------
def test_sqlite_memory_persists_and_reloads(tmp_path):
    from examples.sqlite_memory import SQLiteMemory

    db = str(tmp_path / "mem.db")
    mem = SQLiteMemory(db_path=db, session_id="s1")
    mem.add("user", "hello")
    mem.add("assistant", "hi", tool_calls=[{"name": "x", "arguments": {}}])
    mem.close()

    reloaded = SQLiteMemory(db_path=db, session_id="s1")
    assert [m["content"] for m in reloaded.get()] == ["hello", "hi"]
    assert reloaded.get()[1]["tool_calls"][0]["name"] == "x"  # extra fields persisted

    other = SQLiteMemory(db_path=db, session_id="s2")  # sessions are isolated
    assert other.get() == []
    reloaded.close()
    other.close()


# ---------------------------------------------------------------------------
# Tier 3: MockLLM
# ---------------------------------------------------------------------------
def test_mockllm_is_llm_subclass_with_fixed_reply():
    llm = MockLLM(reply="hello world")
    assert isinstance(llm, LLM)
    assert llm.chat([])["content"] == "hello world"
    assert "".join(llm.stream([])) == "hello world"


def test_mockllm_script_drives_agent_tool_loop():
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    llm = MockLLM(
        script=[
            {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}, "id": "1"}]},
            {"content": "The answer is 5."},
        ]
    )
    agent = Agent(llm, tools=[add])
    assert agent.run("2 + 3?") == "The answer is 5."
    tool_msgs = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "5"


def test_mockllm_handler():
    def handler(messages, tools):
        return f"you said: {messages[-1]['content']}"

    llm = MockLLM(handler=handler)
    assert llm.chat([{"role": "user", "content": "hi"}])["content"] == "you said: hi"


# ---------------------------------------------------------------------------
# Tier 3: RAG with pluggable embeddings
# ---------------------------------------------------------------------------
def test_rag_embedding_mode_ranks_by_cosine():
    def embed(texts):
        # Two-axis toy embedding: presence of "python" and "cat".
        return [[float("python" in t.lower()), float("cat" in t.lower())] for t in texts]

    rag = RAG(embed_fn=embed)
    rag.add_many(
        ["python programming language", "cats are cute animals", "python and cats together"]
    )
    hits = rag.search("python", top_k=2)
    assert "python" in hits[0]["text"].lower()
    assert hits[0]["score"] >= hits[1]["score"] > 0
    assert not rag._vectors  # TF-IDF index is not built in embedding mode


# ---------------------------------------------------------------------------
# Tier 3: response caching
# ---------------------------------------------------------------------------
def test_llm_cache_avoids_second_request(monkeypatch):
    payload = {"choices": [{"message": {"content": "cached", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(monkeypatch, payload)
    llm = LLM(provider="openai", api_key="k", cache=True)
    messages = [{"role": "user", "content": "same question"}]
    first = llm.chat(messages)
    second = llm.chat(messages)
    assert first["content"] == second["content"] == "cached"
    assert captured["calls"] == 1  # second call served from cache


def test_llm_does_not_cache_by_default(monkeypatch):
    payload = {"choices": [{"message": {"content": "x", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(monkeypatch, payload)
    llm = LLM(provider="openai", api_key="k")
    llm.chat([{"role": "user", "content": "hi"}])
    llm.chat([{"role": "user", "content": "hi"}])
    assert captured["calls"] == 2


# ---------------------------------------------------------------------------
# Tier 3: hardened code-execution example
# ---------------------------------------------------------------------------
def test_coder_run_python_executes():
    from examples.coder import run_python

    assert run_python("print(6 * 7)") == "42"


def test_coder_run_python_reports_errors():
    from examples.coder import run_python

    assert run_python("raise ValueError('boom')").startswith("Error")


def test_coder_run_python_times_out(monkeypatch):
    import examples.coder as coder

    monkeypatch.setattr(coder, "EXEC_TIMEOUT_SECONDS", 1)
    assert "timed out" in coder.run_python("while True:\n    pass")


# ---------------------------------------------------------------------------
# Tier 4: richer tool schemas (Literal, Enum, nested BaseModel)
# ---------------------------------------------------------------------------
def test_tool_schema_literal_becomes_enum():
    @tool
    def set_mode(mode: Literal["heat", "cool", "off"]) -> str:
        """Set the mode."""
        return mode

    props = set_mode.schema["function"]["parameters"]["properties"]
    assert props["mode"] == {"type": "string", "enum": ["heat", "cool", "off"]}


def test_tool_schema_literal_mixed_types_omits_single_type():
    @tool
    def pick(value: Literal[1, "two", 3]) -> str:
        """Pick a value."""
        return str(value)

    props = pick.schema["function"]["parameters"]["properties"]
    assert props["value"]["enum"] == [1, "two", 3]
    assert "type" not in props["value"]  # mixed types -> no single JSON type


def test_tool_schema_enum_class_becomes_enum():
    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @tool
    def paint(color: Color) -> str:
        """Paint something."""
        return color.value

    props = paint.schema["function"]["parameters"]["properties"]
    assert props["color"] == {"type": "string", "enum": ["red", "blue"]}


def test_tool_schema_nested_pydantic_model_is_inlined():
    class Address(BaseModel):
        city: str
        zip_code: str

    @tool
    def ship(address: Address) -> str:
        """Ship to an address."""
        return address.city

    props = ship.schema["function"]["parameters"]["properties"]
    assert props["address"]["type"] == "object"
    assert "city" in props["address"]["properties"]
    assert "zip_code" in props["address"]["properties"]


# ---------------------------------------------------------------------------
# Tier 4: async tools and agents
# ---------------------------------------------------------------------------
def test_async_tool_runs_via_tool_run():
    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        return a + b

    assert async_add.is_async is True
    assert async_add.run({"a": 2, "b": 3}) == 5


def test_async_tool_inside_running_loop_raises_clear_error():
    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        return a + b

    async def _inner():
        # Calling the sync .run() from inside a running loop can't use
        # asyncio.run() - it should fail with a clear, actionable message.
        with pytest.raises(RuntimeError, match="already running"):
            async_add.run({"a": 1, "b": 1})

    asyncio.run(_inner())


def test_agent_with_async_tool_end_to_end():
    @tool
    async def fetch(ticker: str) -> str:
        """Fetch a price."""
        return f"{ticker}:100"

    llm = FakeLLM(
        [
            {"tool_calls": [{"name": "fetch", "arguments": {"ticker": "ACME"}, "id": "1"}]},
            {"content": "It's 100."},
        ]
    )
    agent = Agent(llm, tools=[fetch])
    assert agent.run("price?") == "It's 100."
    tool_msgs = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "ACME:100"


def test_agent_arun_matches_sync_run():
    llm = FakeLLM([{"content": "async answer"}])
    agent = Agent(llm)
    result = asyncio.run(agent.arun("hi"))
    assert result == "async answer"


def test_llm_achat_offloads_to_thread(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    _patch_post(monkeypatch, payload)
    llm = LLM(provider="openai", api_key="k")

    async def _inner():
        return await llm.achat([{"role": "user", "content": "hi"}])

    out = asyncio.run(_inner())
    assert out["content"] == "hi"


# ---------------------------------------------------------------------------
# Tier 4: concurrent tool-call execution within a turn
# ---------------------------------------------------------------------------
def test_multiple_tool_calls_run_concurrently():
    barrier = threading.Barrier(2, timeout=5)

    @tool
    def slow_a() -> str:
        """Slow tool A."""
        barrier.wait()  # only returns once both tools have started
        return "a-done"

    @tool
    def slow_b() -> str:
        """Slow tool B."""
        barrier.wait()
        return "b-done"

    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {"name": "slow_a", "arguments": {}, "id": "1"},
                    {"name": "slow_b", "arguments": {}, "id": "2"},
                ]
            },
            {"content": "both done"},
        ]
    )
    agent = Agent(llm, tools=[slow_a, slow_b])
    # If calls ran sequentially, the second tool would block forever waiting
    # on a barrier the first tool (already returned) can never reach again.
    assert agent.run("go") == "both done"
    tool_msgs = [m for m in agent.memory.get() if m["role"] == "tool"]
    # Order in memory still matches the order the model requested them in.
    assert [m["name"] for m in tool_msgs] == ["slow_a", "slow_b"]
    assert [m["content"] for m in tool_msgs] == ["a-done", "b-done"]


def test_single_tool_call_does_not_use_thread_pool(monkeypatch):
    @tool
    def solo() -> str:
        """Solo tool."""
        return "solo-done"

    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("a single tool call should skip the thread pool entirely")

    monkeypatch.setattr(unchained, "ThreadPoolExecutor", _fail_if_constructed)

    llm = FakeLLM(
        [
            {"tool_calls": [{"name": "solo", "arguments": {}, "id": "1"}]},
            {"content": "done"},
        ]
    )
    agent = Agent(llm, tools=[solo])
    assert agent.run("go") == "done"
    tool_msgs = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "solo-done"


def test_concurrent_tool_calls_during_streaming():
    barrier = threading.Barrier(2, timeout=5)

    @tool
    def stream_a() -> str:
        """Stream tool A."""
        barrier.wait()
        return "a"

    @tool
    def stream_b() -> str:
        """Stream tool B."""
        barrier.wait()
        return "b"

    class ToolThenStreamLLM:
        def __init__(self):
            self.chat_calls = 0

        def chat(self, messages, tools=None, response_format=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {"name": "stream_a", "arguments": {}, "id": "1"},
                        {"name": "stream_b", "arguments": {}, "id": "2"},
                    ],
                    "usage": {},
                }
            return {"content": "", "tool_calls": [], "usage": {}}

        def stream(self, messages):
            yield "done"

    agent = Agent(ToolThenStreamLLM(), tools=[stream_a, stream_b])
    assert "".join(agent.stream("go")) == "done"


# ---------------------------------------------------------------------------
# Tier 4: token-aware Memory
# ---------------------------------------------------------------------------
def test_memory_max_tokens_shrinks_window_further():
    # Two huge messages alone (~1000 tokens each) already blow a 500-token
    # budget, even though max_messages=20 wouldn't trigger compression yet.
    memory = Memory(max_messages=20, max_tokens=500)
    memory.add("user", "x" * 4000)  # ~1000 estimated tokens
    memory.add("user", "y" * 4000)  # ~1000 estimated tokens more
    kept = memory.get()
    assert len(kept) <= 1  # shrunk below the message-count cap to fit budget
    assert memory.summary  # the rest was compressed into the summary


def test_memory_max_tokens_keeps_at_least_one_message():
    memory = Memory(max_messages=20, max_tokens=1)  # impossible budget
    memory.add("user", "hello world")
    assert len(memory.get()) == 1  # never compresses away the last message


def test_memory_without_max_tokens_ignores_token_budget():
    # Default behaviour (max_tokens=None) is unaffected: only max_messages counts.
    memory = Memory(max_messages=4)
    memory.add("user", "x" * 10_000)  # would blow any reasonable token budget
    assert len(memory.get()) == 1
    assert memory.summary == ""  # no compression triggered yet (1 <= 4)


def test_estimate_tokens_helper():
    assert unchained._estimate_tokens("abcd") == 1  # 4 chars / 4 = 1 token
    assert unchained._estimate_tokens("a" * 40) == 10
    assert unchained._estimate_tokens("") == 1  # minimum of 1


# ---------------------------------------------------------------------------
# Tier 4: HTTP session reuse
# ---------------------------------------------------------------------------
def test_llm_reuses_a_single_session_across_requests(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    _patch_post(monkeypatch, payload)
    llm = LLM(provider="openai", api_key="k")
    assert isinstance(llm.session, requests.Session)
    session_before = llm.session
    llm.chat([{"role": "user", "content": "one"}])
    llm.chat([{"role": "user", "content": "two"}])
    # Same Session instance backs both calls - connections can be pooled.
    assert llm.session is session_before


def test_llm_close_closes_the_session():
    llm = LLM(provider="openai", api_key="k")
    closed = {"value": False}
    original_close = llm.session.close

    def fake_close():
        closed["value"] = True
        original_close()

    llm.session.close = fake_close
    llm.close()
    assert closed["value"] is True


# ---------------------------------------------------------------------------
# Tier 4: bounded + TTL response cache
# ---------------------------------------------------------------------------
def test_cache_evicts_oldest_entry_past_cache_size(monkeypatch):
    calls = {"n": 0}

    def fake_post(self, url, **kwargs):
        calls["n"] += 1
        content = f"response-{calls['n']}"
        return _FakeResponse(
            {"choices": [{"message": {"content": content, "tool_calls": []}}], "usage": {}}
        )

    monkeypatch.setattr(unchained.requests.Session, "post", fake_post)
    llm = LLM(provider="openai", api_key="k", cache=True, cache_size=2)

    llm.chat([{"role": "user", "content": "one"}])
    llm.chat([{"role": "user", "content": "two"}])
    assert len(llm.cache) == 2

    llm.chat([{"role": "user", "content": "three"}])  # evicts "one"
    assert len(llm.cache) == 2
    assert calls["n"] == 3

    # "one" was evicted, so asking again re-fetches (call count increases).
    llm.chat([{"role": "user", "content": "one"}])
    assert calls["n"] == 4

    # "three" is still cached (was not evicted).
    llm.chat([{"role": "user", "content": "three"}])
    assert calls["n"] == 4


def test_cache_ttl_expires_old_entries(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(monkeypatch, payload)

    clock = {"now": 1000.0}
    monkeypatch.setattr(unchained.time, "time", lambda: clock["now"])

    llm = LLM(provider="openai", api_key="k", cache=True, cache_ttl=60)
    messages = [{"role": "user", "content": "hi"}]
    llm.chat(messages)
    assert captured["calls"] == 1

    clock["now"] += 30  # still within TTL
    llm.chat(messages)
    assert captured["calls"] == 1

    clock["now"] += 40  # now 70s later - past the 60s TTL
    llm.chat(messages)
    assert captured["calls"] == 2


def test_cache_without_ttl_never_expires(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi", "tool_calls": []}}], "usage": {}}
    captured = _patch_post(monkeypatch, payload)
    llm = LLM(provider="openai", api_key="k", cache=True)  # cache_ttl=None
    messages = [{"role": "user", "content": "hi"}]
    llm.chat(messages)
    llm.chat(messages)
    assert captured["calls"] == 1


# ---------------------------------------------------------------------------
# Tier 4: per-provider base_url environment variables
# ---------------------------------------------------------------------------
def test_openai_base_url_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai")
    llm = LLM(provider="openai", api_key="k")
    assert llm.base_url == "https://api.groq.com/openai"


def test_ollama_base_url_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434")
    llm = LLM(provider="ollama")
    assert llm.base_url == "http://remote-ollama:11434"


def test_explicit_base_url_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://from-env.example.com")
    llm = LLM(provider="openai", api_key="k", base_url="https://explicit.example.com")
    assert llm.base_url == "https://explicit.example.com"


def test_no_env_var_falls_back_to_provider_default(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    llm = LLM(provider="openai", api_key="k")
    assert llm.base_url == "https://api.openai.com"


def test_pydantic_schema_helper_handles_v1_and_non_pydantic():
    class FakeV1Model:
        """Duck-types a pydantic v1 model (schema() but no model_json_schema())."""

        @staticmethod
        def schema():
            return {"v1": True}

    class NotAModel:
        pass

    assert unchained._pydantic_schema(FakeV1Model) == {"v1": True}
    assert unchained._pydantic_schema(NotAModel) == {}


# ---------------------------------------------------------------------------
# Tier 5: hardening regressions
#
# Each test below pins a defect that was reachable in the released behaviour.
# The comment on each names what used to happen.
# ---------------------------------------------------------------------------
def _tool_conversation(memory):
    """Build a window whose compression boundary falls inside a tool group."""
    memory.messages = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "f", "arguments": {}, "id": "c3"},
                {"name": "g", "arguments": {}, "id": "c4"},
            ],
        },
    ]
    memory.add("tool", "r3", tool_call_id="c3", name="f")
    memory.add("tool", "r4", tool_call_id="c4", name="g")
    memory.add("assistant", "a2")
    memory.add("user", "q3")  # 7th message -> triggers compression
    return memory


def _assert_tool_results_are_paired(window):
    """Every tool message must follow an assistant turn that offered its id."""
    offered = set()
    for message in window:
        for call in message.get("tool_calls") or []:
            offered.add(call["id"])
        if message["role"] == "tool":
            assert message["tool_call_id"] in offered


def test_memory_compression_never_orphans_a_tool_result():
    # Previously the boundary cut between an assistant's tool_calls and the
    # tool results answering them, so the window began with an unpaired tool
    # message. OpenAI rejects that with HTTP 400; Anthropic rejects the
    # equivalent tool_result block.
    memory = _tool_conversation(Memory(max_messages=6))
    window = memory.get()

    assert window[0]["role"] != "tool", "window must not start with an orphaned tool result"
    _assert_tool_results_are_paired(window)


def test_memory_compression_keeps_a_sendable_window_for_openai():
    # End-to-end shape check: the compressed window survives conversion.
    memory = _tool_conversation(Memory(max_messages=6))
    converted = LLM(provider="openai", api_key="k")._to_openai_messages(memory.get())
    _assert_tool_results_are_paired(converted)


def test_memory_tool_group_rule_does_not_disturb_plain_conversations():
    # The boundary only moves when it lands on a tool message; ordinary
    # windows keep exactly the previous sizing.
    memory = Memory(max_messages=4)
    for i in range(5):
        memory.add("user", f"message number {i}")
    assert len(memory.get()) == 2  # unchanged: recent half
    assert [m["role"] for m in memory.get()] == ["user", "user"]


def test_memory_tool_group_start_walks_back_to_the_assistant():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "id": "c1"}]},
        {"role": "tool", "content": "r", "tool_call_id": "c1"},
        {"role": "tool", "content": "r", "tool_call_id": "c2"},
    ]
    assert Memory._tool_group_start(messages, 3) == 1  # inside a group -> back to assistant
    assert Memory._tool_group_start(messages, 2) == 1
    assert Memory._tool_group_start(messages, 1) == 1  # already valid -> unchanged
    assert Memory._tool_group_start(messages, 0) == 0
    # A window that is nothing but tool results clamps at 0 rather than going negative.
    assert Memory._tool_group_start([{"role": "tool"}, {"role": "tool"}], 1) == 0


def test_memory_token_budget_still_shrinks_below_the_message_cap():
    # Guards the _compress rewrite against index-arithmetic regressions.
    memory = Memory(max_messages=20, max_tokens=500)
    memory.add("user", "x" * 4000)
    memory.add("user", "y" * 4000)
    assert len(memory.get()) <= 1
    assert memory.summary


# ---------------------------------------------------------------------------
# Router fails closed
# ---------------------------------------------------------------------------
def test_router_raises_rather_than_guessing_an_agent():
    # Previously an unmatched reply silently returned agents[0], and an empty
    # reply returned the longest-named agent (because "" is a substring of
    # every name) - routing a query to an agent nobody chose.
    router = Router(
        FakeLLM([]),
        agents=[_named_agent("billing", "b"), _named_agent("admin_delete", "a")],
    )
    for unroutable in ("", "   ", "i cannot determine which agent", "banana", "finance"):
        with pytest.raises(unchained.RoutingError):
            router._match(unroutable)


def test_router_refuses_an_ambiguous_reply_naming_two_agents():
    router = Router(
        FakeLLM([]),
        agents=[_named_agent("billing", "b"), _named_agent("admin_delete", "a")],
    )
    with pytest.raises(unchained.RoutingError):
        router._match("either billing or admin_delete would work")


def test_router_does_not_match_a_name_inside_a_longer_word():
    # "fit" must not match a reply that merely mentions "profit".
    router = Router(FakeLLM([]), agents=[_named_agent("fit", "f"), _named_agent("cost", "c")])
    assert router._match("look at the cost side").name == "cost"
    with pytest.raises(unchained.RoutingError):
        router._match("this is about profit margins")


def test_router_still_matches_exact_and_embedded_names():
    # Backwards compatibility: the cases that resolved before still resolve.
    router = Router(FakeLLM([]), agents=[_named_agent("cost", "c"), _named_agent("fit", "f")])
    assert router._match("cost").name == "cost"
    assert router._match("  Cost  ").name == "cost"
    assert router._match("the best choice is the cost agent").name == "cost"


def test_router_fallback_agent_receives_unroutable_queries():
    triage = _named_agent("triage", "triaged")
    router = Router(
        FakeLLM([{"content": "no idea"}]),
        agents=[_named_agent("cost", "c"), _named_agent("fit", "f")],
        fallback=triage,
    )
    assert router._match("no idea").name == "triage"
    assert router.run("something unroutable") == "triaged"


def test_router_run_propagates_routing_error():
    router = Router(FakeLLM([{"content": "no idea"}]), agents=[_named_agent("cost", "c")])
    with pytest.raises(unchained.RoutingError):
        router.run("something unroutable")


def test_routing_error_is_exported():
    assert "RoutingError" in unchained.__all__
    assert issubclass(unchained.RoutingError, RuntimeError)


# ---------------------------------------------------------------------------
# RAG rejects mismatched metadata
# ---------------------------------------------------------------------------
def test_rag_add_many_rejects_metadata_length_mismatch():
    # Previously zip() truncated to the shorter list, so documents vanished
    # silently in TF-IDF mode.
    rag = RAG()
    with pytest.raises(ValueError):
        rag.add_many(["one", "two", "three"], [{"m": 1}, {"m": 2}])
    assert len(rag) == 0  # nothing was half-indexed


def test_rag_add_many_mismatch_cannot_corrupt_the_embedding_index():
    # Previously this left 3 embeddings against 1 document, and search()
    # raised IndexError later, far from the cause.
    rag = RAG(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    with pytest.raises(ValueError):
        rag.add_many(["one", "two", "three"], [{"m": 1}])
    assert len(rag.docs) == len(rag._embeddings)


def test_rag_add_many_accepts_matching_lengths_and_omitted_metadata():
    rag = RAG()
    rag.add_many(["one", "two"], [{"m": 1}, {"m": 2}])
    rag.add_many(["three", "four"])  # metadatas omitted entirely
    assert len(rag) == 4
    assert rag.metadata[0] == {"m": 1}
    assert rag.metadata[3] == {}


# ---------------------------------------------------------------------------
# No placeholder credential on the wire
# ---------------------------------------------------------------------------
def test_openai_omits_the_auth_header_when_no_key_is_configured(monkeypatch):
    # Previously sent the literal header "Authorization: Bearer None".
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    headers = LLM(provider="openai")._openai_headers()
    assert "Authorization" not in headers
    assert "None" not in json.dumps(headers)


def test_openai_sends_the_auth_header_when_a_key_is_configured():
    headers = LLM(provider="openai", api_key="sk-test")._openai_headers()
    assert headers["Authorization"] == "Bearer sk-test"


def test_anthropic_omits_the_api_key_header_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    headers = LLM(provider="anthropic")._anthropic_headers()
    assert "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"


def test_anthropic_sends_the_api_key_header_when_set():
    headers = LLM(provider="anthropic", api_key="sk-ant")._anthropic_headers()
    assert headers["x-api-key"] == "sk-ant"


def test_streaming_paths_use_the_same_header_policy(monkeypatch):
    # The stream() variants built their own header dicts and had the same bug.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def _capture(path, **kwargs):
        seen.update(kwargs.get("headers") or {})
        return _FakeResponse(lines=["data: [DONE]"])

    llm = LLM(provider="openai")
    monkeypatch.setattr(llm, "_request", _capture)
    list(llm.stream([{"role": "user", "content": "hi"}]))
    assert "Authorization" not in seen


# ---------------------------------------------------------------------------
# Tool fan-out is bounded
# ---------------------------------------------------------------------------
def test_tool_fan_out_is_capped_regardless_of_how_many_calls_the_model_asks_for():
    # How many tool calls arrive in a turn is chosen by the model, i.e. it is
    # untrusted input. Previously max_workers=len(calls), so a single
    # response sized the thread pool (250 calls -> ~176 live threads).
    peak = {"n": 0}
    running = {"n": 0}
    lock = threading.Lock()

    @tool
    def slow(x: int) -> str:
        """Occupy a worker for a moment."""
        with lock:
            running["n"] += 1
            peak["n"] = max(peak["n"], running["n"])
        time.sleep(0.01)
        with lock:
            running["n"] -= 1
        return f"ok{x}"

    agent = Agent(FakeLLM([]), tools=[slow], max_tool_workers=4)
    calls = [{"name": "slow", "arguments": {"x": i}, "id": f"c{i}"} for i in range(60)]
    agent._execute_calls(calls)

    assert peak["n"] <= 4, f"pool grew to {peak['n']} concurrent workers"
    # Every call still ran, and the results kept their original order.
    observations = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert [m["content"] for m in observations] == [f"ok{i}" for i in range(60)]


def test_max_tool_workers_defaults_to_a_bound_and_is_never_below_one():
    assert Agent(FakeLLM([])).max_tool_workers == 8
    assert Agent(FakeLLM([]), max_tool_workers=0).max_tool_workers == 1
    assert Agent(FakeLLM([]), max_tool_workers=-5).max_tool_workers == 1


def test_single_tool_call_still_skips_the_pool_entirely():
    @tool
    def echo(x: int) -> str:
        """Echo the argument."""
        return f"ok{x}"

    agent = Agent(FakeLLM([]), tools=[echo])
    before = threading.active_count()
    agent._execute_calls([{"name": "echo", "arguments": {"x": 1}, "id": "c0"}])
    assert threading.active_count() == before


def test_routing_error_message_truncates_a_runaway_model_reply():
    router = Router(FakeLLM([]), agents=[_named_agent("cost", "c")])
    with pytest.raises(unchained.RoutingError) as excinfo:
        router._match("x" * 5000)
    assert len(str(excinfo.value)) < 400


# ---------------------------------------------------------------------------
# Tier 6: tool authorization (policy layer)
#
# The invariant under test throughout: a model-requested call reaches a
# function only after Tool.validate_arguments and ToolPolicy.authorize have
# both passed. Each test asserts on a side-effect log, not just the returned
# string, so "was refused" means "did not run" rather than "said no".
# ---------------------------------------------------------------------------
@pytest.fixture
def ran():
    """Records which tool functions actually executed."""
    return []


@pytest.fixture
def policy_tools(ran):
    @tool
    def search(query: str) -> str:
        """Read-only lookup."""
        ran.append(("search", query))
        return f"results for {query}"

    @tool(permissions={"db:write"}, side_effects=True)
    def write_row(table: str, value: str) -> str:
        """Insert a row."""
        ran.append(("write_row", table))
        return "written"

    @tool(permissions={"db:admin"}, side_effects=True, requires_approval=True)
    def drop_table(table: str) -> str:
        """Destructive."""
        ran.append(("drop_table", table))
        return "dropped"

    return [search, write_row, drop_table]


def _call(name, **arguments):
    return {"name": name, "arguments": arguments, "id": "call-1"}


# --- authorized execution --------------------------------------------------
def test_authorized_tool_executes(policy_tools, ran):
    agent = Agent(
        FakeLLM([]), tools=policy_tools, policy=unchained.PermissionPolicy(granted={"db:write"})
    )
    assert agent._execute(_call("write_row", table="t", value="v")) == "written"
    assert ran == [("write_row", "t")]


def test_tool_without_declared_permissions_passes_a_permission_policy(policy_tools, ran):
    # A tool that declares no permissions requires none. The agent's own
    # tools list is the first allowlist; PermissionPolicy narrows it.
    agent = Agent(FakeLLM([]), tools=policy_tools, policy=unchained.PermissionPolicy(granted=set()))
    assert agent._execute(_call("search", query="q")) == "results for q"
    assert ran == [("search", "q")]


# --- denied execution ------------------------------------------------------
def test_denied_tool_does_not_execute(policy_tools, ran):
    agent = Agent(
        FakeLLM([]), tools=policy_tools, policy=unchained.PermissionPolicy(granted={"db:read"})
    )
    observation = agent._execute(_call("write_row", table="t", value="v"))
    assert "db:write" in observation
    assert ran == []  # the function never ran


def test_denial_is_reported_to_the_model_not_raised(policy_tools, ran):
    # A denied tool must not crash the run: the model sees an observation and
    # the ReAct loop continues, so it can choose something else.
    script = [
        {
            "content": "",
            "tool_calls": [
                {"name": "write_row", "arguments": {"table": "t", "value": "v"}, "id": "x"}
            ],
        },
        {"content": "I was refused, so here is a plain answer."},
    ]
    agent = Agent(
        MockLLM(script=script),
        tools=policy_tools,
        policy=unchained.PermissionPolicy(granted={"db:read"}),
    )
    assert agent.run("write something") == "I was refused, so here is a plain answer."
    assert ran == []


def test_per_tool_allowed_hook_gates_on_arguments(ran):
    @tool(allowed=lambda arguments, context: arguments["path"].startswith("/safe/"))
    def read_file(path: str) -> str:
        """Read a file."""
        ran.append(("read_file", path))
        return "contents"

    agent = Agent(FakeLLM([]), tools=[read_file])
    assert agent._execute(_call("read_file", path="/safe/notes.txt")) == "contents"
    assert "refused this call" in agent._execute(_call("read_file", path="/etc/shadow"))
    assert ran == [("read_file", "/safe/notes.txt")]


def test_custom_policy_can_deny_on_side_effects(policy_tools, ran):
    class ReadOnlyWindow(unchained.ToolPolicy):
        def authorize(self, tool_obj, arguments, context):
            if tool_obj.side_effects:
                raise unchained.ToolAuthorizationError("writes are frozen right now")
            super().authorize(tool_obj, arguments, context)

    agent = Agent(FakeLLM([]), tools=policy_tools, policy=ReadOnlyWindow())
    assert agent._execute(_call("search", query="q")) == "results for q"
    assert "frozen" in agent._execute(_call("write_row", table="t", value="v"))
    assert ran == [("search", "q")]


# --- approval --------------------------------------------------------------
def test_approval_required_tool_runs_only_when_approved(policy_tools, ran):
    requests = []

    def approve(request):
        requests.append(request)
        return request["arguments"]["table"] == "scratch"

    agent = Agent(
        FakeLLM([]),
        tools=policy_tools,
        policy=unchained.PermissionPolicy(granted={"db:admin"}),
        approve=approve,
    )
    assert agent._execute(_call("drop_table", table="scratch")) == "dropped"
    assert "not approved" in agent._execute(_call("drop_table", table="production"))
    assert ran == [("drop_table", "scratch")]

    # The approver is shown what it needs to decide, including the real args.
    assert requests[0]["tool"] == "drop_table"
    assert requests[0]["arguments"] == {"table": "scratch"}
    assert requests[0]["permissions"] == ["db:admin"]
    assert requests[0]["side_effects"] is True


def test_approval_required_without_an_approver_fails_closed(policy_tools, ran):
    # An unanswerable question is a refusal. A tool marked requires_approval
    # must not run just because nobody wired up an approver.
    agent = Agent(
        FakeLLM([]), tools=policy_tools, policy=unchained.PermissionPolicy(granted={"db:admin"})
    )
    observation = agent._execute(_call("drop_table", table="t"))
    assert "requires approval" in observation
    assert ran == []


def test_an_approver_that_raises_is_a_refusal(policy_tools, ran):
    def approve(request):
        raise RuntimeError("approval backend is down")

    agent = Agent(
        FakeLLM([]),
        tools=policy_tools,
        policy=unchained.PermissionPolicy(granted={"db:admin"}),
        approve=approve,
    )
    assert "could not be approved" in agent._execute(_call("drop_table", table="t"))
    assert ran == []


def test_policy_can_escalate_a_granted_permission_to_approval(ran):
    @tool(permissions={"db:write"}, side_effects=True)
    def write_row(table: str) -> str:
        """Insert a row."""
        ran.append(("write_row", table))
        return "written"

    policy = unchained.PermissionPolicy(granted={"db:write"}, approval_for={"db:write"})
    denied = Agent(FakeLLM([]), tools=[write_row], policy=policy)
    assert "requires approval" in denied._execute(_call("write_row", table="t"))
    assert ran == []

    allowed = Agent(FakeLLM([]), tools=[write_row], policy=policy, approve=lambda request: True)
    assert allowed._execute(_call("write_row", table="t")) == "written"
    assert ran == [("write_row", "t")]


def test_approval_callback_is_serialised_across_concurrent_tool_calls(ran):
    # A turn's calls run concurrently, but a CLI prompt or modal dialog must
    # not be re-entered from several workers at once.
    overlaps = []
    inside = {"n": 0}
    guard = threading.Lock()

    @tool(requires_approval=True)
    def confirmable(x: int) -> str:
        """Needs confirmation."""
        return f"ok{x}"

    def approve(request):
        with guard:
            inside["n"] += 1
            overlaps.append(inside["n"])
        time.sleep(0.005)
        with guard:
            inside["n"] -= 1
        return True

    agent = Agent(FakeLLM([]), tools=[confirmable], approve=approve, max_tool_workers=8)
    agent._execute_calls([_call("confirmable", x=i) for i in range(12)])
    assert max(overlaps) == 1, "approval callback was entered concurrently"


# --- the model cannot reach an unauthorized tool ---------------------------
def test_model_requesting_an_unknown_tool_never_reaches_a_function(policy_tools, ran):
    agent = Agent(FakeLLM([]), tools=policy_tools)
    assert "unknown tool" in agent._execute(_call("rm_rf", path="/"))
    assert ran == []


def test_model_cannot_reach_a_tool_the_agent_was_not_given(ran):
    @tool(permissions={"db:admin"})
    def privileged(x: int) -> str:
        """Not handed to the agent."""
        ran.append(("privileged", x))
        return "ran"

    # The tool exists in the process, but this agent was never given it.
    agent = Agent(FakeLLM([]), tools=[], policy=unchained.PermissionPolicy(granted={"db:admin"}))
    assert "unknown tool" in agent._execute(_call("privileged", x=1))
    assert ran == []


def test_every_model_driven_path_goes_through_the_policy(policy_tools, ran):
    # Both run() and stream() must funnel tool calls through _execute. If a
    # future refactor adds a path that calls tool.run() directly, this fails.
    seen = []

    class Recording(unchained.ToolPolicy):
        def authorize(self, tool_obj, arguments, context):
            seen.append(tool_obj.name)
            raise unchained.ToolAuthorizationError("nothing is allowed here")

    call = {"name": "search", "arguments": {"query": "q"}, "id": "x"}
    agent = Agent(
        MockLLM(script=[{"content": "", "tool_calls": [call]}, {"content": "done"}]),
        tools=policy_tools,
        policy=Recording(),
    )
    agent.run("go")

    streamer = Agent(
        MockLLM(script=[{"content": "", "tool_calls": [call]}, {"content": "done"}]),
        tools=policy_tools,
        policy=Recording(),
    )
    list(streamer.stream("go"))

    assert seen == ["search", "search"]  # once per path
    assert ran == []


def test_a_broken_policy_fails_closed(policy_tools, ran):
    # A policy that raises something unexpected must deny, not fall through
    # to execution and not crash the run.
    class Broken(unchained.ToolPolicy):
        def authorize(self, tool_obj, arguments, context):
            raise ValueError("bug in my own policy")

    agent = Agent(FakeLLM([]), tools=policy_tools, policy=Broken())
    assert "not authorized" in agent._execute(_call("search", query="q"))
    assert ran == []


# --- argument validation ---------------------------------------------------
def test_invalid_arguments_are_rejected_before_execution(policy_tools, ran):
    agent = Agent(FakeLLM([]), tools=policy_tools)
    assert "unexpected argument" in agent._execute(_call("search", query="q", sneaky=1))
    assert "missing required argument" in agent._execute({"name": "search", "arguments": {}})
    assert "expects an object" in agent._execute({"name": "search", "arguments": "not-a-dict"})
    assert ran == []


def test_validate_arguments_rules():
    @tool
    def sample(a: str, b: int = 2) -> str:
        """Sample."""
        return a

    assert sample.validate_arguments({"a": "x"}) == {"a": "x"}
    assert sample.validate_arguments({"a": "x", "b": 5}) == {"a": "x", "b": 5}
    with pytest.raises(unchained.ToolArgumentValidationError):
        sample.validate_arguments({"b": 5})  # missing required 'a'
    with pytest.raises(unchained.ToolArgumentValidationError):
        sample.validate_arguments({"a": "x", "c": 1})  # unknown name
    with pytest.raises(unchained.ToolArgumentValidationError):
        sample.validate_arguments({"a": "x", 1: "y"})  # non-string name
    with pytest.raises(unchained.ToolArgumentValidationError):
        sample.validate_arguments(["a"])  # not a mapping


def test_validate_arguments_treats_a_missing_arguments_dict_as_empty():
    @tool
    def no_args() -> str:
        """Takes nothing."""
        return "ok"

    assert no_args.validate_arguments(None) == {}
    assert no_args.validate_arguments({}) == {}


def test_validate_arguments_allows_unknown_names_for_kwargs_tools():
    @tool
    def flexible(a: str, **rest: str) -> str:
        """Accepts extras by design."""
        return a

    assert flexible.accepts_kwargs is True
    assert flexible.validate_arguments({"a": "x", "anything": 1}) == {"a": "x", "anything": 1}
    with pytest.raises(unchained.ToolArgumentValidationError):
        flexible.validate_arguments({"anything": 1})  # 'a' is still required


def test_argument_validation_does_not_coerce_types():
    # Deliberate: silently turning "3" into 3 would hide real model errors.
    @tool
    def counted(n: int) -> str:
        """Takes an int."""
        return str(n)

    assert counted.validate_arguments({"n": "3"}) == {"n": "3"}


# --- exceptions ------------------------------------------------------------
def test_policy_exceptions_are_exported_and_catchable():
    for name in (
        "ToolPolicy",
        "PermissionPolicy",
        "ToolAuthorizationError",
        "ToolApprovalRequired",
        "ToolArgumentValidationError",
    ):
        assert name in unchained.__all__, name
    assert issubclass(unchained.ToolAuthorizationError, RuntimeError)
    assert issubclass(unchained.ToolApprovalRequired, RuntimeError)
    assert issubclass(unchained.ToolArgumentValidationError, RuntimeError)


def test_policy_hooks_receive_tool_arguments_and_context():
    captured = {}

    class Inspecting(unchained.ToolPolicy):
        def authorize(self, tool_obj, arguments, context):
            captured["tool"] = tool_obj.name
            captured["arguments"] = arguments
            captured["context"] = context

    @tool(permissions={"x"})
    def probe(value: str) -> str:
        """Probe."""
        return value

    agent = Agent(FakeLLM([]), tools=[probe], name="inspector", policy=Inspecting())
    agent._execute({"name": "probe", "arguments": {"value": "v"}, "id": "call-9"})

    assert captured["tool"] == "probe"
    assert captured["arguments"] == {"value": "v"}
    assert captured["context"] == {"agent": "inspector", "tool": "probe", "call_id": "call-9"}


# --- audit -----------------------------------------------------------------
def test_every_decision_is_audited(policy_tools):
    class Sink(unchained.Callback):
        def __init__(self):
            self.events = []

        def on_tool_audit(self, event):
            self.events.append(event)

    sink = Sink()
    agent = Agent(
        FakeLLM([]),
        tools=policy_tools,
        policy=unchained.PermissionPolicy(granted={"db:read"}),
        callbacks=[sink],
        name="auditor",
    )
    agent._execute(_call("search", query="q"))
    agent._execute(_call("write_row", table="t", value="v"))
    agent._execute(_call("ghost"))
    agent._execute(_call("search", query="q", bad=1))
    agent._execute(_call("drop_table", table="t"))

    decisions = [(e["tool"], e["decision"]) for e in sink.events]
    assert decisions == [
        ("search", "allowed"),
        ("write_row", "denied"),
        ("ghost", "unknown_tool"),
        ("search", "invalid_arguments"),
        ("drop_table", "denied"),  # ungranted db:admin is refused before approval
    ]
    allowed_event = sink.events[0]
    assert allowed_event["agent"] == "auditor"
    assert allowed_event["side_effects"] is False
    assert sink.events[1]["permissions"] == ["db:write"]
    assert sink.events[1]["side_effects"] is True
    assert "db:write" in sink.events[1]["reason"]


def test_audit_records_an_approved_call_distinctly(ran):
    class Sink(unchained.Callback):
        def __init__(self):
            self.events = []

        def on_tool_audit(self, event):
            self.events.append(event)

    @tool(requires_approval=True)
    def confirmable(x: int) -> str:
        """Needs confirmation."""
        ran.append(("confirmable", x))
        return "ok"

    sink = Sink()
    agent = Agent(FakeLLM([]), tools=[confirmable], approve=lambda r: True, callbacks=[sink])
    agent._execute(_call("confirmable", x=1))
    assert [e["decision"] for e in sink.events] == ["approved"]


def test_a_failing_audit_callback_never_breaks_the_run(policy_tools, ran):
    class Broken(unchained.Callback):
        def on_tool_audit(self, event):
            raise RuntimeError("audit sink is down")

    agent = Agent(FakeLLM([]), tools=policy_tools, callbacks=[Broken()])
    assert agent._execute(_call("search", query="q")) == "results for q"
    assert ran == [("search", "q")]


def test_logging_callback_implements_the_audit_hook():
    assert unchained.LoggingCallback().on_tool_audit({"tool": "t", "decision": "allowed"}) is None


# --- backwards compatibility ----------------------------------------------
def test_bare_tool_decorator_is_unchanged():
    @tool
    def legacy(a: int, b: int = 1) -> int:
        """Add."""
        return a + b

    assert isinstance(legacy, Tool)
    assert legacy(2, 3) == 5  # still directly callable
    assert legacy.run({"a": 2}) == 3  # run() is not policed
    assert legacy.schema["function"]["name"] == "legacy"
    # Metadata defaults leave the tool exactly as permissive as before.
    assert legacy.permissions == frozenset()
    assert legacy.requires_approval is False
    assert legacy.side_effects is False
    assert legacy.allowed is None


def test_tool_called_as_a_plain_function_still_works():
    def plain(a: str) -> str:
        """Plain."""
        return a

    assert Tool(plain).run({"a": "x"}) == "x"


def test_agent_without_a_policy_gets_the_permissive_default(policy_tools, ran):
    # No policy= argument: everything the agent was given still runs, exactly
    # as it did before this layer existed.
    agent = Agent(FakeLLM([]), tools=policy_tools)
    assert isinstance(agent.policy, unchained.ToolPolicy)
    assert agent.approve is None
    assert agent._execute(_call("search", query="q")) == "results for q"
    assert agent._execute(_call("write_row", table="t", value="v")) == "written"
    assert ran == [("search", "q"), ("write_row", "t")]


def test_default_policy_still_honours_tool_metadata(ran):
    # Requirement: metadata must not be decorative. Even with no explicit
    # policy, a tool marked requires_approval is gated.
    @tool(requires_approval=True)
    def dangerous(x: int) -> str:
        """Marked, but no policy configured."""
        ran.append(("dangerous", x))
        return "ran"

    agent = Agent(FakeLLM([]), tools=[dangerous])
    assert "requires approval" in agent._execute(_call("dangerous", x=1))
    assert ran == []


def test_existing_tool_error_handling_is_unchanged():
    @tool
    def boom(x: int) -> int:
        """Always explodes."""
        raise ValueError("nope")

    agent = Agent(FakeLLM([]), tools=[boom])
    assert "Error executing 'boom'" in agent._execute(_call("boom", x=1))
