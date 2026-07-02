"""Unchained - a single-file agentic AI framework.

Tool calling, memory, RAG, multi-agent orchestration and structured output.
Two dependencies (requests + pydantic). Provider-agnostic: OpenAI, Anthropic
or local Ollama. No metaclasses, no runtime patching, no hidden state.

    from unchained import LLM, Agent, tool

    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    print(Agent(LLM(provider="ollama"), tools=[add]).run("What is 21 + 21?"))
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import requests

try:
    from pydantic import BaseModel, ValidationError
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]
    ValidationError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger("unchained")
logger.addHandler(logging.NullHandler())

__version__ = "0.3.0"
__all__ = [
    "tool",
    "Tool",
    "LLM",
    "MockLLM",
    "Memory",
    "RAG",
    "Agent",
    "Router",
    "Callback",
    "LoggingCallback",
]

# HTTP statuses worth retrying: rate limiting plus transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class _RetryableStatus(Exception):
    """Internal signal that the server returned a retryable HTTP status."""

    def __init__(self, response: requests.Response):
        self.response = response
        super().__init__(f"retryable status {response.status_code}")


# --- 1. Tool system --------------------------------------------------------
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    """Wrap a function as an LLM-callable tool with an auto-generated schema.

    Type hints become JSON-Schema types, the docstring becomes the description,
    and parameters without a default are marked required.
    """

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.name = func.__name__
        self.description = (inspect.getdoc(func) or "").strip()
        self.schema = self._build_schema(func)

    def _build_schema(self, func: Callable[..., Any]) -> Dict[str, Any]:
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:  # pragma: no cover
            hints = {}
        properties, required = {}, []
        for name, param in sig.parameters.items():
            if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            properties[name] = self._type_to_schema(hints.get(name, str))
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    def _type_to_schema(self, hint: Any) -> Dict[str, Any]:
        origin = get_origin(hint)
        if origin in (list, List):
            args = get_args(hint)
            item = self._type_to_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item}
        if origin in (dict, Dict):
            return {"type": "object"}
        if origin is Union:  # Optional[X] / Union[X, None] -> first real type
            real = [a for a in get_args(hint) if a is not type(None)]
            if real:
                return self._type_to_schema(real[0])
        return {"type": _PY_TO_JSON.get(hint, "string")}

    def run(self, arguments: Dict[str, Any]) -> Any:
        """Call the wrapped function with a dict of keyword arguments."""
        return self.func(**(arguments or {}))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


def tool(func: Callable[..., Any]) -> Tool:
    """Decorator: turn any function into a Tool with an auto-generated schema."""
    return Tool(func)


# --- 2. LLM backend (unified across providers) -----------------------------
_PROVIDER_DEFAULTS = {
    "openai": ("gpt-4o-mini", "https://api.openai.com"),
    "anthropic": ("claude-3-5-sonnet-20241022", "https://api.anthropic.com"),
    "ollama": ("llama3.1", "http://localhost:11434"),
}
_PROVIDER_ENV_KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class LLM:
    """One chat interface for OpenAI, Anthropic and Ollama.

    Every provider returns the same normalised dict::

        {"content": str, "tool_calls": [{"name", "arguments", "id"}], "usage": dict}
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 2,
        backoff: float = 0.5,
        cache: bool = False,
    ):
        self.provider = provider.lower().strip()
        if self.provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose from {list(_PROVIDER_DEFAULTS)}."
            )
        default_model, default_url = _PROVIDER_DEFAULTS[self.provider]
        self.model = model or default_model
        self.base_url = (base_url or default_url).rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.max_retries, self.backoff = max_retries, backoff
        self.cache: Optional[Dict[str, Dict[str, Any]]] = {} if cache else None
        env_key = _PROVIDER_ENV_KEY.get(self.provider)
        self.api_key = api_key or (os.getenv(env_key) if env_key else None)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Send a chat request and return the normalised response dict.

        With ``cache=True`` identical requests are served from an in-memory
        cache instead of hitting the provider again.
        """
        if self.cache is not None:
            key = self._cache_key(messages, tools, response_format)
            if key in self.cache:
                return self.cache[key]
            result = self._dispatch(messages, tools, response_format)
            self.cache[key] = result
            return result
        return self._dispatch(messages, tools, response_format)

    def _dispatch(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]],
        response_format: Optional[Any],
    ) -> Dict[str, Any]:
        return {"openai": self._openai, "anthropic": self._anthropic, "ollama": self._ollama}[
            self.provider
        ](messages, tools, response_format)

    def _cache_key(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]],
        response_format: Optional[Any],
    ) -> str:
        return json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": messages,
                "tools": sorted(t.name for t in tools) if tools else None,
                "format": getattr(response_format, "__name__", None),
            },
            sort_keys=True,
            default=str,
        )

    # -- OpenAI --
    def _openai(self, messages, tools, response_format):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [t.schema for t in tools]
        if response_format is not None:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/v1/chat/completions", headers=headers, json=payload)
        msg = data["choices"][0]["message"]
        tool_calls = [
            {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": self._loads(tc.get("function", {}).get("arguments")),
                "id": tc.get("id"),
            }
            for tc in msg.get("tool_calls") or []
        ]
        return {
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
            "usage": self._normalize_usage(data.get("usage")),
        }

    # -- Anthropic --
    def _anthropic(self, messages, tools, response_format):
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        )
        convo = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": self.model,
            "messages": self._to_anthropic_messages(convo),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._anthropic_tool(t) for t in tools]
        data = self._post("/v1/messages", headers=headers, json=payload)
        content, tool_calls = "", []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}) or {},
                        "id": block.get("id"),
                    }
                )
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": self._normalize_usage(data.get("usage")),
        }

    @staticmethod
    def _anthropic_tool(t: Tool) -> Dict[str, Any]:
        fn = t.schema["function"]
        return {
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        }

    def _to_anthropic_messages(self, messages):
        out = []
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []) or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or tc["name"],
                            "name": tc["name"],
                            "input": tc.get("arguments", {}),
                        }
                    )
                out.append(
                    {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
                )
            elif role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id") or m.get("name"),
                                "content": str(m.get("content", "")),
                            }
                        ],
                    }
                )
            else:
                out.append({"role": "user", "content": m.get("content", "")})
        return out

    # -- Ollama --
    def _ollama(self, messages, tools, response_format):
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = [t.schema for t in tools]
        if response_format is not None:
            payload["format"] = "json"
        data = self._post("/api/chat", json=payload)
        msg = data.get("message", {})
        tool_calls = [
            {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": self._loads(tc.get("function", {}).get("arguments")),
                "id": None,
            }
            for tc in msg.get("tool_calls") or []
        ]
        return {
            "content": msg.get("content", ""),
            "tool_calls": tool_calls,
            "usage": self._normalize_usage(data),
        }

    @staticmethod
    def _to_ollama_messages(messages):
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content", ""),
                        "tool_calls": [
                            {"function": {"name": tc["name"], "arguments": tc.get("arguments", {})}}
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif m.get("role") == "tool":
                out.append({"role": "tool", "content": str(m.get("content", ""))})
            else:
                out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        return out

    # -- OpenAI message shaping --
    @staticmethod
    def _to_openai_messages(messages):
        out = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.get("id") or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for i, tc in enumerate(m["tool_calls"])
                        ],
                    }
                )
            elif m.get("role") == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id") or m.get("name"),
                        "content": str(m.get("content", "")),
                    }
                )
            else:
                out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        return out

    # -- streaming --
    def stream(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        """Yield text chunks for a plain assistant reply (no tool calling)."""
        dispatch = {
            "openai": self._stream_openai,
            "anthropic": self._stream_anthropic,
            "ollama": self._stream_ollama,
        }
        yield from dispatch[self.provider](messages)

    def _stream_openai(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature,
            "stream": True,
        }
        resp = self._request("/v1/chat/completions", headers=headers, json=payload, stream=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta

    def _stream_anthropic(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        )
        convo = [m for m in messages if m.get("role") != "system"]
        payload = {
            "model": self.model,
            "messages": self._to_anthropic_messages(convo),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        resp = self._request("/v1/messages", headers=headers, json=payload, stream=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                text = event.get("delta", {}).get("text")
                if text:
                    yield text

    def _stream_ollama(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        resp = self._request("/api/chat", json=payload, stream=True)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = chunk.get("message", {}).get("content")
            if text:
                yield text
            if chunk.get("done"):
                break

    # -- HTTP with retry/backoff --
    def _post(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        return self._request(path, **kwargs).json()

    def _request(self, path: str, **kwargs: Any) -> requests.Response:
        """POST with retries on connection errors and 429/5xx responses."""
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, timeout=self.timeout, **kwargs)
                if resp.status_code in _RETRYABLE_STATUS:
                    raise _RetryableStatus(resp)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout, _RetryableStatus) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay(exc, attempt)
                logger.warning(
                    "request to %s failed (%s); retry %d/%d in %.2fs",
                    path,
                    exc,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
        if isinstance(last_error, _RetryableStatus):
            last_error.response.raise_for_status()
        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without an error")  # pragma: no cover

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, _RetryableStatus):
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and str(retry_after).isdigit():
                return float(retry_after)
        return self.backoff * (2**attempt) + random.uniform(0, self.backoff)

    @staticmethod
    def _normalize_usage(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Map any provider's usage shape to prompt/completion/total tokens."""
        raw = raw or {}
        prompt = int(
            raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("prompt_eval_count") or 0
        )
        completion = int(
            raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("eval_count") or 0
        )
        total = int(raw.get("total_tokens") or (prompt + completion))
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    @staticmethod
    def _loads(raw: Any) -> Dict[str, Any]:
        """Best-effort parse of tool-call arguments (str or dict)."""
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}


class MockLLM(LLM):
    """A no-network stand-in for :class:`LLM` - ideal for demos and tests.

    Configure its behaviour one of three ways:

    * ``reply="..."``  - return the same text every turn.
    * ``script=[...]`` - return queued responses in order. Each item is either
      a string (used as ``content``) or a full response dict, which may include
      ``tool_calls`` to drive the agent loop.
    * ``handler=fn``   - ``fn(messages, tools) -> str | dict`` for custom logic.

    No API key, no server, fully deterministic.
    """

    def __init__(
        self,
        reply: str = "This is a mock response.",
        script: Optional[List[Any]] = None,
        handler: Optional[Callable[[List[Dict[str, Any]], Optional[List[Tool]]], Any]] = None,
        model: str = "mock",
    ):
        super().__init__(provider="ollama", model=model, api_key="mock")
        self.provider = "mock"
        self.reply = reply
        self.script = list(script) if script is not None else None
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {"messages": messages, "tools": tools, "response_format": response_format}
        )
        return self._normalize(self._next(messages, tools))

    def stream(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        text = self._normalize(self._next(messages, None))["content"]
        yield from re.findall(r"\S+\s*", text)

    def _next(self, messages: List[Dict[str, Any]], tools: Optional[List[Tool]]) -> Any:
        if self.handler is not None:
            return self.handler(messages, tools)
        if self.script:
            return self.script.pop(0)
        return self.reply

    @staticmethod
    def _normalize(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return {
                "content": raw.get("content", ""),
                "tool_calls": raw.get("tool_calls", []),
                "usage": raw.get("usage", {}),
            }
        return {"content": str(raw), "tool_calls": [], "usage": {}}


# --- 3. Memory (sliding window + compression) ------------------------------
class Memory:
    """Fixed sliding-window conversation memory.

    On overflow the oldest half is compressed into a running ``summary`` - via
    the LLM if one is supplied, otherwise by truncating each message to 100 chars.
    """

    def __init__(self, max_messages: int = 20, llm: Optional[LLM] = None):
        self.max_messages, self.llm = max_messages, llm
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""

    def add(self, role: str, content: Any, **extra: Any) -> None:
        self.messages.append({"role": role, "content": content, **extra})
        if len(self.messages) > self.max_messages:
            self._compress()

    def get(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def _compress(self) -> None:
        keep = max(1, self.max_messages // 2)
        overflow, self.messages = self.messages[:-keep], self.messages[-keep:]
        if not overflow:
            return
        rendered = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in overflow)
        if self.llm is not None:
            prompt = [
                {
                    "role": "system",
                    "content": "Summarise the conversation so far, "
                    "preserving key facts, decisions and open questions. Be concise.",
                },
                {
                    "role": "user",
                    "content": (f"{self.summary}\n\n" if self.summary else "") + rendered,
                },
            ]
            try:
                self.summary = self.llm.chat(prompt)["content"].strip()
                return
            except Exception:  # pragma: no cover
                pass
        truncated = "\n".join(
            f"{m.get('role')}: {str(m.get('content', ''))[:100]}" for m in overflow
        )
        self.summary = (f"{self.summary}\n{truncated}" if self.summary else truncated).strip()


# --- 4. RAG (TF-IDF + smoothed IDF + cosine similarity) --------------------
class RAG:
    """Tiny in-memory retriever with cosine-similarity search.

    By default it uses TF-IDF with sklearn-style smoothed IDF -
    ``idf(t) = log((1 + N) / (1 + df(t))) + 1`` - which keeps scores positive
    even for a 2-3 document corpus and needs no dependencies.

    Pass ``embed_fn`` (``list[str] -> list[list[float]]``) to switch to dense
    embeddings instead - e.g. an OpenAI or sentence-transformers model. The
    search interface is identical either way.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.embed_fn = embed_fn
        self.docs: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._tf: List[Counter] = []
        self._idf: Dict[str, float] = {}
        self._vectors: List[Dict[str, float]] = []
        self._norms: List[float] = []
        self._embeddings: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.add_many([text], [metadata or {}])

    def add_many(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        metadatas = metadatas or [{} for _ in texts]
        for text, meta in zip(texts, metadatas):
            self.docs.append(text)
            self.metadata.append(meta or {})
            self._tf.append(Counter(self._tokenize(text)))
        if self.embed_fn is not None:
            self._embeddings.extend(self.embed_fn(list(texts)))
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        n = len(self.docs)
        if n == 0:
            self._idf, self._vectors, self._norms = {}, [], []
            return
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self._vectors, self._norms = [], []
        for tf in self._tf:
            total = sum(tf.values()) or 1
            vec = {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}
            self._vectors.append(vec)
            self._norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        scores = self._embedding_scores(query) if self.embed_fn else self._tfidf_scores(query)
        results: List[Dict[str, Any]] = [
            {"text": self.docs[i], "score": score, "metadata": self.metadata[i]}
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:top_k]

    def _tfidf_scores(self, query: str) -> List[float]:
        q_tf = Counter(self._tokenize(query))
        total = sum(q_tf.values()) or 1
        q_vec = {t: (c / total) * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scores = []
        for i, vec in enumerate(self._vectors):
            dot = sum(w * vec.get(t, 0.0) for t, w in q_vec.items())
            scores.append(dot / (q_norm * self._norms[i]))
        return scores

    def _embedding_scores(self, query: str) -> List[float]:
        assert self.embed_fn is not None
        q = self.embed_fn([query])[0]
        return [self._cosine(q, emb) for emb in self._embeddings]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def __len__(self) -> int:
        return len(self.docs)


# --- 5. Observability (callbacks) ------------------------------------------
class Callback:
    """Hook into the agent loop. Subclass and override the methods you need.

    Every method is a no-op by default. Callbacks must not raise: the agent
    swallows and logs any callback error so instrumentation never breaks a run.
    """

    def on_iteration(self, index: int) -> None:
        """Called at the start of each ReAct iteration (0-based)."""

    def on_llm_call(self, messages: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        """Called after every LLM response."""

    def on_tool_call(self, name: str, arguments: Dict[str, Any], result: str) -> None:
        """Called after each tool executes."""

    def on_finish(self, answer: Any) -> None:
        """Called once with the final answer."""


class LoggingCallback(Callback):
    """A ready-made tracer that logs each step of the agent loop."""

    def __init__(self, logger_: Optional[logging.Logger] = None):
        self.log = logger_ or logger

    def on_iteration(self, index: int) -> None:
        self.log.info("iteration %d", index)

    def on_llm_call(self, messages: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        self.log.info(
            "llm: %d msg(s) -> %d tool call(s), %d chars",
            len(messages),
            len(response.get("tool_calls", [])),
            len(response.get("content", "")),
        )

    def on_tool_call(self, name: str, arguments: Dict[str, Any], result: str) -> None:
        self.log.info("tool: %s(%s) -> %s", name, arguments, str(result)[:120])

    def on_finish(self, answer: Any) -> None:
        self.log.info("finished (%d chars)", len(str(answer)))


# --- 6. Agent core (ReAct loop) --------------------------------------------
class Agent:
    """A ReAct agent: think (LLM) -> act (tool) -> observe -> repeat.

    Compose it with tools, memory and RAG. ``run`` optionally takes a Pydantic
    model for validated structured output; ``stream`` yields the answer token
    by token. Attach ``callbacks`` for tracing and read ``usage`` for token
    accounting.
    """

    def __init__(
        self,
        llm: LLM,
        name: str = "agent",
        description: str = "",
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        rag: Optional[RAG] = None,
        max_iterations: int = 6,
        callbacks: Optional[List[Callback]] = None,
        structured_retries: int = 1,
    ):
        self.llm = llm
        self.name = name
        self.description = description or system_prompt
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.memory = memory or Memory()
        self.rag = rag
        self.max_iterations = max_iterations
        self.callbacks = list(callbacks or [])
        self.structured_retries = structured_retries
        self.usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def run(self, user_input: str, response_format: Optional[Type[BaseModel]] = None) -> Any:
        self.memory.add("user", self._augment_with_rag(user_input))
        tool_list = list(self.tools.values())
        answer: Optional[str] = None
        for i in range(self.max_iterations):
            self._emit("on_iteration", i)
            # JSON mode is only requested when no tools are in play; combining
            # tool-calling with JSON mode is unreliable across providers.
            fmt = response_format if not tool_list else None
            result = self._chat(
                self._build_messages(fmt), tools=tool_list or None, response_format=fmt
            )
            if not result["tool_calls"]:
                answer = result["content"]
                self.memory.add("assistant", answer)
                break
            self.memory.add("assistant", result["content"], tool_calls=result["tool_calls"])
            for call in result["tool_calls"]:
                observation = self._execute(call)
                self._emit("on_tool_call", call["name"], call.get("arguments", {}), observation)
                self.memory.add(
                    "tool",
                    observation,
                    tool_call_id=call.get("id") or call["name"],
                    name=call["name"],
                )
        if answer is None:  # exhausted iterations - force a final answer
            answer = self._chat(
                self._build_messages(response_format), response_format=response_format
            )["content"]
            self.memory.add("assistant", answer)
        if response_format is not None:
            if tool_list:  # dedicated formatting pass so output matches the schema
                answer = self._chat(
                    self._build_messages(response_format), response_format=response_format
                )["content"]
            parsed = self._parse_structured(answer, response_format)
            self._emit("on_finish", parsed)
            return parsed
        self._emit("on_finish", answer)
        return answer

    def stream(self, user_input: str) -> Iterator[str]:
        """Stream the final answer token by token.

        Any tool calls are resolved first (non-streaming); the final assistant
        reply is then streamed. With no tools, the reply is streamed directly.
        """
        self.memory.add("user", self._augment_with_rag(user_input))
        tool_list = list(self.tools.values())
        if tool_list:
            for i in range(self.max_iterations):
                self._emit("on_iteration", i)
                result = self._chat(self._build_messages(None), tools=tool_list)
                if not result["tool_calls"]:
                    break
                self.memory.add("assistant", result["content"], tool_calls=result["tool_calls"])
                for call in result["tool_calls"]:
                    observation = self._execute(call)
                    self._emit("on_tool_call", call["name"], call.get("arguments", {}), observation)
                    self.memory.add(
                        "tool",
                        observation,
                        tool_call_id=call.get("id") or call["name"],
                        name=call["name"],
                    )
        chunks: List[str] = []
        for chunk in self.llm.stream(self._build_messages(None)):
            chunks.append(chunk)
            yield chunk
        answer = "".join(chunks)
        self.memory.add("assistant", answer)
        self._emit("on_finish", answer)

    # -- instrumentation helpers --
    def _chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        result = self.llm.chat(messages, tools=tools, response_format=response_format)
        self._track_usage(result.get("usage"))
        self._emit("on_llm_call", messages, result)
        return result

    def _track_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        for key in self.usage:
            self.usage[key] += int(usage.get(key, 0) or 0)

    def _emit(self, event: str, *args: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception:  # instrumentation must never break the run
                logger.exception("callback %s failed", event)

    def _augment_with_rag(self, user_input: str) -> str:
        if not self.rag:
            return user_input
        hits = self.rag.search(user_input)
        if not hits:
            return user_input
        context = "\n\n".join(f"[score={h['score']:.2f}] {h['text']}" for h in hits)
        return (
            f"Use the following context to answer.\n\nContext:\n{context}\n\nQuestion: {user_input}"
        )

    def _build_messages(self, schema: Optional[Type[BaseModel]]) -> List[Dict[str, Any]]:
        system = self.system_prompt
        if self.memory.summary:
            system += f"\n\nConversation summary so far:\n{self.memory.summary}"
        if schema is not None:
            system += (
                "\n\nRespond with a single JSON object matching this schema "
                f"(no prose, no code fences):\n{json.dumps(self._json_schema(schema))}"
            )
        return [{"role": "system", "content": system}] + self.memory.get()

    def _execute(self, call: Dict[str, Any]) -> str:
        tool_obj = self.tools.get(call["name"])
        if tool_obj is None:
            return f"Error: unknown tool '{call['name']}'."
        try:
            return str(tool_obj.run(call.get("arguments", {})))
        except Exception as exc:  # a tool must never crash the loop
            return f"Error executing '{call['name']}': {exc}"

    @staticmethod
    def _json_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()  # pydantic v2
        if hasattr(schema, "schema"):
            return schema.schema()  # pydantic v1
        return {}

    def _parse_structured(self, content: str, schema: Type[BaseModel]):
        """Validate content against the schema, repairing via the LLM on failure."""
        data = self._loads_object(content)
        for attempt in range(self.structured_retries + 1):
            try:
                return schema(**data)
            except ValidationError as exc:
                if attempt >= self.structured_retries:
                    raise
                logger.warning(
                    "structured output failed validation; repair attempt %d", attempt + 1
                )
                content = self._chat(
                    [
                        {
                            "role": "system",
                            "content": "You fix JSON so it matches the given schema. "
                            "Return only the corrected JSON object.",
                        },
                        {
                            "role": "user",
                            "content": f"Schema:\n{json.dumps(self._json_schema(schema))}\n\n"
                            f"Invalid JSON:\n{content}\n\nValidation error:\n{exc}",
                        },
                    ],
                    response_format=schema,
                )["content"]
                data = self._loads_object(content)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _loads_object(content: str) -> Dict[str, Any]:
        content = (content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}


# --- 7. Router (multi-agent orchestration) ---------------------------------
class Router:
    """Coordinate several agents: route to one, run all, or run all and fuse."""

    def __init__(self, llm: LLM, agents: List[Agent], synthesizer: Optional[Agent] = None):
        if not agents:
            raise ValueError("Router needs at least one agent.")
        self.llm, self.agents, self.synthesizer = llm, agents, synthesizer

    def _descriptions(self) -> str:
        return "\n".join(f"- {a.name}: {a.description}" for a in self.agents)

    def route(self, query: str) -> Agent:
        """Ask the LLM which single agent fits best and return it."""
        prompt = [
            {
                "role": "system",
                "content": "You are a router. Pick the single best "
                "agent for the user's query. Reply with ONLY the agent name.",
            },
            {
                "role": "user",
                "content": f"Agents:\n{self._descriptions()}\n\nQuery: {query}\n\nBest agent:",
            },
        ]
        return self._match(self.llm.chat(prompt)["content"].strip().lower())

    def _match(self, choice: str) -> Agent:
        best, best_score = self.agents[0], -1
        for agent in self.agents:
            name = agent.name.lower()
            if name == choice:
                return agent
            if name in choice or choice in name:
                score = len(name)
            else:
                score = sum(1 for w in name.split() if w and w in choice)
            if score > best_score:
                best, best_score = agent, score
        return best

    def run(self, query: str) -> Any:
        return self.route(query).run(query)

    def run_all(self, query: str, parallel: bool = True) -> Dict[str, Any]:
        """Run every agent (in parallel by default) and collect their results."""
        results: Dict[str, Any] = {}
        if parallel and len(self.agents) > 1:
            with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
                futures = {pool.submit(a.run, query): a for a in self.agents}
                for future, agent in futures.items():
                    results[agent.name] = self._safe(future.result)
        else:
            for agent in self.agents:
                results[agent.name] = self._safe(partial(agent.run, query))
        return results

    @staticmethod
    def _safe(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # one failing agent shouldn't sink the rest
            return f"Error: {exc}"

    def synthesize(self, query: str, parallel: bool = True) -> Any:
        """Run all agents, then fuse their findings with the synthesizer."""
        results = self.run_all(query, parallel=parallel)
        combined = "\n\n".join(f"### {n}\n{r}" for n, r in results.items())
        if self.synthesizer is None:
            return combined
        return self.synthesizer.run(
            f"Original query: {query}\n\nFindings from specialist agents:\n{combined}"
            "\n\nSynthesize these into a single, well-reasoned final answer."
        )
