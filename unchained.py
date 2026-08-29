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

import asyncio
import copy
import enum
import inspect
import json
import logging
import math
import os
import random
import re
import threading
import time
import uuid
import warnings
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import requests

try:
    from pydantic import BaseModel, ValidationError

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]
    ValidationError = Exception  # type: ignore[assignment,misc]
    _HAS_PYDANTIC = False

# Runtime argument validation needs Pydantic v2's create_model/ConfigDict.
# v1 has create_model but no ConfigDict, so this import fails there and tools
# fall back to structural checks - see Tool.validate_arguments.
try:
    from pydantic import ConfigDict, create_model

    _HAS_PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment,misc]
    create_model = None  # type: ignore[assignment]
    _HAS_PYDANTIC_V2 = False

logger = logging.getLogger("unchained")
logger.addHandler(logging.NullHandler())

__version__ = "0.4.0"
__all__ = [
    "tool",
    "Tool",
    "LLM",
    "MockLLM",
    "Memory",
    "RAG",
    "Agent",
    "Session",
    "Router",
    "Callback",
    "LoggingCallback",
    "AgentEvent",
    "ToolPolicy",
    "PermissionPolicy",
    "RoutingError",
    "ToolAuthorizationError",
    "ToolApprovalRequired",
    "ToolArgumentValidationError",
    "ToolTimeoutError",
    "ToolOutputTruncated",
    "Budget",
    "RunState",
    "BudgetExceededError",
    "TokenBudgetExceeded",
    "ToolCallBudgetExceeded",
    "ToolOutputBudgetExceeded",
    "TimeBudgetExceeded",
    "CostBudgetExceeded",
]

# HTTP statuses worth retrying: rate limiting plus transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# What an LLM may keep in its response cache.
#   "none"       - cache nothing.
#   "final_only" - cache plain answers; never store a response that asks for
#                  tool calls. The default, because a cached tool call is a
#                  stored decision to act, replayed without the model being
#                  asked again.
#   "all"        - cache tool-call responses too. For read-only tool sets.
_CACHE_POLICIES = ("none", "final_only", "all")


class RoutingError(RuntimeError):
    """Raised when :meth:`Router.route` cannot identify exactly one agent.

    Routing is an authorization decision as much as a dispatch one: agents
    differ in the tools - and therefore the privileges - they carry. Guessing
    when the model's answer is empty, evasive or ambiguous would silently
    hand a query to an agent nobody chose, so ``Router`` fails closed instead.
    Pass ``Router(..., fallback=agent)`` to nominate an explicit destination
    for unroutable queries.
    """


class ToolAuthorizationError(RuntimeError):
    """Raised when a :class:`ToolPolicy` refuses a tool call outright."""


class ToolApprovalRequired(RuntimeError):
    """Raised when a tool call needs approval that was not granted.

    Also raised when a tool is marked ``requires_approval`` but the agent has
    no approval callback: an unanswerable question is a refusal, not a pass.
    """


class ToolArgumentValidationError(RuntimeError):
    """Raised when model-supplied arguments do not fit the tool's signature."""


class BudgetExceededError(RuntimeError):
    """Base class for every budget that can stop a run.

    Carries ``limit_name`` (which budget), ``limit`` and ``used``, so a
    handler can report the overrun without parsing the message. Catch this to
    catch them all, or a subclass to single one out.

    There is deliberately no iteration subclass: running out of iterations is
    the one budget that ends the turn gracefully rather than by raising - see
    :class:`Budget`.
    """

    def __init__(self, limit_name: str, limit: Any, used: Any, detail: str = ""):
        self.limit_name = limit_name
        self.limit = limit
        self.used = used
        message = f"budget '{limit_name}' exhausted: used {used} of {limit}"
        super().__init__(f"{message} - {detail}" if detail else message)


class TokenBudgetExceeded(BudgetExceededError):
    """Raised when a run reaches ``Budget.max_total_tokens``."""


class ToolCallBudgetExceeded(BudgetExceededError):
    """Raised when a run reaches ``Budget.max_tool_calls``."""


class ToolOutputBudgetExceeded(BudgetExceededError):
    """Raised when a run's total tool output passes ``Budget.max_tool_output``."""


class TimeBudgetExceeded(BudgetExceededError):
    """Raised when a run passes ``Budget.timeout`` seconds of wall clock."""


class CostBudgetExceeded(BudgetExceededError):
    """Raised when a run reaches ``Budget.max_cost``, or cannot be priced.

    Also raised when ``max_cost`` is set and a call uses a model absent from
    ``Budget.pricing``: a cap that cannot be computed cannot be honoured, and
    continuing would mean pretending to enforce it.
    """


class ToolOutputTruncated:
    """Records that a tool result was shortened before the model saw it.

    Deliberately **not** an exception: the tool succeeded, and the shortened
    result is still useful. It carries what was cut, so an application can log
    or alert on it, and renders as the note appended to the text the model
    reads - which is what stops the shortening from being silent.
    """

    def __init__(self, tool: str, original_size: int, limit: int, was_json: bool = False):
        self.tool = tool
        self.original_size = original_size
        self.limit = limit
        self.was_json = was_json

    @property
    def metadata(self) -> Dict[str, Any]:
        """A plain, JSON-serialisable record - safe for logs and memory extras."""
        return {
            "tool": self.tool,
            "truncated": True,
            "original_size": self.original_size,
            "limit": self.limit,
            "was_json": self.was_json,
        }

    def __str__(self) -> str:
        note = (
            f"\n\n[output truncated: {self.limit:,} of {self.original_size:,} characters "
            f"shown for tool '{self.tool}']"
        )
        if self.was_json:
            # Say so explicitly. A model handed a JSON fragment will otherwise
            # try to parse it, and a cut structure is not recoverable.
            note += (
                " The full result was valid JSON; this fragment is cut mid-structure "
                "and will not parse."
            )
        return note

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolOutputTruncated {self.tool} {self.limit}/{self.original_size}>"


class ToolTimeoutError(RuntimeError):
    """Raised when a tool call outlives its timeout.

    The timeout bounds how long the **agent** waits, which is not the same as
    stopping the tool. Python cannot cancel a thread that is already running:
    the call keeps going in the background until it finishes on its own. See
    :meth:`Agent._invoke` for what that means in practice.
    """


class _RetryableStatus(Exception):
    """Internal signal that the server returned a retryable HTTP status."""

    def __init__(self, response: requests.Response):
        self.response = response
        super().__init__(f"retryable status {response.status_code}")


# --- 1. Tool system --------------------------------------------------------
# How a tool may be scheduled when the model asks for several at once.
#   "parallel"  - may run alongside anything else. The default, and what
#                 every tool did before this existed.
#   "exclusive" - runs on its own: nothing else from that turn runs while it
#                 does. For tools whose concurrent calls would race.
_CONCURRENCY_MODES = ("parallel", "exclusive")
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _extract_json(content: str) -> tuple:
    """Find the JSON value in a model reply. Returns ``(value, found)``.

    Models fence their JSON, or wrap it in a sentence, so this strips code
    fences and falls back to the first ``{...}`` span. ``found`` distinguishes
    "the reply carried no JSON" from "the reply carried JSON that happens to
    be null" - a distinction :meth:`Router._resolve` needs and a plain return
    value cannot express.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), True
        except json.JSONDecodeError:
            pass
    return None, False


def _pydantic_schema(model: Type[Any]) -> Dict[str, Any]:
    """Return a JSON schema for a Pydantic model, across v1 and v2."""
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()  # pydantic v2
    if hasattr(model, "schema"):
        return model.schema()  # pydantic v1
    return {}


class Tool:
    """Wrap a function as an LLM-callable tool with an auto-generated schema.

    Type hints become JSON-Schema types, the docstring becomes the description,
    and parameters without a default are marked required. Beyond the basic
    scalars, ``Literal[...]`` and ``Enum`` subclasses become a JSON Schema
    ``enum``, and a nested Pydantic ``BaseModel`` parameter is inlined as an
    ``object`` with its own properties.

    ``run()`` also accepts ``async def`` functions: the coroutine is executed
    to completion via ``asyncio.run`` (see the "Async" note on ``run`` for the
    one caveat - it cannot be called from inside a running event loop; use
    ``await tool.func(...)`` directly in that case, or ``Agent.arun``).

    A tool may also carry security metadata, which is inert on its own - it
    is what a :class:`ToolPolicy` reads when deciding whether a model may
    call this tool:

    * ``permissions``  - what this tool needs, e.g. ``{"db:write"}``.
    * ``requires_approval`` - a human must confirm each call.
    * ``side_effects`` - True if calling it changes something.
    * ``allowed``      - ``fn(arguments, context) -> bool`` for a per-call
      check that depends on the arguments (a path prefix, a row limit, ...).
    * ``timeout``      - seconds the agent will wait for this tool before
      giving up on it. Overrides ``Agent(tool_timeout=...)``. Enforced by the
      agent, like the policy - :meth:`run` does not apply it, because a
      direct call is your own code calling your own function.
    * ``max_output_size`` - characters of result the agent will pass on to
      the model. Overrides ``Agent(max_tool_output_size=...)``. Also enforced
      by the agent, for the same reason.
    * ``concurrency``  - ``"parallel"`` (default) or ``"exclusive"``. An
      exclusive tool runs on its own: nothing else from that turn runs
      while it does. Mark a tool exclusive when two of its calls, running at
      once, would race. Note that ``side_effects=True`` does **not** imply
      this - it describes the tool to a policy and to the audit log, and says
      nothing about whether concurrent calls are safe.

    None of this is sent to the model. Metadata describes the tool to your
    policy; it is not a hint the model can read, argue with, or override.

    Three separate jobs, deliberately not conflated:

    * :attr:`schema` - **generation**. The JSON Schema shown to the model.
      Advice: it describes what to send, and the model is free to ignore it.
    * :meth:`validate_arguments` - **enforcement**. Checks and normalises
      what actually arrived, built from the Python signature rather than
      from ``schema``, because enforcing the same document the model was
      free to ignore would guarantee nothing.
    * :meth:`run` - **execution**. Calls the function, with validated
      arguments only.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        permissions: Optional[Iterable[str]] = None,
        requires_approval: bool = False,
        side_effects: bool = False,
        allowed: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
        timeout: Optional[float] = None,
        max_output_size: Optional[int] = None,
        concurrency: str = "parallel",
    ):
        if concurrency not in _CONCURRENCY_MODES:
            raise ValueError(
                f"Unknown concurrency {concurrency!r} for tool {func.__name__!r}. "
                f"Choose from {list(_CONCURRENCY_MODES)}."
            )
        self.func = func
        self.name = func.__name__
        self.description = (inspect.getdoc(func) or "").strip()
        self.is_async = inspect.iscoroutinefunction(func)
        # Security metadata is configuration, fixed at decoration time and
        # never rewritten during a run (frozenset, not set, on purpose).
        self.permissions = frozenset(permissions or ())
        self.requires_approval = requires_approval
        self.side_effects = side_effects
        self.allowed = allowed
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.concurrency = concurrency
        self.accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in inspect.signature(func).parameters.values()
        )
        self.schema = self._build_schema(func)  # what the model is shown
        self._validator = self._build_validator(func)  # what is enforced

    def _build_validator(self, func: Callable[..., Any]) -> Optional[Any]:
        """Build a Pydantic model from the signature, for runtime validation.

        Built from the *signature*, never from :attr:`schema`: the schema is
        advice given to the model, so enforcing it would mean trusting the
        same document the model is free to ignore.

        Returns None - leaving :meth:`validate_arguments` on its structural
        fallback - when Pydantic v2 is absent, or when a signature carries an
        annotation no model can be built from. Degrading is deliberate: a
        tool that used to import must not start raising at decoration time
        because its annotations are unusual.
        """
        if not _HAS_PYDANTIC_V2:  # pragma: no cover - v2 is the pinned floor
            return None
        try:
            hints = get_type_hints(func)
        except Exception:  # pragma: no cover - unresolvable forward refs
            hints = {}
        fields: Dict[str, Any] = {}
        for name, param in inspect.signature(func).parameters.items():
            if name in ("self", "cls") or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            # An unannotated parameter stays permissive (Any). The schema
            # advertises it as a string, but guessing here would reject
            # arguments that the plain Python function accepts happily.
            annotation = hints.get(name, Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)
        config = ConfigDict(
            # A tool without **kwargs cannot receive unknown names; one with
            # **kwargs accepts them by definition.
            extra="allow" if self.accepts_kwargs else "forbid",
            # Don't reject a signature just because Pydantic has no validator
            # for one of its types - fall back to an isinstance check.
            arbitrary_types_allowed=True,
            # A tool parameter may legitimately be called model_name.
            protected_namespaces=(),
        )
        try:
            with warnings.catch_warnings():
                # A parameter named json/copy/schema shadows a BaseModel
                # attribute. It works; the warning is not the author's problem.
                warnings.simplefilter("ignore")
                return create_model(f"{self.name}_arguments", __config__=config, **fields)
        except Exception as exc:  # pragma: no cover - exotic annotations only
            logger.debug(
                "tool %r: no Pydantic validator (%s); using structural checks", self.name, exc
            )
            return None

    def validate_arguments(self, arguments: Any) -> Dict[str, Any]:
        """Validate and normalise model-supplied arguments. **The gate.**

        Nothing reaches the wrapped function without passing here first.
        Every annotation the schema builder understands is enforced -
        ``str``/``int``/``float``/``bool``, lists and dicts (including their
        item types), ``Optional``, ``Literal``, ``Enum`` and Pydantic models
        nested to any depth - by validating against a model built from the
        signature. Rejected: missing required arguments, wrong types,
        malformed nested structures, and unknown names (unless the tool
        declares ``**kwargs``).

        Arguments are also *normalised*, so the function receives what its
        annotations promise: ``"42"`` arrives as ``42`` for an ``int``
        parameter, and a nested dict arrives as the declared Pydantic model.
        An ``Enum``-annotated parameter therefore receives the enum member,
        not the raw value.

        Returns the normalised arguments, ready to splat into the function.
        Raises :class:`ToolArgumentValidationError`, whose message names the
        offending field and the rule it broke but never repeats the value
        (see :meth:`_describe_errors`).

        Validating twice is harmless: normalised arguments pass again
        unchanged, which is what lets :class:`Agent` validate once for the
        policy and :meth:`run` validate again for anyone calling it directly.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolArgumentValidationError(
                f"tool '{self.name}' expects an object of arguments, got {type(arguments).__name__}"
            )
        unnamed = sorted(repr(key) for key in arguments if not isinstance(key, str))
        if unnamed:
            raise ToolArgumentValidationError(
                f"tool '{self.name}' got non-string argument name(s): {', '.join(unnamed)}"
            )
        if self._validator is None:  # pragma: no cover - no Pydantic v2
            return self._validate_structure(arguments)
        # Pydantic's own message embeds the offending input, and this
        # exception's text is handed to the model, written to memory and
        # recorded in the audit log. Only a redacted summary is kept, and it
        # is raised *after* the except block has finished so that neither
        # __cause__ nor __context__ holds the original - a raise inside the
        # block would leave the raw value reachable through __context__ for
        # anything that walks the chain, even though tracebacks hide it.
        problem: Optional[str] = None
        try:
            validated = self._validator.model_validate(arguments)
        except ValidationError as exc:
            problem = self._describe_errors(exc)
        if problem is not None:
            raise ToolArgumentValidationError(
                f"tool '{self.name}' got invalid arguments - {problem}"
            )
        # Return only what was actually sent, normalised: absent optional
        # parameters are left to the function's own defaults rather than
        # filled in here.
        extra = getattr(validated, "__pydantic_extra__", None) or {}
        return {key: extra[key] if key in extra else getattr(validated, key) for key in arguments}

    @staticmethod
    def _describe_errors(exc: Any) -> str:
        """Summarise a ValidationError without echoing the offending values.

        ``str(ValidationError)`` includes ``input_value=...``. That text goes
        back to the model, into conversation memory, and into the audit log,
        so a tool taking a password, token or customer record would leak it
        on any validation failure. Only the field path and the rule that
        failed are reported - enough for the model to correct itself, and
        nothing more.
        """
        details = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ())) or "(arguments)"
            details.append(f"{location}: {error.get('msg', 'is invalid')}")
        return "; ".join(details) or "arguments are invalid"

    def _validate_structure(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Names-and-required fallback for when Pydantic v2 is unavailable.

        Derived from the signature, like the real validator - not from
        :attr:`schema`. No types are checked on this path.
        """
        parameters = inspect.signature(self.func).parameters
        known = {
            name
            for name, param in parameters.items()
            if name not in ("self", "cls")
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }
        if not self.accepts_kwargs:
            unknown = sorted(set(arguments) - known)
            if unknown:
                raise ToolArgumentValidationError(
                    f"tool '{self.name}' got unexpected argument(s): {', '.join(unknown)}. "
                    f"It accepts: {', '.join(sorted(known)) or '(none)'}"
                )
        missing = [
            name
            for name in known
            if parameters[name].default is inspect.Parameter.empty and name not in arguments
        ]
        if missing:
            raise ToolArgumentValidationError(
                f"tool '{self.name}' is missing required argument(s): {', '.join(sorted(missing))}"
            )
        return arguments

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
        if origin is Literal:
            return self._enum_schema(get_args(hint))
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
        if isinstance(hint, type) and issubclass(hint, enum.Enum):
            return self._enum_schema([member.value for member in hint])
        if _HAS_PYDANTIC and isinstance(hint, type) and issubclass(hint, BaseModel):
            return _pydantic_schema(hint)
        return {"type": _PY_TO_JSON.get(hint, "string")}

    @staticmethod
    def _enum_schema(values: Any) -> Dict[str, Any]:
        """Build a JSON Schema ``enum``, plus a ``type`` if every value shares one.

        Shared by ``Literal[...]`` and ``Enum`` handling in ``_type_to_schema``.
        """
        values = list(values)
        kinds = {_PY_TO_JSON.get(type(v), "string") for v in values}
        base = {"type": kinds.pop()} if len(kinds) == 1 else {}
        return {**base, "enum": values}

    def run(self, arguments: Dict[str, Any]) -> Any:
        """Validate a dict of arguments, then call the wrapped function.

        This is the model-shaped entry point - a dict of arguments, as a tool
        call arrives - so it validates first, unconditionally. Nothing gets
        into the function through here without passing
        :meth:`validate_arguments`.

        :meth:`__call__` is the Python-shaped entry point (``my_tool(1, 2)``)
        and does *not* validate: it is your own code calling your own
        function, and Python's own argument handling applies.

        Async note: if the wrapped function is ``async def``, its coroutine is
        driven to completion with ``asyncio.run``. That only works when no
        event loop is already running in the current thread - inside async
        code, ``await`` the tool's underlying function directly instead
        (``await my_tool.func(**args)``), or drive the agent with
        ``Agent.arun``, which offloads the whole turn to a worker thread.
        """
        result = self.func(**self.validate_arguments(arguments))
        if inspect.iscoroutine(result):
            try:
                return asyncio.run(result)
            except RuntimeError as exc:
                result.close()
                raise RuntimeError(
                    f"Tool '{self.name}' is async and can't be run with asyncio.run() "
                    "because an event loop is already running on this thread. Await "
                    "it directly, or call the agent via 'await Agent.arun(...)'."
                ) from exc
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly, unvalidated. See :meth:`run`."""
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


@overload
def tool(func: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    permissions: Optional[Iterable[str]] = ...,
    requires_approval: bool = ...,
    side_effects: bool = ...,
    allowed: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = ...,
    timeout: Optional[float] = ...,
    max_output_size: Optional[int] = ...,
    concurrency: str = ...,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    permissions: Optional[Iterable[str]] = None,
    requires_approval: bool = False,
    side_effects: bool = False,
    allowed: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    timeout: Optional[float] = None,
    max_output_size: Optional[int] = None,
    concurrency: str = "parallel",
) -> Any:
    """Decorator: turn any function into a Tool with an auto-generated schema.

    Use it bare, or with security metadata for a :class:`ToolPolicy` to act
    on::

        @tool
        def search(query: str) -> str:
            "Read-only: needs no permission and has no side effects."

        @tool(permissions={"db:write"}, side_effects=True, requires_approval=True)
        def delete_record(record_id: str) -> str:
            "Destructive: gated by policy, and confirmed per call."

        @tool(timeout=10, max_output_size=8_000)
        def fetch_data(url: str) -> str:
            "Waited on for 10s, and trimmed to 8k characters."

        @tool(concurrency="exclusive", side_effects=True)
        def append_ledger(entry: str) -> str:
            "Two of these at once would interleave, so it runs alone."
    """

    def wrap(target: Callable[..., Any]) -> Tool:
        return Tool(
            target,
            permissions=permissions,
            requires_approval=requires_approval,
            side_effects=side_effects,
            allowed=allowed,
            timeout=timeout,
            max_output_size=max_output_size,
            concurrency=concurrency,
        )

    return wrap(func) if func is not None else wrap


# --- 2. LLM backend (unified across providers) -----------------------------
_PROVIDER_DEFAULTS = {
    "openai": ("gpt-4o-mini", "https://api.openai.com"),
    "anthropic": ("claude-3-5-sonnet-20241022", "https://api.anthropic.com"),
    "ollama": ("llama3.1", "http://localhost:11434"),
}
_PROVIDER_ENV_KEY = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# Overriding these lets provider="openai" target any OpenAI-compatible
# endpoint (Groq, Together, OpenRouter, vLLM, LM Studio, ...) with no code
# change, and lets Ollama point at a non-default host without an argument.
_PROVIDER_ENV_URL = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}


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
        cache: Union[bool, str] = False,
        cache_size: int = 256,
        cache_ttl: Optional[float] = None,
    ):
        self.provider = provider.lower().strip()
        if self.provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose from {list(_PROVIDER_DEFAULTS)}."
            )
        default_model, default_url = _PROVIDER_DEFAULTS[self.provider]
        self.model = model or default_model
        env_url = os.getenv(_PROVIDER_ENV_URL.get(self.provider, ""))
        self.base_url = (base_url or env_url or default_url).rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.max_retries, self.backoff = max_retries, backoff
        self.cache_size, self.cache_ttl = cache_size, cache_ttl
        # ``cache`` accepts a bool for convenience or a policy name outright:
        # True means "final_only", the safe default (see _CACHE_POLICIES).
        policy = "final_only" if cache is True else "none" if cache is False else str(cache)
        self.cache_policy = policy.lower().strip()
        if self.cache_policy not in _CACHE_POLICIES:
            raise ValueError(
                f"Unknown cache policy {cache!r}. Choose from {list(_CACHE_POLICIES)}."
            )
        # An LRU cache (OrderedDict, oldest first) capped at cache_size entries;
        # each value optionally expires after cache_ttl seconds.
        self.cache: Optional[OrderedDict[str, tuple[float, Dict[str, Any]]]] = (
            OrderedDict() if self.cache_policy != "none" else None
        )
        env_key = _PROVIDER_ENV_KEY.get(self.provider)
        self.api_key = api_key or (os.getenv(env_key) if env_key else None)
        # Reused across requests on this instance: cheaper than a fresh
        # requests.post() each time thanks to TCP/TLS connection pooling.
        self.session = requests.Session()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Send a chat request and return the normalised response dict.

        With caching enabled, identical requests are served from an in-memory
        LRU cache (bounded by ``cache_size``, optionally expiring after
        ``cache_ttl`` seconds) instead of hitting the provider again.

        What counts as cacheable is set by ``cache_policy``. Under the default
        ``"final_only"`` a response asking for tool calls is returned to the
        caller but never stored: see :meth:`_should_store`.
        """
        if self.cache is not None:
            key = self._cache_key(messages, tools, response_format)
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            result = self._dispatch(messages, tools, response_format)
            if self._should_store(result):
                self._cache_put(key, result)
            return result
        return self._dispatch(messages, tools, response_format)

    def _should_store(self, result: Dict[str, Any]) -> bool:
        """Decide whether a fresh response may enter the cache.

        A response carrying ``tool_calls`` is a decision to *act*. Storing it
        means a later identical prompt replays that decision without the model
        being consulted - the same refund issued twice, the same message sent
        again - and the replay is invisible, because no request goes out. The
        world the decision was made in has also moved on, while the cached
        answer has not.

        So ``"final_only"`` (the default) stores plain answers and drops
        tool-call responses. ``"all"`` stores them; use it only when every
        tool in play is read-only. The cache never *executes* anything either
        way: it decides what the model is taken to have said, and every tool
        call still passes :class:`ToolPolicy` and any approval hook before it
        runs.
        """
        if self.cache_policy == "all":
            return True
        return not result.get("tool_calls")

    def clear_cache(self) -> None:
        """Drop every cached response. Safe to call when caching is off."""
        if self.cache is not None:
            self.cache.clear()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        assert self.cache is not None
        entry = self.cache.get(key)
        if entry is None:
            return None
        timestamp, result = entry
        if self.cache_ttl is not None and (time.time() - timestamp) > self.cache_ttl:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)  # refresh LRU order on hit
        return self._copy_result(result)

    def _cache_put(self, key: str, result: Dict[str, Any]) -> None:
        assert self.cache is not None
        self.cache[key] = (time.time(), self._copy_result(result))
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # evict the oldest entry

    @staticmethod
    def _copy_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a response in and out of the cache.

        Callers put responses into conversation memory and read them later;
        without this, every hit would hand back the *same* dict, and one
        caller mutating it would silently rewrite what the cache serves
        everyone else.
        """
        # Deep, not shallow: tool-call arguments are nested dicts, and a
        # per-call shallow copy would still share them.
        return copy.deepcopy(
            {
                "content": result.get("content", ""),
                "tool_calls": result.get("tool_calls") or [],
                "usage": result.get("usage") or {},
            }
        )

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Async wrapper around :meth:`chat`.

        Unchained's HTTP layer is built on the synchronous ``requests``
        library by design (fewer dependencies, one code path to read). This
        offloads the blocking call to a worker thread with
        ``asyncio.to_thread`` so it can be awaited from async code without
        blocking the event loop - it does not make the request itself
        non-blocking or concurrent at the socket level.
        """
        return await asyncio.to_thread(self.chat, messages, tools, response_format)

    def close(self) -> None:
        """Close the underlying HTTP session. Optional; safe to skip."""
        self.session.close()

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
        """Identify a request by everything that can change its answer.

        Everything the provider is sent, or that selects which provider is
        sent to: endpoint, model, generation parameters, the conversation, the
        **full** tool schemas, and the requested output schema.

        Tools are keyed by schema rather than by name. Two tools can share a
        name and differ completely - a ``search`` over public documents and a
        ``search`` over internal records - and keying on the name alone would
        serve one tool's answer for the other. Schemas are sorted so that
        offering the same tools in a different order still hits.

        Response formats are keyed by their JSON schema for the same reason:
        two unrelated models are often both called ``Item``.

        Over-keying only costs a cache miss. Under-keying returns the wrong
        answer, so anything uncertain belongs in here.
        """
        return json.dumps(
            {
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "tools": sorted(json.dumps(t.schema, sort_keys=True) for t in tools)
                if tools
                else None,
                "response_format": None
                if response_format is None
                else {
                    "name": getattr(response_format, "__name__", None),
                    "schema": _pydantic_schema(response_format),
                },
            },
            sort_keys=True,
            default=str,
        )

    # -- OpenAI --
    def _openai_headers(self) -> Dict[str, str]:
        """Headers for an OpenAI-compatible endpoint.

        The ``Authorization`` header is omitted entirely when no key is
        configured, rather than sent as the literal string ``Bearer None``.
        Sending a placeholder credential is never useful - hosted providers
        reject it with a confusing 401 - and local OpenAI-compatible servers
        (vLLM, LM Studio, llama.cpp, Ollama's compat endpoint) accept an
        unauthenticated request but not a malformed one.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _anthropic_headers(self) -> Dict[str, str]:
        """Headers for the Anthropic Messages API. See :meth:`_openai_headers`."""
        headers = {"anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _openai(self, messages, tools, response_format):
        headers = self._openai_headers()
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
        headers = self._anthropic_headers()
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
        headers = self._openai_headers()
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
        headers = self._anthropic_headers()
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
        """POST with retries on connection errors and 429/5xx responses.

        Uses a persistent ``requests.Session`` so repeated calls on the same
        ``LLM`` instance reuse the underlying TCP/TLS connection instead of
        renegotiating one per request.
        """
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, timeout=self.timeout, **kwargs)
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
# A dependency-free rule of thumb (no tokenizer needed): English text averages
# roughly 4 characters per token. It's approximate, but good enough to decide
# when a window is at risk of overflowing a model's context window.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: Any) -> int:
    return max(1, len(str(content)) // _CHARS_PER_TOKEN)


class Memory:
    """Fixed sliding-window conversation memory.

    On overflow the oldest half is compressed into a running ``summary`` - via
    the LLM if one is supplied, otherwise by truncating each message to 100 chars.

    ``max_messages`` bounds the window by message count. Optionally also pass
    ``max_tokens`` to additionally shrink the retained window (using a rough
    4-chars-per-token estimate, no tokenizer dependency) whenever the kept
    messages alone would exceed it - message count alone can't promise a
    token budget, since a handful of long messages can still blow the context
    window even under a small ``max_messages``.

    One exception to both bounds: the window is never cut between an
    assistant message carrying ``tool_calls`` and the ``tool`` messages that
    answer them, because providers reject an unpaired tool result. When the
    boundary would land inside such a group it moves back to the start of the
    group, so the window can exceed ``max_messages`` (or ``max_tokens``) by
    that one group. See :meth:`_tool_group_start`.
    """

    def __init__(
        self,
        max_messages: int = 20,
        llm: Optional[LLM] = None,
        max_tokens: Optional[int] = None,
    ):
        self.max_messages, self.llm = max_messages, llm
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""

    def add(self, role: str, content: Any, **extra: Any) -> None:
        self.messages.append({"role": role, "content": content, **extra})
        if len(self.messages) > self.max_messages or self._over_token_budget():
            self._compress()

    def get(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def _over_token_budget(self) -> bool:
        if self.max_tokens is None:
            return False
        return self._window_tokens(self.messages) > self.max_tokens

    @staticmethod
    def _window_tokens(messages: List[Dict[str, Any]]) -> int:
        return sum(_estimate_tokens(m.get("content", "")) for m in messages)

    @staticmethod
    def _tool_group_start(messages: List[Dict[str, Any]], index: int) -> int:
        """Move a window boundary back so it never splits a tool group.

        A ``tool`` message is only meaningful directly after the assistant
        message whose ``tool_calls`` requested it, and every provider rejects
        an unpaired one (OpenAI answers HTTP 400 "messages with role 'tool'
        must be a response to a preceding message with 'tool_calls'";
        Anthropic rejects the equivalent ``tool_result`` block). So if the
        boundary lands on a tool result, walk back to the assistant turn that
        asked for it.

        The kept window can therefore exceed ``max_messages`` by one tool
        group. That is deliberate: a window one group over budget is still
        sendable, whereas a window that starts with an orphaned tool result
        is rejected outright.
        """
        while index > 0 and messages[index].get("role") == "tool":
            index -= 1
        return index

    def _compress(self) -> None:
        keep = max(1, self.max_messages // 2)
        split = max(0, len(self.messages) - keep)
        # A message-count window can still bust a token budget (e.g. 20 huge
        # messages), so keep shrinking from the front until it fits - but
        # always leave at least the single most recent message.
        if self.max_tokens is not None:
            while (
                split < len(self.messages) - 1
                and self._window_tokens(self.messages[split:]) > self.max_tokens
            ):
                split += 1
        split = self._tool_group_start(self.messages, split)
        overflow, kept = self.messages[:split], self.messages[split:]
        self.messages = kept
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

    Inputs are validated rather than trusted, because the failures here are
    silent ones: a vector of the wrong length used to be zipped against a
    longer one and scored 1.0, and a negative ``top_k`` used to slice the
    result list from the end. Specifically:

    * ``embed_fn`` must return exactly one vector per text.
    * The index takes its dimension from the first vector it accepts, and
      every later vector - document or query - must match it.
    * Vectors must be non-empty and finite.
    * Documents must be non-empty; ``top_k`` must be an integer of at least 1.
    * Nothing is stored until all of it validates, so a failing ``embed_fn``
      leaves the index exactly as it was.

    Duplicate documents are kept, not merged: the same text can legitimately
    arrive twice with different metadata. Both are returned, and equal scores
    keep insertion order.
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
        # Fixed by the first vector accepted; everything after must match.
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> Optional[int]:
        """The embedding width this index has settled on, or None if empty."""
        return self._dimension

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.add_many([text], [metadata or {}])

    def add_many(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index several documents at once.

        Everything is checked before anything is stored, so a rejected batch
        leaves the index exactly as it was - half-adding a batch is how an
        index ends up with more embeddings than documents and an
        ``IndexError`` from ``search()`` much later, far from the cause.

        ``metadatas``, when given, must line up one-to-one with ``texts``.
        Zipping the two used to truncate to the shorter list, silently
        dropping documents.
        """
        texts = list(texts)
        if metadatas is not None:
            metadatas = list(metadatas)
            if len(metadatas) != len(texts):
                raise ValueError(
                    f"add_many() got {len(texts)} texts but {len(metadatas)} metadatas; "
                    "they must be the same length."
                )
        if not texts:
            return  # nothing to do, and no reason to call embed_fn with nothing
        for position, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"add_many() got a {type(text).__name__} at position {position}; "
                    "documents must be strings."
                )
            if not text.strip():
                raise ValueError(
                    f"add_many() got an empty document at position {position}. An empty "
                    "document can never match anything and only dilutes results."
                )
        embeddings, dimension = (None, self._dimension)
        if self.embed_fn is not None:
            embeddings, dimension = self._embed(texts)

        # Everything validated; commit.
        self.docs.extend(texts)
        self.metadata.extend(meta or {} for meta in (metadatas or [{} for _ in texts]))
        self._tf.extend(Counter(self._tokenize(text)) for text in texts)
        self._dimension = dimension
        if embeddings is not None:
            self._embeddings.extend(embeddings)
        else:
            self._rebuild_index()

    def _embed(self, texts: List[str]) -> tuple:
        """Embed a batch and validate it. Returns ``(vectors, dimension)``.

        Nothing here touches the index: the dimension is returned rather than
        assigned, so a batch that fails half-way cannot leave the index
        claiming a width it never accepted a vector for.
        """
        assert self.embed_fn is not None
        produced = self.embed_fn(list(texts))
        try:
            vectors = [list(vector) for vector in produced]
        except TypeError:
            raise ValueError(
                "embed_fn must return a sequence of vectors, one per text, "
                f"but returned {type(produced).__name__}."
            ) from None
        if len(vectors) != len(texts):
            raise ValueError(
                f"embed_fn returned {len(vectors)} vectors for {len(texts)} texts. "
                "It must return exactly one per text, in order, or documents and "
                "embeddings drift apart."
            )
        dimension = self._dimension
        for position, vector in enumerate(vectors):
            dimension = self._check_vector(vector, f"document {position}", dimension)
        return vectors, dimension

    @staticmethod
    def _check_vector(vector: List[float], what: str, dimension: Optional[int]) -> int:
        """Validate one vector against the index width. Returns the width.

        A zero vector is allowed - some models emit one for input they cannot
        represent, and cosine handles it - but an *empty* one is not, since a
        zero-width index can never rank anything.
        """
        if not vector:
            raise ValueError(
                f"embed_fn produced an empty vector for {what}. A vector needs at "
                "least one dimension to be comparable."
            )
        if dimension is not None and len(vector) != dimension:
            raise ValueError(
                f"{what} has {len(vector)} dimensions, but this index is "
                f"{dimension}-dimensional. Mixed widths cannot be compared - the "
                "shorter one used to be silently zipped against the longer and "
                "scored as if it matched."
            )
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{what} contains a non-numeric value ({value!r}); vectors must be numbers."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"{what} contains {value!r}; a non-finite value makes every "
                    "score involving it meaningless."
                )
        return len(vector)

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
        """Return the ``top_k`` best matches, highest score first.

        An empty index returns nothing. In TF-IDF mode a query with no
        indexable tokens also returns nothing - there is nothing to search
        *with*, which is different from having searched and found nothing
        similar, and returning arbitrary documents scored 0.0 only looks like
        a result. Embedding mode leaves that judgement to ``embed_fn``.

        Equal scores keep insertion order, so duplicate documents come back
        in the order they were added.
        """
        if not isinstance(query, str):
            raise ValueError(f"search() needs a string query, got {type(query).__name__}.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k < 1:
            raise ValueError(
                f"top_k must be at least 1, got {top_k}. A negative value used to slice "
                "the ranked list from the end and quietly return the wrong documents."
            )
        if not self.docs:
            return []
        if self.embed_fn is None and not self._tokenize(query):
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
        """Score the corpus against the query vector, which must match its width."""
        assert self.embed_fn is not None
        produced = list(self.embed_fn([query]))
        if len(produced) != 1:
            raise ValueError(
                f"embed_fn returned {len(produced)} vectors for one query; it must "
                "return exactly one."
            )
        vector = list(produced[0])
        self._check_vector(vector, "the query", self._dimension)
        return [self._cosine(vector, embedding) for embedding in self._embeddings]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Cosine similarity, in ``[-1.0, 1.0]``.

        Raises on mismatched lengths rather than zipping to the shorter one,
        which silently compared a prefix and could report a perfect match
        between vectors of different widths.

        A zero vector has no direction, so its similarity to anything is
        undefined; 0.0 is returned rather than raising, because a model
        emitting one for unrepresentable input is not a caller error. Values
        large enough to overflow the squared sum also give 0.0 - the ranking
        is not recoverable, and NaN would corrupt the sort.
        """
        if len(a) != len(b):
            raise ValueError(
                f"cosine similarity needs vectors of equal length, got {len(a)} and {len(b)}."
            )
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        score = dot / (na * nb)
        if not math.isfinite(score):
            return 0.0
        # Rounding can push an identical pair a hair past 1.0.
        return max(-1.0, min(1.0, score))

    def __len__(self) -> int:
        return len(self.docs)


# --- 5. Observability (callbacks and events) -------------------------------
@dataclass(frozen=True)
class AgentEvent:
    """One thing that happened during a run.

    A flat, immutable record - enough for a log line, a metric, or a trace
    you assemble yourself, without becoming a tracing framework. Correlate
    with ``run_id`` (unique per run, so concurrent runs never interleave) and
    ``session_id`` (stable across a conversation).

    ``event_type`` is one of:

    ===================  =========================================
    ``AgentStarted``     a run began
    ``AgentIteration``   a think/act cycle began
    ``LLMStarted``       a provider request went out
    ``LLMFinished``      it came back (carries ``usage``, ``duration``)
    ``ToolStarted``      an authorized tool call began
    ``ToolFinished``     it returned (carries ``duration``)
    ``ToolFailed``       it did not return a result - refused, invalid,
                         denied, timed out or raised
    ``AgentFinished``    the run produced an answer
    ``AgentFailed``      the run raised
    ===================  =========================================

    ``ToolFailed`` can arrive without a preceding ``ToolStarted``: a call
    refused before it ran never started. Its ``metadata["reason"]`` says
    which stage refused it.

    **Payloads are excluded by default.** Prompts, tool arguments, tool
    results and the final answer are the things most likely to hold personal
    data or credentials, and an event stream usually ends up in a log
    aggregator. Events carry shapes and sizes instead - message counts,
    character counts. Pass ``Agent(event_payloads=True)`` to include the
    content itself, deliberately.

    Frozen, so a handler cannot rewrite an event other handlers will see.
    ``metadata`` is copied in on construction; it is a plain dict, so it is
    not deeply frozen.
    """

    event_type: str
    run_id: str
    session_id: str
    agent: str
    timestamp: float = 0.0
    duration: Optional[float] = None
    model: Optional[str] = None
    tool: Optional[str] = None
    tool_call_id: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy the caller's mapping. frozen=True blocks attribute assignment,
        # not mutation of a dict the event was built from - without this an
        # event could still be rewritten through it after the fact.
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> Dict[str, Any]:
        """A plain dict, for structured logging."""
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"{self.event_type} run={self.run_id}"]
        for name in ("tool", "model"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={value}")
        if self.duration is not None:
            parts.append(f"{self.duration * 1000:.0f}ms")
        if self.usage:
            parts.append(f"tokens={self.usage.get('total_tokens', 0)}")
        return " ".join(parts)


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

    def on_tool_audit(self, event: Dict[str, Any]) -> None:
        """Called with every tool authorization decision, before execution.

        ``event`` carries ``agent``, ``tool``, ``arguments``, ``decision``
        (one of ``allowed``, ``approved``, ``denied``, ``approval_denied``,
        ``invalid_arguments``, ``unknown_tool``), ``reason``, ``permissions``
        and ``side_effects``.

        It fires before the tool runs, so the record survives a tool that
        hangs or crashes. ``arguments`` are verbatim - redact them in your
        sink if your tools take secrets.
        """

    def on_event(self, event: AgentEvent) -> None:
        """Called with every :class:`AgentEvent`.

        The structured stream: one handler covering starts, finishes,
        failures, timings and token usage, rather than a method per hook. The
        older, narrower callbacks above still fire alongside it, so existing
        subclasses keep working unchanged.
        """

    def on_finish(self, answer: Any) -> None:
        """Called once with the final answer."""


class _EventListener(Callback):
    """Adapts a plain function into a Callback. See :meth:`Agent.subscribe`."""

    def __init__(self, handler: Callable[[AgentEvent], None]):
        self._handler = handler

    def on_event(self, event: AgentEvent) -> None:
        self._handler(event)


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

    def on_tool_audit(self, event: Dict[str, Any]) -> None:
        self.log.info(
            "audit: %s %s(%s)%s",
            event.get("decision"),
            event.get("tool"),
            event.get("arguments"),
            f" - {event['reason']}" if event.get("reason") else "",
        )

    def on_event(self, event: AgentEvent) -> None:
        self.log.info("%s", event)

    def on_finish(self, answer: Any) -> None:
        self.log.info("finished (%d chars)", len(str(answer)))


# --- 6. Tool authorization (policy layer) ----------------------------------
class ToolPolicy:
    """Decide whether a model-requested tool call may run.

    An LLM choosing a tool is a *request*, not a decision. This is where the
    request is granted or refused, in Python, before the function is called.
    Nothing here is expressed to the model as an instruction: a policy is not
    a prompt saying "only call safe tools", and no wording the model emits
    can widen what it is allowed to do.

    This base class is also the default policy, and it is permissive: it
    allows any tool the agent was given and asks for approval only when the
    tool itself is marked ``requires_approval``. An agent written before this
    layer existed therefore behaves exactly as it did - the boundary is
    always present, its default answer is "yes".

    Subclass to restrict. Deny by raising :class:`ToolAuthorizationError`;
    the agent turns that into an observation for the model rather than
    crashing the run, so the model learns it was refused and can try
    something else::

        class BusinessHoursOnly(ToolPolicy):
            def authorize(self, tool, arguments, context):
                if tool.side_effects and not is_working_hours():
                    raise ToolAuthorizationError("no writes outside business hours")
                super().authorize(tool, arguments, context)

    Both hooks receive the tool, the arguments (already through
    :meth:`Tool.validate_arguments`, so the shape is known-good), and a
    context dict of ``agent``, ``tool`` and ``call_id``. A policy that raises
    anything else is treated as a denial - a broken policy must not fail open.
    """

    def authorize(self, tool: Tool, arguments: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Allow the call by returning; deny it by raising ToolAuthorizationError."""
        if tool.allowed is not None and not tool.allowed(arguments, context):
            raise ToolAuthorizationError(
                f"tool '{tool.name}' refused this call (its allowed() hook returned False)"
            )

    def requires_approval(
        self, tool: Tool, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """Return True if a human must confirm this call before it runs."""
        return tool.requires_approval


class PermissionPolicy(ToolPolicy):
    """Allow only tools whose declared ``permissions`` have all been granted.

    The tool declares what it needs; the application grants what this agent
    may have. Anything ungranted is denied, so a tool added to the agent
    later without a matching grant is refused rather than quietly allowed::

        read_only = PermissionPolicy(granted={"db:read"})
        writer = PermissionPolicy(granted={"db:read", "db:write"},
                                  approval_for={"db:write"})

    ``approval_for`` names granted permissions that must still be confirmed
    per call, on top of any tool marked ``requires_approval``.

    Note the deliberate limit: a tool that declares *no* permissions requires
    none, and passes. The agent's own ``tools`` list is the first allowlist -
    this policy narrows it, it does not replace it. Declare permissions on
    every tool you intend to gate, and watch the audit log (which records the
    permissions of each call) for tools you forgot to label.
    """

    def __init__(
        self,
        granted: Optional[Iterable[str]] = None,
        approval_for: Optional[Iterable[str]] = None,
    ):
        self.granted = frozenset(granted or ())
        self.approval_for = frozenset(approval_for or ())

    def authorize(self, tool: Tool, arguments: Dict[str, Any], context: Dict[str, Any]) -> None:
        ungranted = tool.permissions - self.granted
        if ungranted:
            raise ToolAuthorizationError(
                f"tool '{tool.name}' requires permission(s) {sorted(ungranted)}, "
                f"which this agent was not granted"
            )
        super().authorize(tool, arguments, context)

    def requires_approval(
        self, tool: Tool, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        return super().requires_approval(tool, arguments, context) or bool(
            tool.permissions & self.approval_for
        )


# --- 7. Budgets (runtime governance) ---------------------------------------
class Budget:
    """What a single run of an agent may consume. Every limit is optional.

    A budget is **per run** - one ``agent.run()`` or ``session.run()`` call -
    not per session lifetime. A ten-turn conversation gets the budget ten
    times. Lifetime token accounting is ``session.usage``, which keeps
    accumulating regardless.

    * ``max_iterations``   - think/act cycles. Defaults to the agent's own
      ``max_iterations``. **This one does not raise:** when the loop is
      exhausted the agent makes one final call and answers, exactly as it
      always has. Ending a turn with no answer at all is worse than one more
      call, and that behaviour predates budgets.
    * ``max_tool_calls``   - tool calls in the run, counted across every
      iteration and every concurrent call in a turn.
    * ``max_total_tokens`` - prompt + completion tokens, as reported by the
      provider.
    * ``max_tool_output``  - total characters of tool output for the run.
      Distinct from ``Tool.max_output_size``, which caps a *single* result;
      this caps the sum, so a hundred well-behaved tools cannot add up to a
      context overflow.
    * ``timeout``          - wall-clock seconds for the whole run.
    * ``max_cost``         - estimated spend, in whatever unit ``pricing``
      uses. Requires ``pricing``.

    ``pricing`` maps a model name to ``(input_rate, output_rate)`` per
    1,000,000 tokens::

        Budget(max_cost=0.50, pricing={"gpt-4o-mini": (0.15, 0.60)})

    Rates are yours to supply. Unchained ships no price table, because
    published prices change and a table baked into this file would quietly go
    stale - and a cost cap computed from stale numbers is worse than none.
    Cost is therefore always an **estimate** from the provider's own token
    counts, never an invoice.

    Budgets that are reached stop the run by raising a
    :class:`BudgetExceededError`. They are checked before spending, but a
    single call's cost is not known until it returns, so the final call can
    carry the total slightly past the limit; the run stops immediately
    afterwards. This is runtime governance, not billing.
    """

    def __init__(
        self,
        max_iterations: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        max_tool_output: Optional[int] = None,
        timeout: Optional[float] = None,
        max_cost: Optional[float] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ):
        if max_cost is not None and not pricing:
            raise ValueError(
                "Budget(max_cost=...) needs pricing to compute against, e.g. "
                'pricing={"gpt-4o-mini": (0.15, 0.60)} per 1M tokens. Without it '
                "the cap could not be enforced, and pretending otherwise would be "
                "worse than having no cap."
            )
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_total_tokens = max_total_tokens
        self.max_tool_output = max_tool_output
        self.timeout = timeout
        self.max_cost = max_cost
        self.pricing = dict(pricing or {})

    def __repr__(self) -> str:  # pragma: no cover
        set_limits = {
            name: value
            for name, value in vars(self).items()
            if name != "pricing" and value is not None
        }
        return f"<Budget {set_limits or 'unlimited'}>"


class RunState:
    """Live accounting for one run, and the record of it afterwards.

    Readable during a run from ``session.last_run`` and left in place when the
    run ends - including when a budget stopped it, where ``exceeded`` names
    the limit that did. :meth:`snapshot` returns the same information as a
    plain dict for logging.

    Counters are updated from the worker threads that run a turn's tool calls,
    so they are taken under a lock. That is bookkeeping, not shared
    conversation state: the accounting for a run belongs to that run.
    """

    def __init__(self, budget: Budget):
        self.budget = budget
        # Distinct per run, so concurrent runs - of one agent or many - never
        # share an id and their events can be told apart.
        self.id = uuid.uuid4().hex[:12]
        self.started = time.monotonic()
        self.iterations = 0
        self.tool_calls = 0
        self.tool_output_chars = 0
        self.usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.estimated_cost = 0.0
        # False once any call could not be priced, so a reported cost is
        # never mistaken for a complete one.
        self.cost_is_complete = True
        self.exceeded: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since the run started."""
        return time.monotonic() - self.started

    def snapshot(self) -> Dict[str, Any]:
        """A plain, JSON-serialisable record of the run so far."""
        return {
            "run_id": self.id,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tool_output_chars": self.tool_output_chars,
            "usage": dict(self.usage),
            "elapsed": round(self.elapsed, 3),
            "estimated_cost": round(self.estimated_cost, 6),
            "cost_is_complete": self.cost_is_complete,
            "exceeded": self.exceeded,
        }

    # -- checks ----------------------------------------------------------
    def _stop(self, error: BudgetExceededError) -> BudgetExceededError:
        self.exceeded = error.limit_name
        logger.warning("run stopped: %s", error)
        return error

    def check_before_call(self) -> None:
        """Raise if the run may not spend anything more.

        Called before every LLM call and before every tool call, so nothing
        that costs time, tokens or money starts once a budget is reached.
        """
        budget = self.budget
        if budget.timeout is not None and self.elapsed >= budget.timeout:
            raise self._stop(
                TimeBudgetExceeded(
                    "timeout", budget.timeout, round(self.elapsed, 3), "wall-clock seconds"
                )
            )
        if (
            budget.max_total_tokens is not None
            and self.usage["total_tokens"] >= budget.max_total_tokens
        ):
            raise self._stop(
                TokenBudgetExceeded(
                    "max_total_tokens", budget.max_total_tokens, self.usage["total_tokens"]
                )
            )
        if budget.max_cost is not None and self.estimated_cost >= budget.max_cost:
            raise self._stop(
                CostBudgetExceeded(
                    "max_cost",
                    budget.max_cost,
                    round(self.estimated_cost, 6),
                    "estimated from provider token counts",
                )
            )

    def reserve_tool_call(self, name: str) -> None:
        """Claim one tool call against the budget, or raise.

        Reserved under the lock rather than checked then incremented: a turn's
        tool calls run concurrently, and a check-then-act would let several
        pass a limit only one of them could have.
        """
        limit = self.budget.max_tool_calls
        with self._lock:
            if limit is not None and self.tool_calls >= limit:
                raise self._stop(
                    ToolCallBudgetExceeded(
                        "max_tool_calls", limit, self.tool_calls, f"tool {name!r}"
                    )
                )
            self.tool_calls += 1

    def record_tool_output(self, size: int) -> None:
        """Add a tool result's size to the run, raising if the total passes the cap.

        The size is only known once the tool has returned, so the total can
        pass the cap by that one result; the run stops before anything else
        is spent. ``Tool.max_output_size`` is what bounds an individual one.
        """
        limit = self.budget.max_tool_output
        with self._lock:
            self.tool_output_chars += size
            total = self.tool_output_chars
        if limit is not None and total > limit:
            raise self._stop(
                ToolOutputBudgetExceeded("max_tool_output", limit, total, "characters")
            )

    def record_llm_call(self, usage: Optional[Dict[str, Any]], model: str) -> None:
        """Fold one LLM response's usage, and its estimated cost, into the run."""
        with self._lock:
            for key in self.usage:
                self.usage[key] += int((usage or {}).get(key, 0) or 0)
            self.estimated_cost += self._price(usage or {}, model)

    def _price(self, usage: Dict[str, Any], model: str) -> float:
        """Estimate one call's cost, or refuse to guess."""
        rates = self.budget.pricing.get(model)
        if rates is None:
            self.cost_is_complete = False
            if self.budget.max_cost is not None:
                raise self._stop(
                    CostBudgetExceeded(
                        "max_cost",
                        self.budget.max_cost,
                        round(self.estimated_cost, 6),
                        f"no pricing for model {model!r}; add it to Budget(pricing=...) "
                        "or drop max_cost, because a cap that cannot be computed "
                        "cannot be enforced",
                    )
                )
            return 0.0
        input_rate, output_rate = rates
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        return (prompt / 1_000_000) * input_rate + (completion / 1_000_000) * output_rate


# --- 8. Agent core (ReAct loop) --------------------------------------------
class Agent:
    """A ReAct agent: think (LLM) -> act (tool) -> observe -> repeat.

    Compose it with tools, memory and RAG. ``run`` optionally takes a Pydantic
    model for validated structured output; ``stream`` yields the answer token
    by token. Attach ``callbacks`` for tracing and read ``usage`` for token
    accounting.

    **An Agent is configuration and behaviour; a** :class:`Session` **is one
    conversation's state.** Nothing about a particular conversation - its
    memory, its token counters - lives on the Agent, so one Agent can serve
    many users and many concurrent requests::

        agent = Agent(llm, tools=[...])      # build once, share freely
        alice = agent.session()              # independent conversations
        bob = agent.session()

    ``agent.run(...)`` still works and is unchanged for single-conversation
    scripts: it uses one persistent default session, created on first use.
    See :meth:`session` and :attr:`default_session`.

    Three things *are* deliberately shared by every session of an Agent,
    because they are resources rather than conversation state:

    * ``llm`` - its HTTP connection pool and response cache. Sharing is the
      point; that is what makes them worth having.
    * ``rag`` - a knowledge base is read by every conversation. Adding
      documents at runtime affects them all, by design.
    * ``callbacks`` - Agent-level callbacks observe every session. Pass
      ``agent.session(callbacks=[...])`` for a sink scoped to one
      conversation instead.

    Anything you add yourself that holds per-conversation state belongs on a
    Session, not here.
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
        max_tool_workers: int = 8,
        policy: Optional[ToolPolicy] = None,
        approve: Optional[Callable[[Dict[str, Any]], bool]] = None,
        memory_factory: Optional[Callable[[], Memory]] = None,
        tool_timeout: Optional[float] = None,
        max_tool_output_size: Optional[int] = None,
        budget: Optional[Budget] = None,
        event_payloads: bool = False,
        strict_callbacks: bool = False,
    ):
        """``memory`` and ``memory_factory`` differ, and the difference matters.

        ``memory`` is an instance, and it belongs to *the default session
        only* - the one behind ``agent.run()``. It is never handed to a
        session from :meth:`session`, because sharing one Memory between
        conversations is precisely the bug this split exists to prevent.

        ``memory_factory`` is how every other session gets its memory: a
        zero-argument callable returning a fresh Memory. Use it to carry
        configuration (a window size, a subclass) into every conversation::

            Agent(llm, memory_factory=lambda: Memory(max_messages=50))

        It defaults to ``Memory``, i.e. framework defaults per session.
        """
        self.llm = llm
        self.name = name
        self.description = description or system_prompt
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        # How a new session gets its memory. A factory, not an instance:
        # handing the same Memory to two sessions would merge two
        # conversations, which is the failure this whole split prevents.
        self.memory_factory = memory_factory or Memory
        self.rag = rag
        self.max_iterations = max_iterations
        self.callbacks = list(callbacks or [])
        self.structured_retries = structured_retries
        # How many of one turn's tool calls may run at once. The number of
        # calls in a turn is chosen by the model, so it is untrusted input;
        # without a cap a single response sizes the thread pool. See
        # _execute_calls.
        self.max_tool_workers = max(1, max_tool_workers)
        # Every model-requested tool call goes through this policy; the
        # default one allows what the agent was already given. See _execute.
        self.policy = policy or ToolPolicy()
        # Approval is application-controlled by construction: this callback
        # comes from the caller of Agent(), and nothing the model emits can
        # set, reach or influence it.
        self.approve = approve
        # Seconds to wait for any tool that does not set its own timeout.
        # None means wait forever, which is the old behaviour.
        self.tool_timeout = tool_timeout
        # Characters of tool output to pass on, for tools that do not set
        # their own. None means unbounded, which is the old behaviour.
        self.max_tool_output_size = max_tool_output_size
        # What one run may consume. An empty Budget limits nothing except the
        # iteration count, which max_iterations has always bounded.
        self.budget = budget or Budget()
        # Whether AgentEvents carry prompts, tool arguments, tool results and
        # the final answer. Off by default: an event stream usually ends up
        # in a log aggregator, and those fields are where the personal data
        # and credentials are. See AgentEvent.
        self.event_payloads = event_payloads
        # Instrumentation must not break a run, so callback errors are logged
        # and swallowed. Set this to surface them instead - useful in tests,
        # where a silently broken sink looks like a working one.
        self.strict_callbacks = strict_callbacks
        # A turn's tool calls run concurrently, but a CLI prompt or a modal
        # dialog must not be re-entered from several workers at once. This
        # stays on the Agent on purpose: it guards the application's single
        # approval UI, so it must serialise across sessions too.
        self._approval_lock = threading.Lock()
        # The default session is built on first use, not here, so an Agent
        # that only ever serves explicit sessions never allocates one.
        self._default_memory = memory
        self._default_session: Optional[Session] = None
        self._default_session_lock = threading.Lock()

    # -- sessions --
    def session(
        self,
        memory: Optional[Memory] = None,
        callbacks: Optional[List[Callback]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        budget: Optional[Budget] = None,
    ) -> Session:
        """Start a new, independent conversation with this agent.

        The returned :class:`Session` owns its own memory and usage counters,
        so sessions do not see each other's history and can run concurrently::

            alice = agent.session(metadata={"user": "alice"})
            bob = agent.session(metadata={"user": "bob"})

        ``memory`` overrides this session's store - the hook for per-user
        persistence (``agent.session(memory=SQLiteMemory(session_id=uid))``).
        Otherwise ``memory_factory`` supplies a fresh one.

        ``callbacks`` are additional and scoped to this session; Agent-level
        callbacks still fire. ``metadata`` is yours - it is passed to
        :class:`ToolPolicy` hooks as ``context["metadata"]`` and included in
        audit events, which is how a policy authorizes per user rather than
        per agent.
        """
        return Session(
            self,
            memory=memory if memory is not None else self.memory_factory(),
            callbacks=callbacks,
            metadata=metadata,
            session_id=session_id,
            budget=budget,
        )

    @property
    def default_session(self) -> Session:
        """The one persistent session behind ``agent.run()``.

        Created on first use and reused for the life of the Agent, so a
        script that calls ``agent.run()`` in a loop holds a single
        conversation - exactly as it did before sessions existed. It is
        *not* recreated per call.

        That persistence is why ``agent.run()`` is for single-conversation
        use. Serving several users from one Agent means one session each
        (:meth:`session`); leaning on the default session would merge them.
        Call :meth:`reset` to start the default conversation over.
        """
        if self._default_session is None:
            with self._default_session_lock:
                if self._default_session is None:
                    self._default_session = Session(
                        self,
                        memory=(
                            self._default_memory
                            if self._default_memory is not None
                            else self.memory_factory()
                        ),
                        session_id="default",
                    )
        return self._default_session

    def reset(self) -> None:
        """Clear the default session's memory and usage counters.

        Sessions from :meth:`session` are untouched - they are not the
        Agent's to clear.
        """
        self.default_session.reset()

    @property
    def memory(self) -> Memory:
        """The default session's memory. See :attr:`default_session`."""
        return self.default_session.memory

    @memory.setter
    def memory(self, value: Memory) -> None:
        self.default_session.memory = value

    @property
    def usage(self) -> Dict[str, int]:
        """The default session's token counters. See :attr:`default_session`.

        Usage is per conversation. There is no Agent-wide total: that would
        be shared mutable state, and two concurrent sessions would race on
        it. Sum the sessions you care about in your own code.
        """
        return self.default_session.usage

    # -- running a turn --
    #
    # Every method below that touches conversation state takes the session
    # explicitly as its first argument. That is deliberate: the signature is
    # the audit trail. A method without a ``session`` parameter cannot reach
    # a conversation's memory or usage, so "what is shared?" is answerable by
    # reading the signatures rather than the bodies.
    def run(self, user_input: str, response_format: Optional[Type[BaseModel]] = None) -> Any:
        """Take a turn in this agent's persistent default conversation.

        Equivalent to ``agent.default_session.run(...)``. Convenient for
        scripts and single-conversation use; for concurrent users, give each
        one its own :meth:`session`.
        """
        return self.default_session.run(user_input, response_format)

    def stream(self, user_input: str) -> Iterator[str]:
        """Stream a turn of the default conversation. See :meth:`run`."""
        return self.default_session.stream(user_input)

    async def arun(self, user_input: str, response_format: Optional[Type[BaseModel]] = None) -> Any:
        """Await a turn of the default conversation. See :meth:`run`."""
        return await self.default_session.arun(user_input, response_format)

    def _run(
        self,
        session: Session,
        user_input: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """The ReAct loop, against one session's state. See :meth:`Session.run`."""
        state = self._begin_run(session)
        self._emit_event(
            session,
            "AgentStarted",
            state,
            model=getattr(self.llm, "model", None),
            metadata={
                "input_chars": len(user_input),
                "tools": len(self.tools),
                **self._payload(input=user_input),
            },
        )
        try:
            answer = self._run_loop(session, state, user_input, response_format)
        except Exception as exc:
            self._emit_event(
                session,
                "AgentFailed",
                state,
                duration=state.elapsed,
                usage=dict(state.usage),
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        self._emit_event(
            session,
            "AgentFinished",
            state,
            duration=state.elapsed,
            usage=dict(state.usage),
            metadata={
                "iterations": state.iterations,
                "tool_calls": state.tool_calls,
                **self._payload(answer=str(answer)),
            },
        )
        return answer

    def _run_loop(
        self,
        session: Session,
        state: RunState,
        user_input: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """The loop itself. See :meth:`_run`, which reports around it."""
        session.memory.add("user", self._augment_with_rag(user_input))
        tool_list = list(self.tools.values())
        answer: Optional[str] = None
        for i in range(self._iteration_limit(state)):
            state.iterations = i + 1
            state.check_before_call()
            self._emit(session, "on_iteration", i)
            self._emit_event(session, "AgentIteration", state, metadata={"iteration": i})
            # JSON mode is only requested when no tools are in play; combining
            # tool-calling with JSON mode is unreliable across providers.
            fmt = response_format if not tool_list else None
            result = self._chat(
                session,
                self._build_messages(session, fmt),
                tools=tool_list or None,
                response_format=fmt,
            )
            if not result["tool_calls"]:
                answer = result["content"]
                session.memory.add("assistant", answer)
                break
            session.memory.add("assistant", result["content"], tool_calls=result["tool_calls"])
            self._execute_calls(session, result["tool_calls"], state)
        if answer is None:  # exhausted iterations - force a final answer
            answer = self._chat(
                session,
                self._build_messages(session, response_format),
                response_format=response_format,
            )["content"]
            session.memory.add("assistant", answer)
        if response_format is not None:
            if tool_list:  # dedicated formatting pass so output matches the schema
                answer = self._chat(
                    session,
                    self._build_messages(session, response_format),
                    response_format=response_format,
                )["content"]
            parsed = self._parse_structured(session, answer, response_format)
            self._emit(session, "on_finish", parsed)
            return parsed
        self._emit(session, "on_finish", answer)
        return answer

    def _stream(self, session: Session, user_input: str) -> Iterator[str]:
        """Stream the final answer token by token, against one session.

        Any tool calls are resolved first (non-streaming); the final assistant
        reply is then streamed. With no tools, the reply is streamed directly.
        """
        state = self._begin_run(session)
        self._emit_event(
            session,
            "AgentStarted",
            state,
            model=getattr(self.llm, "model", None),
            metadata={
                "streaming": True,
                "input_chars": len(user_input),
                **self._payload(input=user_input),
            },
        )
        session.memory.add("user", self._augment_with_rag(user_input))
        tool_list = list(self.tools.values())
        if tool_list:
            for i in range(self._iteration_limit(state)):
                state.iterations = i + 1
                state.check_before_call()
                self._emit(session, "on_iteration", i)
                self._emit_event(session, "AgentIteration", state, metadata={"iteration": i})
                result = self._chat(session, self._build_messages(session, None), tools=tool_list)
                if not result["tool_calls"]:
                    break
                session.memory.add("assistant", result["content"], tool_calls=result["tool_calls"])
                self._execute_calls(session, result["tool_calls"], state)
        chunks: List[str] = []
        try:
            for chunk in self.llm.stream(self._build_messages(session, None)):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._emit_event(
                session,
                "AgentFailed",
                state,
                duration=state.elapsed,
                usage=dict(state.usage),
                metadata={"streaming": True, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        answer = "".join(chunks)
        session.memory.add("assistant", answer)
        self._emit(session, "on_finish", answer)
        self._emit_event(
            session,
            "AgentFinished",
            state,
            duration=state.elapsed,
            usage=dict(state.usage),
            metadata={
                "streaming": True,
                "iterations": state.iterations,
                "chunks": len(chunks),
                **self._payload(answer=answer),
            },
        )

    # -- observability --
    def subscribe(self, handler: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Receive every :class:`AgentEvent` from this agent. Returns an unsubscribe.

        The one-liner form of attaching a :class:`Callback` with an
        ``on_event`` method::

            stop = agent.subscribe(lambda e: log.info("%s", e))
            ...
            stop()

        Agent-level subscribers see every session. For one conversation only,
        pass a Callback to ``agent.session(callbacks=[...])``.
        """
        listener = _EventListener(handler)
        self.callbacks.append(listener)

        def unsubscribe() -> None:
            if listener in self.callbacks:
                self.callbacks.remove(listener)

        return unsubscribe

    def _emit_event(
        self,
        session: Session,
        event_type: str,
        state: Optional[RunState] = None,
        duration: Optional[float] = None,
        model: Optional[str] = None,
        tool: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build and dispatch one AgentEvent, if anyone is listening.

        The early return matters: with no callbacks attached - the common
        case - instrumentation costs one attribute check per event rather
        than building a record nobody reads.
        """
        if not self.callbacks and not session.callbacks:
            return
        self._emit(
            session,
            "on_event",
            AgentEvent(
                event_type=event_type,
                run_id=state.id if state is not None else "",
                session_id=session.id,
                agent=self.name,
                timestamp=time.time(),
                duration=duration,
                model=model,
                tool=tool,
                tool_call_id=tool_call_id,
                usage=usage,
                metadata=metadata or {},  # AgentEvent copies it
            ),
        )

    def _payload(self, **fields: Any) -> Dict[str, Any]:
        """Content for an event's metadata, included only when opted in."""
        return dict(fields) if self.event_payloads else {}

    # -- budgets --
    def _begin_run(self, session: Session) -> RunState:
        """Open the accounting for one run and publish it on the session.

        A session's own ``budget`` wins over the agent's, so one agent can
        serve callers on different allowances. The state stays on
        ``session.last_run`` after the run ends - including when a budget
        stopped it - so the caller can read what was spent.
        """
        budget = session.budget if session.budget is not None else self.budget
        state = RunState(budget)
        session.last_run = state
        return state

    def _iteration_limit(self, state: RunState) -> int:
        """Iterations allowed: the budget's, else the agent's ``max_iterations``."""
        limit = state.budget.max_iterations
        return self.max_iterations if limit is None else limit

    # -- instrumentation helpers --
    def _chat(
        self,
        session: Session,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Tool]] = None,
        response_format: Optional[Any] = None,
    ) -> Dict[str, Any]:
        state = session.last_run
        model = getattr(self.llm, "model", None)
        self._emit_event(
            session,
            "LLMStarted",
            state,
            model=model,
            metadata={
                "messages": len(messages),
                "tools": len(tools or []),
                **self._payload(messages=messages),
            },
        )
        started = time.monotonic()
        result = self.llm.chat(messages, tools=tools, response_format=response_format)
        self._emit_event(
            session,
            "LLMFinished",
            state,
            duration=time.monotonic() - started,
            model=model,
            usage=dict(result.get("usage") or {}),
            metadata={
                "content_chars": len(result.get("content") or ""),
                "tool_calls": len(result.get("tool_calls") or []),
                **self._payload(content=result.get("content")),
            },
        )
        self._track_usage(session, result.get("usage"))
        if state is not None:
            # getattr: the LLM interface is the chat() contract, not a class.
            # A custom backend need not carry a `model` name, and an unnamed
            # one simply has no pricing entry - which the budget reports as
            # incomplete rather than as zero.
            state.record_llm_call(result.get("usage"), model or "")
        self._emit(session, "on_llm_call", messages, result)
        return result

    @staticmethod
    def _track_usage(session: Session, usage: Optional[Dict[str, Any]]) -> None:
        """Accumulate token usage on the session that spent it."""
        if not usage:
            return
        for key in session.usage:
            session.usage[key] += int(usage.get(key, 0) or 0)

    def _emit(self, session: Session, event: str, *args: Any) -> None:
        """Notify this agent's callbacks, then the session's own.

        Agent-level callbacks see every conversation (a tracer, a metrics
        sink); session-level ones see only theirs, which is what you want for
        anything that accumulates per conversation.
        """
        for cb in (*self.callbacks, *session.callbacks):
            try:
                getattr(cb, event)(*args)
            except Exception:
                if self.strict_callbacks:
                    raise
                # Instrumentation must not break a run: a broken metrics sink
                # should not cost an answer the model already produced.
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

    def _build_messages(
        self, session: Session, schema: Optional[Type[BaseModel]]
    ) -> List[Dict[str, Any]]:
        system = self.system_prompt
        if session.memory.summary:
            system += f"\n\nConversation summary so far:\n{session.memory.summary}"
        if schema is not None:
            system += (
                "\n\nRespond with a single JSON object matching this schema "
                f"(no prose, no code fences):\n{json.dumps(self._json_schema(schema))}"
            )
        return [{"role": "system", "content": system}] + session.memory.get()

    def _execute_calls(
        self,
        session: Session,
        calls: List[Dict[str, Any]],
        state: Optional[RunState] = None,
    ) -> None:
        """Run one turn's tool calls, add each result to memory in order.

        A single call runs inline. Multiple parallel calls (the model asked
        for several tools in the same turn) run concurrently on a thread pool
        - most tools are I/O-bound (HTTP, disk, subprocess), so this cuts
        wall-clock latency for that turn without changing the observed order
        of results.

        A tool marked ``concurrency="exclusive"`` runs on its own: the turn is
        split into consecutive groups, in the order the model asked for them,
        and an exclusive call is a group of one. See :meth:`_schedule`.

        The pool is capped at ``max_tool_workers``. How many calls arrive in
        a turn is decided by the model, not by the application, so sizing the
        pool to the request let one response spawn a thread per call - a
        resource limit that must be enforced here, in Python, and not by
        asking the model to request fewer tools. Excess calls queue and still
        run; only their concurrency is bounded, and results keep their order.
        """
        if len(calls) == 1:
            observations = [self._execute(session, calls[0], state)]
        else:
            observations = []
            for group in self._schedule(calls):
                if len(group) == 1:
                    observations.append(self._execute(session, group[0], state))
                    continue
                workers = min(len(group), self.max_tool_workers)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    observations.extend(
                        pool.map(partial(self._execute, session), group, [state] * len(group))
                    )
        for call, observation in zip(calls, observations):
            self._emit(
                session, "on_tool_call", call["name"], call.get("arguments", {}), observation
            )
            session.memory.add(
                "tool",
                observation,
                tool_call_id=call.get("id") or call["name"],
                name=call["name"],
            )

    def _schedule(self, calls: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split a turn's calls into groups that may run together.

        Calls stay in the order the model asked for them. Consecutive
        parallel ones form a group and run concurrently; an exclusive one is
        a group of its own, so nothing else from the turn runs while it does.
        A turn of only parallel tools is a single group - exactly the
        behaviour every tool had before this existed.

        That order is preserved because reordering would be surprising, not
        because it means anything. **The model is not expressing a
        dependency.** It emits a list of calls it wants; nothing in the
        protocol says the second depends on the first, and the framework
        cannot infer that it does. If two tools must run in a particular
        order, or must not overlap with a *specific* other tool, that is a
        relationship only you know - express it in the tools themselves (one
        tool that does both steps, or a lock inside them), not by hoping the
        scheduler guesses.

        The guarantee is also per turn. Two exclusive calls in the same turn
        never overlap; the same tool called from two concurrent sessions
        still can, because those are different runs. For process-wide
        exclusion, take a lock inside the tool.
        """
        groups: List[List[Dict[str, Any]]] = []
        batch: List[Dict[str, Any]] = []
        for call in calls:
            tool_obj = self.tools.get(call.get("name", ""))
            # An unknown name is scheduled as parallel: it is refused in
            # _execute long before anything runs, so it cannot race.
            if tool_obj is not None and tool_obj.concurrency == "exclusive":
                if batch:
                    groups.append(batch)
                    batch = []
                groups.append([call])
            else:
                batch.append(call)
        if batch:
            groups.append(batch)
        return groups

    def _execute(
        self, session: Session, call: Dict[str, Any], state: Optional[RunState] = None
    ) -> str:
        """Authorize, then run, one model-requested tool call.

        This is the only path from model output to a tool function, and it is
        a fixed sequence: locate the tool, validate the arguments, ask the
        policy, get approval if the policy wants it, execute, and record the
        decision. There is no branch that reaches ``tool.run`` without
        passing the policy first, and the model cannot choose a different
        route through it.

        A refusal at any step becomes an observation string, so the model
        learns it was refused and the loop continues; a denied tool is never
        a way to crash the run. Application code calling ``tool.run(...)``
        Application code calling ``tool.run(...)``
        directly is trusted and deliberately not policed - this boundary is
        for model intent.

        The run's budget is claimed here, before the tool is located or
        authorized, because this is the only path a model-requested call
        takes - so there is no tool call that does not count against it. A
        budget that is out raises rather than returning an observation: the
        run must stop, not be told about it and carry on spending.
        """
        if state is not None:
            state.check_before_call()
            state.reserve_tool_call(call.get("name", ""))
        name = call.get("name", "")
        arguments = call.get("arguments", {})
        tool_obj = self.tools.get(name)
        if tool_obj is None:
            # A hallucinated or out-of-scope name never reaches a function.
            self._audit(session, name, arguments, "unknown_tool", "not in this agent's tool set")
            self._tool_failed(session, state, call, "unknown_tool", "not in this agent's tool set")
            return f"Error: unknown tool '{name}'."
        # The policy sees who is asking, not just what for: session metadata
        # is how a policy authorizes per user rather than per agent.
        context = {
            "agent": self.name,
            "tool": name,
            "call_id": call.get("id"),
            "session": session.id,
            "metadata": session.metadata,
        }
        try:
            arguments = tool_obj.validate_arguments(arguments)
            self.policy.authorize(tool_obj, arguments, context)
            decision = "allowed"
            if self.policy.requires_approval(tool_obj, arguments, context):
                self._request_approval(session, tool_obj, arguments, context)
                decision = "approved"
        except ToolArgumentValidationError as exc:
            self._audit(session, name, arguments, "invalid_arguments", str(exc), tool_obj)
            self._tool_failed(session, state, call, "invalid_arguments", str(exc))
            return f"Error: {exc}"
        except ToolApprovalRequired as exc:
            self._audit(session, name, arguments, "approval_denied", str(exc), tool_obj)
            self._tool_failed(session, state, call, "approval_denied", str(exc))
            return f"Error: {exc}"
        except ToolAuthorizationError as exc:
            self._audit(session, name, arguments, "denied", str(exc), tool_obj)
            self._tool_failed(session, state, call, "denied", str(exc))
            return f"Error: {exc}"
        except Exception as exc:  # a policy that breaks must not fail open
            logger.exception("policy raised while authorizing '%s'", name)
            self._audit(session, name, arguments, "denied", f"policy error: {exc}", tool_obj)
            self._tool_failed(session, state, call, "denied", f"policy error: {exc}")
            return f"Error: tool '{name}' was not authorized (policy error)."
        self._audit(session, name, arguments, decision, "", tool_obj)
        self._emit_event(
            session,
            "ToolStarted",
            state,
            tool=name,
            tool_call_id=call.get("id"),
            metadata={
                "side_effects": tool_obj.side_effects,
                **self._payload(arguments=arguments),
            },
        )
        started = time.monotonic()
        failure = ""
        try:
            output = str(self._invoke(tool_obj, arguments))
        except ToolTimeoutError as exc:
            logger.warning("%s", exc)
            self._tool_failed(session, state, call, "timeout", str(exc), time.monotonic() - started)
            return f"Error: {exc}"
        except Exception as exc:  # a tool must never crash the loop
            # Bounded too: an exception message can be as large as a result.
            failure = f"{type(exc).__name__}: {exc}"
            output = f"Error executing '{name}': {exc}"
        observation = self._bound_output(tool_obj, output)
        elapsed = time.monotonic() - started
        if failure:
            self._tool_failed(session, state, call, "raised", failure, elapsed)
        else:
            self._emit_event(
                session,
                "ToolFinished",
                state,
                duration=elapsed,
                tool=name,
                tool_call_id=call.get("id"),
                metadata={
                    "output_chars": len(observation),
                    **self._payload(result=observation),
                },
            )
        if state is not None:
            state.record_tool_output(len(observation))
        return observation

    def _output_limit_for(self, tool_obj: Tool) -> Optional[int]:
        """Characters of output to keep: the tool's own setting, else the agent's."""
        if tool_obj.max_output_size is not None:
            return tool_obj.max_output_size
        return self.max_tool_output_size

    def _bound_output(self, tool_obj: Tool, text: str) -> str:
        """Shorten a tool result to its budget, before it reaches the context.

        This runs inside :meth:`_execute`, so the bound is applied before the
        observation is added to memory, sent to a provider, or shown to a
        callback - there is no path where the full text reaches the model.

        The budget counts **characters, not bytes**. Python strings are
        sequences of code points, so slicing can never split one and produce
        invalid text; a multi-character emoji sequence can be split, which is
        cosmetic. Characters also compose with ``Memory(max_tokens=...)``,
        which estimates tokens the same way.

        Nothing is shortened silently. An oversized result keeps its first
        ``limit`` characters and gains a :class:`ToolOutputTruncated` note
        saying how much was dropped - and, when the full result was valid
        JSON, saying that the fragment is cut mid-structure and will not
        parse, since a model handed a JSON fragment will otherwise try. The
        note is appended on top of the budget rather than counted inside it,
        so ``max_output_size`` bounds the tool's own text.

        This is a **context and cost boundary, not a security control**. It
        limits what a tool sends onward; it does not stop a tool reading or
        computing anything, and a secret inside the retained prefix is
        retained. Do not use it to contain a hostile tool.
        """
        limit = self._output_limit_for(tool_obj)
        if limit is None or len(text) <= limit:
            return text
        indicator = ToolOutputTruncated(
            tool=tool_obj.name,
            original_size=len(text),
            limit=limit,
            was_json=self._looks_like_json(text),
        )
        logger.warning(
            "tool %s: output truncated to %d of %d characters",
            tool_obj.name,
            limit,
            indicator.original_size,
        )
        return text[:limit] + str(indicator)

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        """Whether the untruncated text parsed as JSON.

        Only asked on the truncation path, and only for text that opens like
        JSON, so a large result is not parsed for nothing.
        """
        if text[:512].lstrip()[:1] not in ("{", "["):
            return False
        try:
            json.loads(text)
        except (ValueError, RecursionError):
            return False
        return True

    def _tool_failed(
        self,
        session: Session,
        state: Optional[RunState],
        call: Dict[str, Any],
        reason: str,
        detail: str,
        duration: Optional[float] = None,
    ) -> None:
        """Report a tool call that produced no result.

        ``reason`` names the stage that refused it - ``unknown_tool``,
        ``invalid_arguments``, ``denied``, ``approval_denied``, ``timeout`` or
        ``raised`` - so a metrics sink can separate "the tool broke" from "the
        tool was not allowed to run". These arrive without a preceding
        ``ToolStarted`` whenever the call never began.
        """
        self._emit_event(
            session,
            "ToolFailed",
            state,
            duration=duration,
            tool=call.get("name", ""),
            tool_call_id=call.get("id"),
            metadata={"reason": reason, "error": detail},
        )

    def _timeout_for(self, tool_obj: Tool) -> Optional[float]:
        """Seconds to wait for this tool: its own setting, else the agent's."""
        return tool_obj.timeout if tool_obj.timeout is not None else self.tool_timeout

    def _invoke(self, tool_obj: Tool, arguments: Dict[str, Any]) -> Any:
        """Run one tool, bounded by its timeout if it has one.

        **The timeout bounds the wait, not the work.** Python cannot cancel a
        running thread, so when a call overruns, the agent stops waiting and
        reports a :class:`ToolTimeoutError` while the tool keeps running in
        the background until it returns on its own. Consequences worth
        knowing before relying on this:

        * A tool with side effects may still complete after the timeout. On
          timeout you do not know whether the effect happened - treat it as
          unknown, not as failed.
        * The orphaned thread is not reclaimed until the call ends, and
          because executor threads are non-daemon it can delay interpreter
          exit. A tool that hangs forever holds a thread forever.
        * Hard cancellation of arbitrary Python needs process isolation - run
          the work in a subprocess and kill it, as ``examples/coder.py``
          does. A thread timeout is a liveness guard for the agent loop, not
          a containment boundary.
        * A timeout does not abort a socket read either, so HTTP tools still
          need their own network timeout. ``requests.get(...)`` without
          ``timeout=`` can block for a very long time; the tool timeout will
          free the agent but leave the request running.

        Without a timeout the call runs inline, exactly as before - no thread,
        no executor, and an unbounded tool blocks the agent.
        """
        timeout = self._timeout_for(tool_obj)
        if timeout is None:
            return tool_obj.run(arguments)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool_obj.name}")
        try:
            future = pool.submit(tool_obj.run, arguments)
            try:
                return future.result(timeout=timeout)
            except _FutureTimeout:
                raise ToolTimeoutError(
                    f"tool '{tool_obj.name}' did not finish within {timeout}s and was "
                    "abandoned; it may still be running, so treat any side effect as "
                    "unknown rather than as not having happened"
                ) from None
        finally:
            # wait=False is the whole point: the default shutdown(wait=True)
            # would block on the very call we just gave up on, re-creating
            # the hang this exists to prevent. A finished call's worker exits
            # promptly and nothing leaks; an overrunning one keeps its thread
            # because Python has no way to take it back.
            pool.shutdown(wait=False)

    def _request_approval(
        self,
        session: Session,
        tool_obj: Tool,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Ask the application to confirm one call, or raise ToolApprovalRequired.

        With no ``approve`` callback configured the call is refused: a tool
        marked ``requires_approval`` must not run merely because nobody wired
        up an approver. An approver that raises is likewise a refusal (it is
        caught upstream in _execute), never a pass.
        """
        if self.approve is None:
            raise ToolApprovalRequired(
                f"tool '{tool_obj.name}' requires approval, but this agent has no "
                "approval callback configured (pass Agent(approve=...))"
            )
        request = {
            "agent": self.name,
            "session": session.id,
            "metadata": session.metadata,
            "tool": tool_obj.name,
            "arguments": dict(arguments),
            "permissions": sorted(tool_obj.permissions),
            "side_effects": tool_obj.side_effects,
            "call_id": context.get("call_id"),
        }
        try:
            with self._approval_lock:
                granted = self.approve(request)
        except Exception as exc:  # an approver that breaks refuses, never passes
            logger.exception("approval callback failed for '%s'", tool_obj.name)
            raise ToolApprovalRequired(
                f"tool '{tool_obj.name}' could not be approved: the approval "
                f"callback raised {type(exc).__name__}"
            ) from exc
        if not granted:
            raise ToolApprovalRequired(f"tool '{tool_obj.name}' was not approved")

    def _audit(
        self,
        session: Session,
        name: str,
        arguments: Any,
        decision: str,
        reason: str = "",
        tool_obj: Optional[Tool] = None,
    ) -> None:
        """Record one authorization decision, before the tool runs.

        The event carries the session id, so a decision is still traceable to
        one conversation when many run concurrently.
        """
        event = {
            "agent": self.name,
            "session": session.id,
            "tool": name,
            "arguments": arguments,
            "decision": decision,
            "reason": reason,
            "permissions": sorted(tool_obj.permissions) if tool_obj else [],
            "side_effects": bool(tool_obj and tool_obj.side_effects),
        }
        if decision in ("allowed", "approved"):
            logger.debug("tool %s: %s", name, decision)
        else:
            # Refusals are logged by the core as well as emitted, so there is
            # a record even when no callback is attached.
            logger.warning("tool %s: %s (%s)", name, decision, reason)
        self._emit(session, "on_tool_audit", event)

    @staticmethod
    def _json_schema(schema: Type[BaseModel]) -> Dict[str, Any]:
        return _pydantic_schema(schema)

    def _parse_structured(self, session: Session, content: str, schema: Type[BaseModel]):
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
                    session,
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
        """Best-effort parse of a model reply into a JSON object.

        Always a dict: a reply of ``null`` or ``[1, 2]`` is valid JSON but not
        an object, and returning it would hand a non-mapping to
        ``schema(**data)`` further down, where it fails as a TypeError rather
        than as the ValidationError the repair loop expects.
        """
        value, found = _extract_json(content)
        return value if found and isinstance(value, dict) else {}


# --- 9. Session (one conversation's state) ---------------------------------
class Session:
    """One conversation with an :class:`Agent`: its memory, usage and metadata.

    An Agent is configuration and behaviour - safe to build once and share
    across users, requests and threads. A Session is the mutable state of a
    single conversation, so each one gets its own::

        agent = Agent(llm, tools=[...])
        alice = agent.session(metadata={"user": "alice"})
        bob = agent.session(metadata={"user": "bob"})

        alice.run("my name is Alice")
        bob.run("what is my name?")     # cannot see Alice's history

    Sessions are isolated by construction rather than by locking: each owns
    its own :class:`Memory` and usage counters, and no conversational state
    lives on the Agent, so two sessions have nothing to contend over. They
    can therefore run concurrently - subject to the resources you chose to
    share between them (``llm``, ``rag``, Agent-level callbacks; see the
    Agent docstring).

    A single Session is *not* itself concurrency-safe: it is one
    conversation, and two threads adding to the same history interleave. One
    session per conversation, not per request to the same conversation.

    Create these with :meth:`Agent.session`; constructing one directly is
    fine but you must supply the memory yourself.
    """

    def __init__(
        self,
        agent: Agent,
        memory: Optional[Memory] = None,
        callbacks: Optional[List[Callback]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        budget: Optional[Budget] = None,
    ):
        self.agent = agent
        self.memory = memory if memory is not None else Memory()
        # Copied, not aliased: a caller's dict must not become shared state.
        self.callbacks = list(callbacks or [])
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.id = session_id or uuid.uuid4().hex[:12]
        # Overrides the agent's budget for this conversation only; None means
        # "use the agent's". Set per session so one agent can serve callers on
        # different allowances.
        self.budget = budget
        # Accounting for the most recent run, left in place afterwards - see
        # RunState. None until this session has run once.
        self.last_run: Optional[RunState] = None
        self.usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def run(self, user_input: str, response_format: Optional[Type[BaseModel]] = None) -> Any:
        """Take one turn in this conversation.

        Optionally pass a Pydantic model as ``response_format`` for validated
        structured output. The agent's ReAct loop does the work; this session
        supplies the history it reads and the counters it updates.
        """
        return self.agent._run(self, user_input, response_format)

    def stream(self, user_input: str) -> Iterator[str]:
        """Stream this turn's answer token by token. See :meth:`run`."""
        return self.agent._stream(self, user_input)

    async def arun(self, user_input: str, response_format: Optional[Type[BaseModel]] = None) -> Any:
        """Await one turn off the event loop.

        No separate async HTTP stack: this offloads the blocking turn to a
        worker thread with ``asyncio.to_thread``. Because sessions are
        independent, awaiting several at once is safe - which is the point of
        having them in an async server.
        """
        return await asyncio.to_thread(self.run, user_input, response_format)

    def reset(self) -> None:
        """Start this conversation over: clear memory, summary and counters."""
        self.memory.clear()
        for key in self.usage:
            self.usage[key] = 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Session {self.id} of {self.agent.name}: {len(self.memory.get())} messages>"


# --- 10. Router (multi-agent orchestration) --------------------------------
class Router:
    """Coordinate several agents: route to one, run all, or run all and fuse.

    :meth:`route` **fails closed**. A routing decision is only valid if it
    names exactly one registered agent; an empty, evasive, hallucinated,
    malformed or ambiguous reply raises :class:`RoutingError` rather than
    dispatching to an agent nobody chose. Pass ``fallback=`` to nominate an
    agent for those queries - explicitly, in your code.

    The decision is asked for as JSON and then **checked against the agent
    registry in Python**. The prompt is where the model is told what is
    allowed; the registry lookup is what enforces it. Nothing the model
    writes can name an agent this Router does not hold.

    ``strict=True`` accepts only a structured reply or a bare exact name,
    refusing prose entirely. The default also accepts a name appearing as a
    contiguous run of words in a longer reply ("the best agent is cost"),
    which small local models often produce - see :meth:`_resolve` for what
    that does and does not catch.

    Every dispatch runs in a **fresh** :class:`Session` per agent, so a
    router can serve concurrent queries without two of them landing in the
    same agent's history. It never touches an agent's default session. Pass
    ``metadata=`` to hand the caller's identity to each session, and so to
    :class:`ToolPolicy`.

    :meth:`run_all` and :meth:`synthesize` do not route - they run every
    agent - so none of this applies to them.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")
    # An agent description is developer-supplied, but it is often built from
    # data that is not. Cap it so a long one cannot crowd out the
    # instruction, and see _descriptions for why newlines are collapsed.
    _MAX_DESCRIPTION = 200

    def __init__(
        self,
        llm: LLM,
        agents: List[Agent],
        synthesizer: Optional[Agent] = None,
        fallback: Optional[Agent] = None,
        strict: bool = False,
    ):
        """``fallback`` receives queries that :meth:`route` cannot resolve.

        Leave it ``None`` (the default) to fail closed with
        :class:`RoutingError` instead of dispatching to an unchosen agent.

        Agent names are validated here rather than at routing time: a name
        that is blank, or that collides with another once case and spacing
        are normalised, makes "resolve to exactly one agent" impossible. Both
        used to be silent - a blank-named agent was simply unreachable
        forever, and one of two identically-named agents always won.
        """
        if not agents:
            raise ValueError("Router needs at least one agent.")
        registry: Dict[str, Agent] = {}
        for position, agent in enumerate(agents):
            key = self._normalise(agent.name)
            if not key:
                raise ValueError(
                    f"The agent at position {position} has a blank name "
                    f"({agent.name!r}), so nothing could ever route to it."
                )
            if key in registry:
                raise ValueError(
                    f"Two agents share the routing name {key!r}. Names must be unique "
                    "once case and surrounding whitespace are normalised, or a routing "
                    "decision cannot identify one agent."
                )
            registry[key] = agent
        self.llm, self.agents, self.synthesizer = llm, agents, synthesizer
        self.fallback = fallback
        self.strict = strict
        self._registry = registry
        self._decision_model = self._build_decision_model()

    # -- the registry is the authority ------------------------------------
    @staticmethod
    def _normalise(text: Any) -> str:
        """Casefold, strip, and collapse internal whitespace to one space."""
        return " ".join(str(text or "").casefold().split())

    def _tokens(self, text: Any) -> List[str]:
        return self._TOKEN_RE.findall(self._normalise(text))

    def _build_decision_model(self) -> Optional[Any]:
        """A Pydantic model whose ``agent`` field is one of the known names.

        This is what makes the *request* structured: the allowed values come
        from the registry, so the model is shown a closed set rather than
        asked to invent a name. It is not the enforcement - :meth:`_resolve`
        looks the answer up in the registry regardless.
        """
        if not _HAS_PYDANTIC_V2:  # pragma: no cover - v2 is the pinned floor
            return None
        names = tuple(sorted(self._registry))
        try:
            return create_model("RouterDecision", agent=(Literal[names], ...))
        except Exception:  # pragma: no cover - exotic agent names only
            return None

    def _descriptions(self) -> str:
        """One line per agent, for the router prompt.

        Whitespace inside a description is collapsed and the text is capped.
        A description containing newlines could otherwise forge extra "- name:"
        lines in this list, presenting agents that do not exist or attaching
        instructions to one that does. Collapsing removes that shape; the
        registry check removes its effect.
        """
        lines = []
        for agent in self.agents:
            description = " ".join(str(agent.description or "").split())
            if len(description) > self._MAX_DESCRIPTION:
                description = description[: self._MAX_DESCRIPTION - 3] + "..."
            lines.append(f"- {self._normalise(agent.name)}: {description}")
        return "\n".join(lines)

    def route(self, query: str) -> Agent:
        """Ask the LLM which single agent fits best and return it.

        Raises :class:`RoutingError` if the answer does not identify exactly
        one registered agent and no ``fallback`` was configured.
        """
        names = ", ".join(sorted(self._registry))
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a router. Choose the single best agent for the user's "
                    'query. Reply with ONLY a JSON object: {"agent": "<name>"}. '
                    f"The value must be exactly one of: {names}. "
                    'If none of them fits, reply {"agent": null}.'
                ),
            },
            {
                "role": "user",
                "content": f"Agents:\n{self._descriptions()}\n\nQuery: {query}",
            },
        ]
        reply = self.llm.chat(prompt, response_format=self._decision_model)
        return self._match(reply["content"])

    def _match(self, choice: str) -> Agent:
        """Resolve a router reply to one agent, or fail closed."""
        agent = self._resolve(choice)
        if agent is not None:
            return agent
        if self.fallback is not None:
            return self.fallback
        shown = self._normalise(choice)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise RoutingError(
            f"could not identify exactly one agent from the model's reply {shown!r}. "
            f"Known agents: {sorted(self._registry)}. Pass Router(..., fallback=agent) "
            "to handle unroutable queries explicitly."
        )

    def _resolve(self, choice: str) -> Optional[Agent]:
        """Return the one agent this reply names, or None. Never guesses.

        A reply carrying JSON is judged **only** as a structured decision: it
        must be an object with an ``agent`` field naming a registered agent.
        It is never rescanned as prose, because salvaging one would invert its
        meaning - ``{"rejected": "cost"}`` mentions exactly one agent, and
        scanning it for names would route to the agent the model just ruled
        out.

        A reply carrying no JSON is matched as text, exactly:

        1. **Bare name** - the whole reply, normalised, *is* a known name.
        2. **Embedded name** - the name appears as a contiguous run of words
           inside a longer reply, and exactly one agent's does. Skipped under
           ``strict=True``.

        Anything else returns None: an empty or whitespace reply, a refusal,
        an unregistered name, unparseable JSON, or a reply naming two agents.
        Ambiguity is not broken by preference or order - two matches is a
        failure, not a contest.

        Rule 2 is exact containment, not fuzzy matching: ``fit`` does not
        match "profit", and a two-word name does not match a reply using both
        words apart ("admin ... delete"). What it cannot see is *sense*: prose
        mentioning one agent in order to reject it ("not cost") reads as
        choosing it. The structured path has no such gap, so set
        ``strict=True`` where that matters.
        """
        value, found = _extract_json(choice)
        if found:
            name = value.get("agent") if isinstance(value, dict) else None
            return self._registry.get(self._normalise(name)) if isinstance(name, str) else None
        agent = self._registry.get(self._normalise(choice))
        if agent is not None:
            return agent
        if self.strict:
            return None
        words = self._tokens(choice)
        if not words:
            return None
        matches = [
            candidate
            for key, candidate in self._registry.items()
            if self._contains_run(words, self._TOKEN_RE.findall(key))
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _contains_run(words: List[str], name: List[str]) -> bool:
        """True if ``name`` appears in ``words`` as consecutive whole words."""
        if not name or len(name) > len(words):
            return False
        return any(words[i : i + len(name)] == name for i in range(len(words) - len(name) + 1))

    def run(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Route the query to one agent and run it in a fresh session."""
        return self.route(query).session(metadata=metadata).run(query)

    def run_all(
        self,
        query: str,
        parallel: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run every agent (in parallel by default) and collect their results.

        Each agent runs in its own new session. That is what makes the
        parallel path safe: the agents share no conversation state, and two
        concurrent ``run_all`` calls cannot interleave in one agent's
        history. Nothing is retained afterwards - these are one-shot
        conversations, so the answers do not accumulate anywhere.
        """
        results: Dict[str, Any] = {}
        work = {a: partial(a.session(metadata=metadata).run, query) for a in self.agents}
        if parallel and len(self.agents) > 1:
            with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
                futures = {pool.submit(fn): agent for agent, fn in work.items()}
                for future, agent in futures.items():
                    results[agent.name] = self._safe(future.result)
        else:
            for agent, fn in work.items():
                results[agent.name] = self._safe(fn)
        return results

    @staticmethod
    def _safe(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # one failing agent shouldn't sink the rest
            return f"Error: {exc}"

    def synthesize(
        self,
        query: str,
        parallel: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run all agents, then fuse their findings with the synthesizer.

        The synthesizer also gets a fresh session, for the same reason.
        """
        results = self.run_all(query, parallel=parallel, metadata=metadata)
        combined = "\n\n".join(f"### {n}\n{r}" for n, r in results.items())
        if self.synthesizer is None:
            return combined
        return self.synthesizer.session(metadata=metadata).run(
            f"Original query: {query}\n\nFindings from specialist agents:\n{combined}"
            "\n\nSynthesize these into a single, well-reasoned final answer."
        )
