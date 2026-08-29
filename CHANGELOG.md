# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Runtime argument validation before every tool execution.** Type
  annotations previously only generated the JSON schema shown to the model;
  what actually arrived reached the function after structural checks only. A
  model sending `{"degrees": "hot"}` for an `int` parameter got a `TypeError`
  from inside the tool, if it failed at all.

  Arguments are now validated against a Pydantic model built from the Python
  **signature** — deliberately not from `Tool.schema`, since enforcing the
  same document the model is free to ignore would guarantee nothing. Every
  annotation the schema builder understands is enforced: `str`, `int`,
  `float`, `bool`, lists and dicts including item types, `Optional`,
  `Literal`, `Enum`, and Pydantic models nested to any depth. Rejected:
  missing required arguments, wrong types, malformed nested structures, and
  unknown argument names (unless the tool declares `**kwargs`).
  - Validation runs in `Tool.run()`, so no model-facing path can skip it.
    `Agent` validates first, so a `ToolPolicy` now sees normalised,
    correctly-typed arguments.
  - Failures come back to the model as an observation naming the field and
    the rule (`person.address.zip: Input should be a valid integer`), so it
    can correct itself; the tool does not run.
  - Errors never repeat the offending **value**. Pydantic's own message
    embeds `input_value=...`, which would reach the model, conversation
    memory and the audit log; only the field path and rule are reported, and
    the exception is raised outside the `except` block so neither
    `__cause__` nor `__context__` retains the original.
  - No new dependency: Pydantic was already required.

### Changed
- **Arguments are now normalised**, reversing an earlier deliberate choice not
  to coerce. A function receives what its annotations promise: `"42"` arrives
  as `42` for an `int`, and a nested dict arrives as the declared model.
  - **An `Enum`-annotated parameter now receives the enum member** (`Color.red`)
    rather than the raw value (`"red"`). Tools annotated with an `Enum` that
    assumed a string need `.value`.
  - A parameter with **no annotation** accepts anything, matching the plain
    Python function. The schema still advertises it as a string.
- `Tool.run(dict)` validates; `Tool.__call__` (`my_tool(1, 2)`) still calls
  straight through, unvalidated — it is your code calling your function.
- Tools with parameters named `model_name`, `json` or `schema` no longer emit
  Pydantic shadowing warnings at decoration time.
- **`Session`: agent configuration is now separate from conversation state.**
  An `Agent` holds the LLM, tools, prompt, RAG, callbacks and policy - things
  that are safe to share. A `Session` holds one conversation's memory, usage
  counters and metadata. One agent can therefore serve many users and many
  concurrent requests:

  ```python
  agent = Agent(llm, tools=[...])  # build once, share freely
  alice = agent.session(metadata={"user": "alice"})
  bob = agent.session()
  ```

  The isolation is structural rather than lock-based: each session owns its
  own `Memory` and counters, and no conversational state remains on the Agent,
  so two sessions have nothing to contend over.
  - `Agent.session(memory=, callbacks=, metadata=, session_id=)` starts an
    independent conversation. `metadata` reaches `ToolPolicy` hooks as
    `context["metadata"]` and appears in audit events and approval requests,
    which is how a policy authorizes per user rather than per agent.
  - `Agent.memory_factory` (default `Memory`) supplies each new session's
    memory. `Agent(memory=<instance>)` still works and now seeds the *default
    session only* - it is deliberately not shared with `agent.session()`.
  - `Session.run/stream/arun/reset`, and `Agent.reset()` for the default one.
  - Session-level callbacks fire only for that conversation; Agent-level
    callbacks still see every session.
  - Audit events and approval requests now carry the session id.

### Changed
- `agent.run()`, `agent.stream()` and `agent.arun()` are unchanged, and
  `agent.memory` / `agent.usage` still work: they now address a **persistent
  default session**, created on first use and reused for the life of the
  agent. Because it persists, `agent.run()` remains a single-conversation
  API; serving several users means one session each.
- `agent.usage` is per conversation. There is deliberately no Agent-wide
  total, which would reintroduce the shared mutable state this change
  removes; sum the sessions you care about.
- **`Router` now runs every dispatch in a fresh session per agent**, so
  concurrent `run_all` / `synthesize` calls no longer interleave in one
  agent's history, and routing never touches an agent's default session.
  Results are returned as before but are no longer retained in agent memory.
  `run`, `run_all` and `synthesize` accept `metadata=` to pass the caller's
  identity through to each session and its policy.
- **Tool authorization layer.** A model requesting a tool is now a request
  that is granted or refused in Python, before the function runs - not a
  prompt asking the model to behave.
  - `@tool` accepts optional security metadata: `permissions`,
    `requires_approval`, `side_effects`, and an `allowed(arguments, context)`
    hook for per-call rules. `@tool` bare is unchanged. None of this metadata
    is sent to the model.
  - `ToolPolicy` decides: `authorize()` (raise `ToolAuthorizationError` to
    deny) and `requires_approval()`. It is also the default policy, and the
    default is permissive - an agent with no `policy=` behaves exactly as it
    did before, except that a tool marked `requires_approval` is now gated
    rather than decorative.
  - `PermissionPolicy(granted=..., approval_for=...)` allows only tools whose
    declared permissions have all been granted.
  - `Agent(policy=..., approve=...)`. The approval callback is supplied by the
    application and is unreachable from model output; it is serialised with a
    lock so a turn's concurrent tool calls cannot re-enter your prompt.
  - `Tool.validate_arguments()` rejects non-mappings, non-string argument
    names, unknown parameter names and missing required parameters before the
    call. Structural only - types are not coerced.
  - `Callback.on_tool_audit(event)` records every decision before execution.
  - New exceptions: `ToolAuthorizationError`, `ToolApprovalRequired`,
    `ToolArgumentValidationError`.
  - Fail-closed throughout: a policy that raises denies, an approval callback
    that raises is a refusal, and a tool needing approval with no approver
    configured does not run.
  - See `examples/policy.py` and the SECURITY.md boundary list.

### Fixed
- **Memory compression no longer produces an unsendable window.** When the
  sliding window overflowed, the boundary could fall between an assistant
  message carrying `tool_calls` and the `tool` messages answering them,
  leaving the window starting with an orphaned tool result. Providers reject
  that outright (OpenAI: HTTP 400, "messages with role 'tool' must be a
  response to a preceding message with 'tool_calls'"; Anthropic rejects the
  equivalent `tool_result` block), so any sufficiently long tool-using
  conversation eventually failed. The boundary now moves back to the start of
  the tool group.
- **`RAG.add_many()` no longer loses documents silently.** Passing `metadatas`
  of a different length to `texts` used to `zip()` down to the shorter list:
  in TF-IDF mode documents simply vanished, and with an `embed_fn` it left
  more embeddings than documents, so `search()` raised `IndexError` later,
  far from the cause. It now raises `ValueError` at the call site.
- **No placeholder credential is sent when no API key is configured.**
  `provider="openai"` with no key sent the literal header
  `Authorization: Bearer None`; Anthropic sent `x-api-key: `. Both headers are
  now omitted entirely, which is also what local OpenAI-compatible servers
  (vLLM, LM Studio, llama.cpp) expect. Affects both `chat()` and `stream()`.

### Changed
- **BREAKING - `Router.route()` now fails closed.** It previously fell back to
  `agents[0]` whenever the model's reply didn't match an agent, and matched by
  substring in both directions - so an empty reply matched *every* agent (and
  returned the longest-named one), and an agent named `fit` matched a reply
  mentioning "profit". A query could therefore be dispatched to an agent
  nobody chose, which matters because agents differ in the tools, and so the
  privileges, they carry. Matching is now exact-name-first, then whole-word,
  and must resolve to exactly one agent; an empty, evasive, hallucinated or
  ambiguous reply raises the new `RoutingError`.

  To restore a default destination, name it explicitly:

  ```python
  Router(llm, agents=[...], fallback=triage_agent)
  ```
- **BREAKING - `RAG.add_many()` raises `ValueError`** on a `texts`/`metadatas`
  length mismatch instead of silently truncating (see Fixed, above).
- **Tool fan-out is bounded.** `Agent` sized its thread pool to the number of
  tool calls in a turn, but that count is chosen by the model, so one response
  could spawn a thread per call (250 calls produced ~176 live threads). The
  pool is now capped by the new `Agent(max_tool_workers=8)`. Excess calls
  queue and still run, in the same order; only their concurrency is bounded.
- `Memory` may now keep slightly more than `max_messages` (or `max_tokens`)
  rather than split a tool group - a window one group over budget is still
  sendable, whereas one starting with an orphaned tool result is not.

### Added
- `RoutingError`, exported from the package, raised by `Router.route()` when
  no single agent can be identified.
- `Router(fallback=...)` to nominate an agent for unroutable queries.
- `Agent(max_tool_workers=8)` to tune the tool-call concurrency cap.

## [0.4.0] - 2026-08-10

### Added
- **Richer tool schemas**: `@tool` parameters typed as `Literal[...]`, `Enum`
  subclasses, or nested Pydantic `BaseModel`s now produce an accurate JSON
  Schema (`enum` values, or an inlined object schema) instead of falling back
  to a plain string.
- **Async tools and agents**: `@tool` functions may be `async def` (driven by
  `Tool.run` via `asyncio.run`); `Agent.arun()` and `LLM.achat()` offload a
  full turn / request to a worker thread (`asyncio.to_thread`) so they can be
  awaited from async frameworks (FastAPI, aiohttp, ...) without blocking the
  event loop. This does not add a non-blocking async HTTP client - the
  underlying request is still made with `requests`.
- **Concurrent tool-call execution**: when a model requests more than one
  tool call in the same turn, `Agent` now runs them concurrently on a thread
  pool instead of sequentially, then feeds results back in the original
  order. Most tools are I/O-bound, so this cuts wall-clock latency per turn.
- **Token-aware `Memory`**: an optional `max_tokens` argument (rough
  4-chars/token estimate, no tokenizer dependency) additionally shrinks the
  retained window when the message-count cap alone would still risk
  overflowing the model's context window.
- **Per-provider `base_url` environment variables**: `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, `OLLAMA_BASE_URL` are consulted when `base_url` isn't
  passed explicitly - this is what lets `provider="openai"` target any
  OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM, LM Studio,
  ...) purely via configuration.
- `LLM.close()` to explicitly close the instance's HTTP session.

### Changed
- `LLM` now reuses a single `requests.Session` per instance instead of a
  fresh connection per call, so repeated requests reuse the underlying
  TCP/TLS connection.
- `LLM(cache=True)` is now a bounded LRU cache (`cache_size`, default 256)
  with an optional `cache_ttl` in seconds, instead of an unbounded dict -
  long-running processes no longer accumulate cache entries indefinitely.
- Fixed documentation drift: dropped hardcoded, quickly-stale line-count
  claims ("417 lines", "~620 lines") from `docs/ARCHITECTURE.md` and the
  PickMyStack knowledge base in favour of measuring live via
  `benchmarks/compare_frameworks.py`. The root `ARCHITECTURE.md` (an
  unreferenced duplicate of `docs/ARCHITECTURE.md`) is now a short pointer to
  the canonical copy used by the MkDocs site.

### Notes
- Investigated shipping a PEP 561 `py.typed` marker so downstream type
  checkers trust the inline types on an installed `pip install unchained-ai`.
  PEP 561 has no supported mechanism for module-only (`py-modules`)
  distributions to ship it - only package (directory) distributions can, and
  that would require restructuring `unchained.py` into a package, which
  conflicts with the single-file design. Not changed; documented in
  `pyproject.toml`.

## [0.3.0] - 2026-07-02

### Added
- **`MockLLM`**: a deterministic, no-network stand-in for `LLM` so agents run
  with zero setup (fixed reply, scripted responses, or a custom handler).
- **Pluggable embeddings for RAG**: pass `embed_fn` to switch from TF-IDF to
  dense-vector cosine similarity; TF-IDF remains the zero-dependency default.
- **Opt-in response caching**: `LLM(cache=True)` memoizes identical requests.
- `examples/quickstart.py`: a no-API-key tour of the whole framework.
- MkDocs-material documentation site with an auto-generated API reference
  (mkdocstrings) and a GitHub Pages deploy workflow.

### Changed
- `examples/coder.py` now runs generated code in an isolated subprocess with a
  timeout instead of in-process `exec` (still not a full security sandbox).

## [0.2.0] - 2026-07-02

### Added
- **Streaming**: `LLM.stream()` for all three providers and `Agent.stream()`
  to yield the final answer token by token.
- **Reliability**: automatic retry with exponential backoff and jitter on
  connection errors and `429`/`5xx` responses, honouring `Retry-After`
  (`max_retries`, `backoff` on `LLM`).
- **Observability**: a `Callback` base class and ready-made `LoggingCallback`,
  plus a module logger (`logging.getLogger("unchained")`). Agents accept
  `callbacks=[...]` and emit `on_iteration`/`on_llm_call`/`on_tool_call`/`on_finish`.
- **Token usage tracking**: usage is normalised across providers to
  `{prompt_tokens, completion_tokens, total_tokens}` and accumulated on
  `Agent.usage`.
- **Structured-output repair**: on a Pydantic `ValidationError`, the agent
  feeds the error and schema back to the model to self-correct
  (`structured_retries`).
- `examples/sqlite_memory.py`: a persistent `SQLiteMemory` demonstrating the
  `Memory` extension point.
- Developer infrastructure: GitHub Actions CI (lint, type-check, test matrix on
  Python 3.9-3.13), PyPI publish workflow, `ruff`/`mypy`/`pre-commit` config,
  an expanded offline test suite, and contributor scaffolding.

### Changed
- Line-count claims replaced with "single file, two dependencies" after
  adopting `ruff format`.

## [0.1.0] - 2026-07-02

### Added
- Single-file core `unchained.py`: `@tool`/`Tool`, multi-provider `LLM`
  (OpenAI, Anthropic, Ollama), `Memory`, `RAG`, `Agent` (ReAct loop), `Router`,
  and Pydantic structured output.
- Examples: `researcher`, `coder`, `data_analyst`, and the flagship
  multi-agent `pickmystack` app with a CLI and a Streamlit UI.
- Benchmarks, unit tests, and documentation (README, architecture, user manual).

[Unreleased]: https://github.com/NiravRVaghasiya/unchained/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NiravRVaghasiya/unchained/releases/tag/v0.1.0
