# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`LLMResponse`: one normalised reply shape across providers.** `chat()`
  returned a three-key dict - content, tool_calls, usage - and discarded
  everything else the provider said. It now returns an `LLMResponse` carrying
  `finish_reason`, `provider`, `model`, `request_id` and a `metadata` dict as
  well.
  - **`finish_reason` is normalised** onto OpenAI's vocabulary: Anthropic's
    `end_turn`/`tool_use`/`max_tokens` and Ollama's `done_reason` all map
    onto `stop`/`tool_calls`/`length`. An unrecognised reason is passed
    through unchanged rather than forced into a bucket, and the provider's
    own wording is kept in `metadata["raw_finish_reason"]`.
  - `response.truncated` reports the `length` case, which nothing else in a
    reply reveals - an agent used to treat a cut-off answer as a complete one.
  - `model` is the model the provider reports, which is often more specific
    than the one requested (`gpt-4o-mini-2024-07-18` for `gpt-4o-mini`).
  - `request_id` comes from the response headers, falling back to the body's
    own id, for quoting to a provider's support.
  - `metadata` keeps provider-specific detail - OpenAI's
    `system_fingerprint`, Ollama's timings, Anthropic's `stop_sequence`, and
    the `endpoint` that answered, which distinguishes an OpenAI-compatible
    host from OpenAI itself. Nothing in the agent loop reads it.
  - Backwards compatible: `LLMResponse` supports the mapping access the old
    dict had (`response["content"]`, `response.get("usage")`, `in`), so
    existing code and test doubles returning a bare dict keep working. It is
    frozen, so a reply cannot be rewritten after the fact.
  - `MockLLM` produces the same model, so a stand-in cannot pass a test
    against a shape no provider produces.
  - The response cache stores the whole reply; rebuilding a few keys would
    have silently dropped `finish_reason`, `request_id` and `metadata` from
    cached answers.
  - OpenAI parsing is defensive about missing `choices`: "OpenAI-compatible"
    is a claim, not a guarantee, and a thin reply should not surface as a
    `KeyError` from inside the framework.
  - No provider SDKs; still `requests` + `pydantic`.

### Security
- **Untrusted text is now structurally separated from instructions.**
  Retrieved documents and tool results were inserted into the conversation
  verbatim, with nothing marking them as data. Both are now fenced with a
  random per-agent marker, and any occurrence of that marker is stripped from
  the text, so content cannot close its own block and continue at instruction
  level.
- **The conversation summary reached the system prompt unfenced.** Summaries
  are written by the model from earlier turns - which include tool results
  and retrieved documents - and spliced into the *system* message, the
  highest-trust slot there is. That was a path from a tool result straight
  into the instructions. It is fenced now.
- **Retrieved documents are no longer spliced into the user's turn.** They
  are stored beside it and rendered at send time, so memory records what the
  user actually said, a document can no longer forge the `Question:` boundary
  the old wrapper used, and a conversation reloaded from disk never carries a
  dead agent's markers.
- **The retrieval framing is descriptive rather than imperative.** It said
  "Use the following context to answer", which tells the model to act on
  whatever the corpus contains.
- The system prompt now states the data boundary when a run can contain
  untrusted content. This is **one layer and the weakest**: prompt injection
  is not solved by a system prompt, and SECURITY.md says so. The boundary
  that holds is in Python - `permissions` are a frozenset fixed at
  decoration, and `ToolPolicy` never sees retrieved or returned text.
- SECURITY.md gains a trust ladder for the five sources of text, what is
  enforced structurally, and what is explicitly not defended against.

### Fixed
- **RAG silently mis-scored mismatched embeddings.** A vector of the wrong
  width was zipped against a longer one and compared on the overlap, so a
  2-dimensional vector scored 1.0 against a 3-dimensional corpus. The index
  now fixes its dimension on the first vector it accepts and rejects any
  later document or query vector that does not match; `RAG.dimension`
  reports it.
- **`embed_fn` returning the wrong number of vectors corrupted the index.**
  Documents and embeddings drifted apart - too few silently dropped
  documents from every search, too many raised `IndexError` from `search()`
  long afterwards. It must now return exactly one vector per text.
- **A negative `top_k` returned the wrong documents.** `search(q, top_k=-1)`
  sliced the ranked list from the end, returning everything except the best
  match. `top_k` must now be an integer of at least 1, and non-integers are
  rejected rather than reaching the slice.
- **A failing or rejected batch left the index half-updated.** Documents were
  appended before `embed_fn` was called, so an exception left documents with
  no embeddings. Everything is validated before anything is stored.
- **Empty and non-finite vectors are rejected.** A zero-width vector can
  never rank anything, and a NaN propagates into every score it touches. A
  *zero* vector is still accepted - some models emit one - and scores 0.0.
- **`_cosine` no longer compares a prefix.** Mismatched lengths raise instead
  of zipping to the shorter vector, results are clamped to `[-1.0, 1.0]`, and
  a magnitude that overflows the squared sum returns 0.0 rather than NaN,
  which would corrupt the ranking.
- **Empty documents are rejected.** They can never match and only dilute
  results. Non-string documents get a distinct message.
- **A TF-IDF query with no indexable tokens returns nothing** instead of the
  first `top_k` documents scored 0.0. Nothing to search *with* is not the
  same as having searched and found nothing.

### Added
- **Tool execution semantics.** A model can request several tools in one turn
  and they all ran concurrently, with no way to say that a tool must not
  overlap with itself. `@tool(concurrency="exclusive")` marks one that runs
  alone: nothing else from that turn runs while it does.
  - `concurrency="parallel"` is the default and is exactly the previous
    behaviour - a turn of only parallel tools is still a single concurrent
    batch. An unknown mode raises at decoration time.
  - Mixed turns are defined: calls keep the order the model asked for,
    consecutive parallel ones are grouped, and an exclusive call is a group
    of one.
  - `side_effects=True` still does **not** serialise anything - it describes
    a tool to a policy and to the audit log. A tool whose concurrent calls
    would race needs `concurrency="exclusive"` as well. Making
    `side_effects` imply exclusivity would silently change scheduling for
    tools already marked with it.
  - The framework does not infer dependencies between different tools, and
    the docs say so: a tool-call list expresses no ordering, and none can be
    recovered from it. Exclusivity holds within a turn - two concurrent
    sessions can still overlap, so process-wide exclusion needs a lock
    inside the tool.
- **Structured `AgentEvent` observability.** A flat, immutable event stream
  for logging, metrics and debugging - no OpenTelemetry, no new dependency,
  no tracing framework:

  ```python
  stop = agent.subscribe(lambda event: log.info("%s", event))
  ```

  - Events: `AgentStarted`, `AgentIteration`, `LLMStarted`, `LLMFinished`,
    `ToolStarted`, `ToolFinished`, `ToolFailed`, `AgentFinished`,
    `AgentFailed`. Each carries `run_id`, `session_id`, `agent`,
    `timestamp`, and where they apply `duration`, `model`, `tool`,
    `tool_call_id`, `usage` and `metadata`. `as_dict()` is JSON-safe.
  - `run_id` is unique per run, so concurrent runs never interleave
    ambiguously. `RunState.snapshot()` reports the same id.
  - `ToolFailed` may arrive without a preceding `ToolStarted` - a call
    refused before it ran never started - and `metadata["reason"]`
    distinguishes `unknown_tool`, `invalid_arguments`, `denied`,
    `approval_denied`, `timeout` and `raised`.
  - **Payloads are excluded by default**: prompts, tool arguments, tool
    results and answers are omitted in favour of counts and sizes, because
    an event stream usually ends up in a log aggregator.
    `Agent(event_payloads=True)` opts in.
  - `Agent.subscribe(fn)` returns an unsubscribe. `Callback.on_event` is the
    class-based form; existing `Callback` subclasses are unaffected, since
    the base `on_event` is a no-op and every older hook still fires.
  - Handler errors stay logged and swallowed; `Agent(strict_callbacks=True)`
    re-raises them, which is what you want in tests.
- **Run budgets.** `Budget` bounds what a single run may consume, so a
  runaway agent stops deterministically instead of looping until something
  else gives out:

  ```python
  Agent(
      llm,
      tools=[...],
      budget=Budget(
          max_tool_calls=20,
          max_total_tokens=50_000,
          max_tool_output=200_000,
          timeout=60,
      ),
  )
  ```

  - `BudgetExceededError` with `TokenBudgetExceeded`,
    `ToolCallBudgetExceeded`, `ToolOutputBudgetExceeded`,
    `TimeBudgetExceeded` and `CostBudgetExceeded`. Each carries
    `.limit_name`, `.limit` and `.used`.
  - `max_iterations` keeps its existing contract: exhausting it forces a
    final answer rather than raising, and it defaults to the agent's own
    `max_iterations`, so existing agents are unchanged. It is the only
    budget that does not raise, and there is deliberately no iteration
    exception for one that never fires.
  - Budgets are **per run**. `session.usage` still accumulates over the
    session's lifetime. `agent.session(budget=...)` overrides per caller.
  - No tool call escapes: the budget is claimed on the single path every
    model-requested call takes, before the tool is located or authorized, so
    unknown and policy-denied calls count too.
  - `session.last_run` exposes a `RunState` with iterations, tool calls,
    output characters, usage, elapsed time, estimated cost and which limit
    stopped the run; `.snapshot()` returns it as a JSON-safe dict.
  - `max_tool_output` is the run total, distinct from `Tool.max_output_size`,
    which caps a single result.
  - **Cost is an estimate and needs your rates.** Unchained ships no price
    table - published prices change and a stale table would under-report.
    `Budget(max_cost=...)` without `pricing` raises at construction, and a
    model absent from `pricing` raises rather than being costed as zero.
  - Budgets are checked before spending, so a call already under way can
    carry the total slightly past a limit; the run stops immediately after.
- **Tool output limits.** A tool could return megabytes straight into the
  transcript - overflowing the context window, costing money on every
  subsequent turn, and growing memory. `@tool(max_output_size=8_000)` bounds
  one tool and `Agent(max_tool_output_size=20_000)` all of them, with the
  tool's own setting taking precedence. `None` still means unbounded, so
  existing agents are unchanged.
  - Applied inside `Agent._execute`, before the result reaches memory, a
    provider or a callback - there is no path where the full text gets
    through. An oversized exception message is bounded the same way.
  - **Nothing is truncated silently**: an oversized result carries a note
    stating how much was dropped.
  - **JSON is never silently corrupted**: if the full result was valid JSON,
    the note says so and warns that the fragment will not parse, because a
    model handed JSON will otherwise try. Structured output that fits is
    passed through byte-for-byte.
  - The budget counts **characters, not bytes**, so a slice can never split a
    code point and produce invalid text, and it lines up with
    `Memory(max_tokens=...)`, which estimates tokens the same way.
  - New `ToolOutputTruncated`, carrying `.metadata` as a plain JSON-safe dict.
    It is not an exception - the tool succeeded and a shortened result is
    still useful.
  - This is a context and cost boundary, **not a security sandbox**: it limits
    what a tool sends onward, not what it can read or do. Documented in the
    README and SECURITY.md.
- **Tool execution timeouts.** LLM calls had a timeout; an arbitrary Python
  tool could block the agent forever. `@tool(timeout=10)` sets a budget for
  one tool and `Agent(tool_timeout=30)` a default for all of them, with the
  tool's own setting taking precedence. `None` still means wait forever, so
  existing agents are unchanged, and a tool with no timeout runs inline as
  before — no executor, no thread.
  - On overrun the model receives an ordinary tool error and the loop
    continues; the run does not hang and does not raise.
  - Concurrent calls each get their own budget. The executor is shut down
    with `wait=False`, because the default `wait=True` would block on the
    very call just abandoned — re-creating the hang the timeout exists to
    prevent.
  - New `ToolTimeoutError`.
  - **The timeout bounds the agent's wait, not the tool's work.** Python
    cannot cancel a running thread: the abandoned call keeps running, keeps
    its thread, and may still complete, so a side effect after a timeout must
    be treated as unknown rather than as not having happened. Hard
    cancellation needs process isolation (`examples/coder.py`), and HTTP
    tools still need their own network timeout. Documented in the README and
    SECURITY.md rather than papered over.
  - Approval waits are deliberately outside the budget: a human pausing to
    confirm must not be read as a hanging tool.

### Security
- **Routing decisions are now validated against the agent registry.**
  `Router.route()` asks for JSON constrained to the registered agent names
  and resolves the answer by looking it up in that registry, so no reply can
  name an agent the router does not hold. A decision is valid only if it
  identifies exactly one registered agent.
- **Agent names are validated when the `Router` is built.** A blank name, or
  two names that collide once case and surrounding whitespace are normalised,
  now raises `ValueError`. Both were previously silent: a blank-named agent
  could never be routed to, and one of two identically-named agents always
  won, so "resolve to exactly one agent" was not achievable.
- **Agent descriptions can no longer forge entries in the router's agent
  list.** Descriptions are interpolated into the routing prompt; one
  containing newlines could present extra `- name:` lines. Whitespace is now
  collapsed and the text capped.
- **Text matching no longer ignores word order.** A multi-word agent name
  matched any reply containing its words anywhere, so an `admin delete` agent
  was selected by "do not use admin, use the delete path". Matching now
  requires the name to appear as a contiguous run of words.
- `Agent._loads_object()` now always returns a dict. A reply of `null` or
  `[1, 2]` is valid JSON but not an object, and returning it handed a
  non-mapping to `schema(**data)` in the structured-output repair loop, where
  it raised `TypeError` instead of the expected `ValidationError`.

### Added
- `Router(strict=True)` accepts only a structured decision or a bare exact
  name, refusing prose. Text matching cannot read sense - prose mentioning one
  agent in order to reject it ("not cost") resolves to it - and `strict`
  closes that gap for deployments that need it.

### Security
- **Tool-call responses are no longer cached by default.** A cached response
  carrying `tool_calls` is a stored decision to act: an identical later prompt
  replayed it without the model being consulted, and without a request going
  out to make the replay visible — the same refund issued twice from one
  model decision. `LLM(cache=True)` now means the `"final_only"` policy:
  plain answers are cached, tool-call responses are returned to the caller but
  never stored.
- **The cache key no longer collides across different tools sharing a name.**
  Tools were keyed by name alone, so a `search` over public documents and a
  `search` over internal records — or the same tool before and after its
  description or parameters changed — served each other's cached responses.
  Tools are now keyed by their full schema, sorted so that offering the same
  tools in a different order still hits.
- **Response formats are keyed by schema, not by class name.** Two unrelated
  Pydantic models both called `Item` previously collided.
- **`max_tokens`, `base_url` and `provider` are now part of the cache key.**
  `max_tokens` is sent to Anthropic and truncates the reply, so two `LLM`s
  differing only in it returned each other's answers.
- **Cached responses are handed out as private copies.** Every hit previously
  returned the same dict, and `Agent` puts responses into conversation memory
  — so one conversation mutating a response silently rewrote what the cache
  served the next.

### Added
- `LLM(cache=...)` accepts a policy name as well as a bool: `"none"`,
  `"final_only"` (the default, and what `cache=True` means) or `"all"` for
  read-only tool sets that want tool-call responses cached too. An unknown
  name raises. The active policy is readable as `llm.cache_policy`.
- `LLM.clear_cache()` to invalidate every entry.
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
