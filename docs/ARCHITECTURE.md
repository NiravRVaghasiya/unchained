# System Architecture

## Overview

Unchained is a **single-file agentic AI framework** that provides everything needed to build intelligent AI agents — tool calling, memory, retrieval-augmented generation, multi-agent orchestration, and structured output.

## Design Principles

1. **Single-file core** — Everything in `unchained/__init__.py`, no submodules
2. **Two dependencies** — `requests` + `pydantic` only
3. **Provider-agnostic** — Same code works with OpenAI, Anthropic, local Ollama, or any OpenAI-compatible endpoint
4. **No magic** — No metaclasses, no runtime patching, no hidden state
5. **Composition over inheritance** — Agents compose tools, memory, and RAG

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UNCHAINED FRAMEWORK                              │
│                            (unchained/__init__.py)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         MULTI-AGENT LAYER                           │    │
│  │  ┌──────────┐                                                       │    │
│  │  │  Router  │ → Intent classification → Route to best agent         │    │
│  │  │          │ → run_all() / synthesize(): parallel (+ synthesis)    │    │
│  │  └──────────┘                                                       │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                         │
│  ┌────────────────────────────────▼────────────────────────────────────┐    │
│  │                          AGENT CORE                                 │    │
│  │                                                                     │    │
│  │   ┌──────────────────────────────────────────────────────────┐      │    │
│  │   │                    Agent Loop                             │      │    │
│  │   │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │      │    │
│  │   │  │  THINK  │───▶│   ACT   │───▶│ OBSERVE │──┐           │      │    │
│  │   │  │ (LLM)   │    │ (Tool)  │    │(Result) │  │           │      │    │
│  │   │  └─────────┘    └─────────┘    └─────────┘  │           │      │    │
│  │   │       ▲                                      │           │      │    │
│  │   │       └──────────────────────────────────────┘           │      │    │
│  │   │                  (repeat until done)                     │      │    │
│  │   └──────────────────────────────────────────────────────────┘      │    │
│  │                                                                     │    │
│  └──┬──────────────┬──────────────┬──────────────┬─────────────────────┘    │
│     │              │              │              │                           │
│  ┌──▼───┐  ┌──────▼─────┐  ┌────▼────┐  ┌─────▼──────┐                    │
│  │ LLM  │  │   Memory   │  │   RAG   │  │   Tools    │                    │
│  │      │  │            │  │         │  │            │                    │
│  │OpenAI│  │ Window(20) │  │ TF-IDF  │  │ @tool deco │                    │
│  │Anthro│  │ + Summary  │  │ + Cosine │  │ Auto-schema│                    │
│  │Ollama│  │ Compress   │  │ Smoothed │  │ Type infer │                    │
│  └──────┘  └────────────┘  └─────────┘  └────────────┘                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     STRUCTURED OUTPUT                               │    │
│  │  Pydantic schema → JSON mode → validated response                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                          EXTERNAL DEPENDENCIES
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│    requests      │  │    pydantic      │  │  LLM Provider (external)     │
│  (HTTP client)   │  │ (validation)     │  │  • Ollama (localhost:11434)  │
│                  │  │                  │  │  • OpenAI API                │
│                  │  │                  │  │  • Anthropic API             │
└──────────────────┘  └──────────────────┘  └──────────────────────────────┘
```

---

## Component Details

### 1. Tool System (`Tool` class + `@tool` decorator)

```
@tool decorator
     │
     ├── Introspects function signature (inspect module)
     ├── Extracts type hints → maps to JSON Schema types
     │     scalars (str/int/float/bool), list, dict, Optional[X]
     │     Literal[...] / Enum        → JSON Schema "enum"
     │     nested Pydantic BaseModel  → inlined object schema
     ├── Extracts docstring → becomes tool description
     ├── Identifies optional params (has default value)
     ├── Detects async def            → run() drives it via asyncio.run()
     │
     └── Produces OpenAI-compatible function schema:
         {
           "type": "function",
           "function": {
             "name": "...",
             "description": "...",
             "parameters": { "type": "object", "properties": {...} }
           }
         }
```

When an agent turn produces more than one tool call, `Agent._execute_calls`
runs consecutive parallel calls concurrently on a `ThreadPoolExecutor` (capped
by `max_tool_workers`; most tools are I/O-bound); a `concurrency="exclusive"`
tool runs alone; results return to the model in the original order.

### 2. LLM Backend (Unified Interface)

```
LLM.chat(messages, tools, response_format)
     │
     ├── provider == "openai"
     │   └── POST /v1/chat/completions
     │       └── Returns: LLMResponse
     │
     ├── provider == "anthropic"
     │   └── POST /v1/messages
     │       └── Converts: system msg → system param
     │       └── Converts: tools → Anthropic tool format
     │       └── Returns: LLMResponse
     │
     └── provider == "ollama"
         └── POST /api/chat
             └── Returns: LLMResponse

All providers return an LLMResponse (content, tool_calls [{name, arguments,
id}], usage, finish_reason, provider, model, request_id, metadata), which also
supports dict-style read access.

Cross-cutting LLM behaviour:
  • base_url per provider can come from an env var (OPENAI_BASE_URL /
    ANTHROPIC_BASE_URL / OLLAMA_BASE_URL) - this is what lets provider="openai"
    target any OpenAI-compatible endpoint with no code change.
  • Requests go through a persistent requests.Session (one per LLM instance)
    so repeated calls reuse the TCP/TLS connection.
  • cache=True (policy "final_only") stores plain answers, never tool-call
    responses, in a bounded LRU (cache_size, default 256) with an optional
    cache_ttl.
  • achat() offloads chat() to a worker thread (asyncio.to_thread) so it can
    be awaited from async code - it does not add non-blocking sockets.
```

### 3. Memory (Sliding Window + Compression)

```
                    max_messages = 20
┌─────────────────────────────────────────────┐
│ [msg1] [msg2] ... [msg18] [msg19] [msg20]   │  ← Window full
└─────────────────────────────────────────────┘
                      │
            Overflow triggers _compress()
                      │
                      ▼
┌──────────────┐  ┌───────────────────────────┐
│   Summary    │  │ [msg11] ... [msg19] [msg20]│  ← Keeps recent half
│ (compressed) │  │                           │
│  msg1-msg10  │  │                           │
└──────────────┘  └───────────────────────────┘

Compression strategies:
  • With LLM: Summarizes overflow via LLM call
  • Without LLM: Truncates to first 100 chars per message

Optional max_tokens (rough 4-chars/token estimate, no tokenizer dependency):
  a message-count window alone can't promise a token budget - a handful of
  very long messages can still overflow the context window even under a
  small max_messages. When set, _compress() keeps shrinking the retained
  window below max_messages // 2 (down to a minimum of 1 message) until the
  kept messages fit the token budget too.

The window is never cut inside an assistant tool_calls group, so it may
exceed max_messages/max_tokens by that one group.
```

### 4. RAG (TF-IDF + Smoothed IDF + Cosine Similarity)

```
Documents → Tokenize → TF (term frequency per doc)
                            │
                            ▼
              IDF = log((1 + n) / (1 + df)) + 1   ← Smoothed (sklearn-style)
                            │
                            ▼
              TF-IDF Matrix (sparse, in-memory)
                            │
Query ─────────────────────▶│
                            │
              Cosine Similarity = dot(q, d) / (|q| * |d|)
                            │
                            ▼
              Ranked results [{text, score, metadata}]

Why smoothed IDF?
  • Standard log(n/df) → 0 when term appears in all docs
  • Smoothed version always produces positive scores
  • Works correctly even with 2-3 documents
```

### 5. Agent Core (ReAct Loop)

```
agent.run(user_input)
     │
     ├── [1] Add the user turn to memory (RAG hits, if any, are stored
     │        beside it and fenced when the messages are rendered)
     │
     ├── [2] Loop (max_iterations):
     │        │
     │        ├── Build messages (system + summary + history)
     │        ├── Call LLM with tool schemas
     │        │
     │        ├── IF no tool_calls → that is the answer; leave the loop
     │        │
     │        └── IF tool_calls:
     │             ├── Authorize + execute each (concurrently if >1)
     │             ├── Add each tool call + result to memory, in order
     │             └── Continue loop (LLM sees results next iteration)
     │
     └── [3] After the loop:
              ├── Iterations exhausted → one final call without tools
              │   (under the schema, if response_format is set)
              ├── response_format + tools, loop answered → formatting pass
              └── response_format → validate, repairing up to structured_retries
```

Every model-requested tool call reaches a function through one fixed path in
`Agent._execute`, and there is no branch around it:

```
model asks for a tool
     │
     ├── claim one call from the run budget   → exhausted: raise
     │                                          BudgetExceededError (run stops)
     ├── locate it in this agent's tools      → unknown  ─┐
     ├── Tool.validate_arguments(...)         → invalid  ─┤
     ├── ToolPolicy.authorize(...)            → denied   ─┤
     ├── approval, if the policy wants it     → refused  ─┤
     │                                                    │
     ├── audit the decision  ◄─────────────────────────────┘
     │                              (refusals end here as an observation
     │                               for the model; the loop continues)
     ├── tool.run(arguments)  (via _invoke: timeout)
     └── bound output
```

Three jobs are kept separate inside `Tool`, and conflating them is how
frameworks end up trusting the model's own paperwork:

| | what it is | trusted? |
|---|---|---|
| `Tool.schema` | JSON Schema shown to the model | no — advice the model may ignore |
| `Tool.validate_arguments()` | checks + normalises what arrived, from the **signature** | yes — the gate |
| `Tool.run()` | calls the function | only with validated arguments |

Validation is built from `inspect.signature` + `get_type_hints`, never from
`schema`. Enforcing the schema would mean enforcing the same document the
model was free to disregard.

The policy is Python, not prompt text: nothing the model emits can widen what
it is allowed to call. With no `policy=`, the default `ToolPolicy` allows every
tool the agent was given, subject only to each tool's own `allowed=` hook and
`requires_approval` flag. See [Tool authorization](GUIDE.md#tool-authorization) in the Developer guide and
`SECURITY.md`.

`agent.arun(...)` offloads the whole method above to a worker thread via
`asyncio.to_thread`, so it can be awaited from async code (FastAPI, aiohttp,
...) without blocking the event loop.

**Configuration vs state.** The loop above runs against a `Session`, not
against the Agent:

```
Agent  (shared, safe to reuse)        Session  (one conversation)
  llm, tools, system_prompt             memory
  rag, callbacks, policy                usage, last_run
  max_iterations, approve, budget       metadata, callbacks, budget
```

Every Agent method that touches conversation state takes the session as its
first argument (`_run(session, ...)`, `_execute(session, call)`), so the
signatures are the audit trail: a method without a `session` parameter cannot
reach a conversation. `agent.run()` is `agent.default_session.run()` - one
persistent session, created on first use, for single-conversation scripts.
Concurrent users get `agent.session()` each; isolation comes from owning
separate objects, not from locking.

### 6. Router (Multi-Agent Orchestration)

```
router.route(query)              router.run_all(query)
     │                                │
     ├── Build agent descriptions     ├── For each agent:
     ├── Ask LLM for JSON: {agent}    │   └── agent.session().run(query)
     ├── Look it up in the registry   │
     └── Delegate, or RoutingError    └── Return {name: result}
```

Routing fails closed: the reply must be a JSON decision or a bare registered
name, and it is resolved by a lookup in the agent registry, so anything else -
prose included - raises `RoutingError` unless `Router(..., fallback=agent)`
names a default. Agents differ in the tools - and so the privileges - they
hold, so this is an authorization decision, enforced in Python rather than by
trusting the router prompt. See [Multi-agent routing](GUIDE.md#multi-agent-routing) and SECURITY.md.

`synthesize()` runs `run_all()` and passes the results to the synthesizer
agent; see Pattern 4.

---

## Data Flow Patterns

### Pattern 1: Simple Q&A
```
User → Agent → LLM → Response
```

### Pattern 2: Tool-Augmented
```
User → Agent → LLM → [tool_call] → Policy → Tool → LLM → Response
                ↑                     │                   │
                │                     └── denied ─────────┤
                └──────────── loop ────────────────────────┘
```

### Pattern 3: RAG-Augmented
```
User → RAG.search() → [context] → Agent → LLM → Response
```

### Pattern 4: Multi-Agent Synthesis
```
User → Router.synthesize() → Agent_A → result_a ─┐
                           → Agent_B → result_b ──┤─→ Synthesizer → Final
                           → Agent_C → result_c ─┘
```

---

## Key Design Decisions

### Why TF-IDF instead of embeddings?
- Zero additional dependencies (no sentence-transformers, no API calls)
- Works offline, instant indexing
- Good enough for 100-1000 documents
- Uses smoothed IDF (sklearn-style) for small corpus robustness
- Dense embeddings are opt-in via `RAG(embed_fn=...)`

### Why sliding window memory?
- Predictable token usage (no surprise $100 bills)
- Simple to reason about
- Summary compression preserves key context
- No external database needed

### Why unified LLM interface?
- Switch providers with one line change
- Test with free Ollama, deploy with cloud
- Same code works everywhere
- No provider lock-in

### Why single file?
- Entire framework readable in one sitting, no jumping between submodules
- Copy the one file into any project as `unchained.py` — no install needed
- Easy to audit, understand, and modify
- Forces discipline: every line must earn its place

---

## Extension Points

Unchained is designed to be extended, not forked:

| Extension | How |
|---|---|
| New tool | `@tool` on any function (sync or async) |
| Stricter argument rules | annotate the parameter with a Pydantic model |
| New LLM provider | add `_yourprovider()` (chat) and `_stream_yourprovider()` methods to `LLM`, then register them in `_PROVIDER_DEFAULTS`, `_dispatch` and `stream()` (and `_PROVIDER_ENV_KEY` / `_PROVIDER_ENV_URL` if it reads env vars). Without editing core: pass `Agent` any object with `chat(messages, tools, response_format)` (and `stream(messages)`), as `MockLLM` does |
| OpenAI-compatible provider | reuse `provider="openai"` with a different `base_url` |
| True async HTTP (direct LLM calls) | subclass `LLM` and override `achat()` with an async client; `Agent.arun()` still runs the synchronous loop on a worker thread |
| Better retrieval | pass `embed_fn=` for dense embeddings, or subclass `RAG` (or supply any object) with a `search(query, top_k)` returning `[{"text", "score", "metadata"}]` |
| Persistent memory | subclass `Memory` — see `examples/sqlite_memory.py` |
| Custom tracing | subclass `Callback` and pass `callbacks=[...]` |
| Custom authorization | subclass `ToolPolicy` and pass `policy=...` |
| Per-user conversations | `agent.session(...)` — one `Session` per conversation |
| Custom routing | subclass `Router`, override `route()` |

Streaming and thread-offloaded async (`arun`/`achat`) already ship in core -
see `LLM.stream()`, `Agent.stream()`, `Agent.arun()`, `LLM.achat()`.

---

## Performance Characteristics

| Operation | Time Complexity | Space |
|---|---|---|
| Tool registration | O(1) | O(params) |
| Agent.run() | O(iterations × LLM latency) | O(messages) |
