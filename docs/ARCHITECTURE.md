# System Architecture

## Overview

Unchained is a **single-file agentic AI framework** that provides everything needed to build intelligent AI agents — tool calling, memory, retrieval-augmented generation, multi-agent orchestration, and structured output.

For the exact current size, run `python benchmarks/compare_frameworks.py` (it
counts live rather than relying on a number in prose, which goes stale as the
file grows).

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
│  │  │          │ → run_all(): parallel execution + synthesis           │    │
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
runs them concurrently on a `ThreadPoolExecutor` (most tools are I/O-bound)
and feeds the results back to the model in the original order.

### 2. LLM Backend (Unified Interface)

```
LLM.chat(messages, tools, response_format)
     │
     ├── provider == "openai"
     │   └── POST /v1/chat/completions
     │       └── Returns: {content, tool_calls[], usage}
     │
     ├── provider == "anthropic"
     │   └── POST /v1/messages
     │       └── Converts: system msg → system param
     │       └── Converts: tools → Anthropic tool format
     │       └── Returns: {content, tool_calls[], usage}
     │
     └── provider == "ollama"
         └── POST /api/chat
             └── Returns: {content, tool_calls[], usage}

All providers return IDENTICAL response format:
{
  "content": str,        # Text response
  "tool_calls": [        # Tools the LLM wants to call
    {"name": str, "arguments": dict}
  ],
  "usage": dict          # Token counts
}

Cross-cutting LLM behaviour:
  • base_url per provider can come from an env var (OPENAI_BASE_URL /
    ANTHROPIC_BASE_URL / OLLAMA_BASE_URL) - this is what lets provider="openai"
    target any OpenAI-compatible endpoint with no code change.
  • Requests go through a persistent requests.Session (one per LLM instance)
    so repeated calls reuse the TCP/TLS connection.
  • cache=True stores responses in a bounded LRU (cache_size, default 256)
    with an optional cache_ttl, rather than an unbounded dict.
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
     ├── [1] RAG augmentation (if rag provided)
     │        └── Search → prepend context to input
     │
     ├── [2] Add to memory
     │
     └── [3] Loop (max_iterations):
              │
              ├── Build messages (system + summary + history)
              ├── Call LLM with tool schemas
              │
              ├── IF no tool_calls → return content (DONE)
              │
              └── IF tool_calls:
                   ├── Authorize + execute each (concurrently if >1)
                   ├── Add each tool call + result to memory, in order
                   └── Continue loop (LLM sees results next iteration)
```

Every model-requested tool call reaches a function through one fixed path in
`Agent._execute`, and there is no branch around it:

```
model asks for a tool
     │
     ├── locate it in this agent's tools      → unknown  ─┐
     ├── Tool.validate_arguments(...)         → invalid  ─┤
     ├── ToolPolicy.authorize(...)            → denied   ─┤
     ├── approval, if the policy wants it     → refused  ─┤
     │                                                    │
     ├── tool.func(**arguments)                           │
     └── audit the decision  ◄─────────────────────────────┘
                                    (refusals become an observation
                                     for the model; the loop continues)
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
it is allowed to call. With no `policy=`, the default allows everything the
agent was given, so agents written before this layer existed are unaffected.
See the Tool authorization section of the README and `SECURITY.md`.

`agent.arun(...)` offloads the whole method above to a worker thread via
`asyncio.to_thread`, so it can be awaited from async code (FastAPI, aiohttp,
...) without blocking the event loop.

**Configuration vs state.** The loop above runs against a `Session`, not
against the Agent:

```
Agent  (shared, safe to reuse)        Session  (one conversation)
  llm, tools, system_prompt             memory
  rag, callbacks, policy                usage
  max_iterations, approve               metadata, callbacks
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
     ├── Ask LLM for JSON: {agent}    │   └── agent.run(query)
     ├── Look it up in the registry   │
     └── Delegate, or RoutingError    └── Return {name: result}

Routing fails closed. The decision is requested as JSON constrained to the
registered names, then resolved by a lookup in that registry - the prompt is
where the model is told what is allowed, the lookup is what enforces it. A
decision is valid only if it identifies exactly one registered agent; empty,
evasive, hallucinated, malformed and ambiguous replies all raise
`RoutingError` rather than falling back to an arbitrary agent. Agents differ
in the tools - and so the privileges - they hold, so this is an authorization
decision, enforced in Python rather than by trusting the router prompt.
`Router(..., fallback=agent)` names a default explicitly; `strict=True`
refuses prose replies entirely.

PickMyStack uses run_all() → Synthesizer pattern:
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ CostAgent│  │ FitAgent │  │TrendAgent│   ← All run in parallel
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              ┌──────────────┐
              │ Synthesizer  │   ← Combines all results
              └──────────────┘
                      │
                      ▼
              Final Recommendation
```

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
User → Router.run_all() → Agent_A → result_a ─┐
                        → Agent_B → result_b ──┤─→ Synthesizer → Final
                        → Agent_C → result_c ─┘
```

---

## PickMyStack Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                           │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐     │
│  │ Use Case     │  │ Constraints │  │ Progress Display  │     │
│  │ (text input) │  │ (budget,    │  │ (real-time agent  │     │
│  │              │  │  team, etc) │  │  status updates)  │     │
│  └──────┬───────┘  └──────┬──────┘  └───────────────────┘     │
│         └──────────────────┘                                   │
│                    │                                           │
└────────────────────┼───────────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                   UNCHAINED BACKEND                              │
│                                                                │
│  ┌─────────────────────┐                                       │
│  │   Knowledge Base    │  ← RAG over framework docs            │
│  │   (frameworks.md,   │    (models_guide.md, pricing,         │
│  │    models_guide.md, │     deployment_options.md)             │
│  │    deploy.md)       │                                       │
│  └─────────┬───────────┘                                       │
│            │                                                   │
│  ┌─────────▼───────────────────────────────────────────┐       │
│  │              EVALUATION PIPELINE                     │       │
│  │                                                     │       │
│  │  ┌────────────┐ ┌───────────┐ ┌─────────────┐      │       │
│  │  │ CostAgent  │ │ FitAgent  │ │ TrendAgent  │      │       │
│  │  │ • pricing  │ │ • scoring │ │ • GitHub    │      │       │
│  │  │ • hosting  │ │ • match % │ │ • community │      │       │
│  │  │ • TCO      │ │ • fit     │ │ • momentum  │      │       │
│  │  └─────┬──────┘ └─────┬─────┘ └──────┬──────┘      │       │
│  │        └───────────────┼──────────────┘             │       │
│  │                        ▼                            │       │
│  │               ┌─────────────────┐                   │       │
│  │               │   Synthesizer   │                   │       │
│  │               │  (Final rank)   │                   │       │
│  │               └────────┬────────┘                   │       │
│  └────────────────────────┼────────────────────────────┘       │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │ TOP 3 STACKS    │
                   │ with reasoning  │
                   └─────────────────┘
```

---

## File Structure (Complete)

```
unchained/
├── unchained/__init__.py         ← THE framework (single file)
├── pyproject.toml               ← Package config + dependencies
├── README.md                    ← Star-attracting documentation
├── LICENSE                      ← MIT License
├── .gitignore
├── examples/
│   ├── researcher.py            ← Web research agent
│   ├── coder.py                 ← Code execution agent
│   ├── data_analyst.py          ← CSV analysis agent
│   ├── quickstart.py            ← Zero-setup tour using MockLLM
│   ├── sqlite_memory.py         ← Persistent, session-scoped Memory
│   └── pickmystack/             ← Flagship multi-agent app
│       ├── app.py               ← CLI entry point
│       ├── tools/
│       │   ├── cost_estimator.py    ← Pricing data + estimation
│       │   ├── benchmark_fetcher.py ← Framework comparison data
│       │   └── doc_retriever.py     ← RAG search over knowledge
│       ├── knowledge/
│       │   ├── frameworks_comparison.md
│       │   ├── models_guide.md
│       │   └── deployment_options.md
│       └── ui/
│           ├── app_ui.py        ← Streamlit web interface
│           ├── requirements.txt ← UI dependencies
│           ├── Dockerfile       ← Container deployment
│           ├── README_DEPLOY.md ← Deployment guide
│           └── .streamlit/
│               └── config.toml  ← Theme configuration
├── benchmarks/
│   └── compare_frameworks.py    ← Unchained vs LangChain vs CrewAI
├── tests/
│   └── test_unchained.py         ← Unit tests (pytest)
└── docs/
    ├── ARCHITECTURE.md          ← This file
    └── USER_MANUAL.md           ← Non-technical user guide
```

---

## Key Design Decisions

### Why TF-IDF instead of embeddings?
- Zero additional dependencies (no sentence-transformers, no API calls)
- Works offline, instant indexing
- Good enough for 100-1000 documents
- Uses smoothed IDF (sklearn-style) for small corpus robustness
- Can be swapped for embeddings later (same interface)

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

| Extension | How | Difficulty |
|---|---|---|
| New tools | `@tool` decorator on any function (sync or async) | Easy |
| Tool authorization | Subclass `ToolPolicy`, pass `policy=` to `Agent` | Easy |
| New LLM provider | Add `_provider_name()` method to `LLM` | Easy |
| OpenAI-compatible provider | Reuse `provider="openai"` with a different `base_url` | Easy |
| Better retrieval | Replace `RAG._rebuild_index()` + `search()` | Medium |
| Persistent memory | Subclass `Memory`, add SQLite backend | Medium |
| Custom routing | Subclass `Router`, override `route()` | Medium |
| True async HTTP | Subclass `LLM`, override `chat()`/`_request()` with an async client | Medium |

Streaming and thread-offloaded async (`arun`/`achat`) already ship in core -
see `LLM.stream()`, `Agent.stream()`, `Agent.arun()`, `LLM.achat()`.

---

## Performance Characteristics

| Operation | Time Complexity | Space |
|---|---|---|
| Tool registration | O(1) | O(params) |
| RAG indexing | O(n × d) | O(n × vocab) |
| RAG search | O(n × vocab) | O(n) |
| Memory add | O(1) amortized | O(max_messages) |
| Memory compress | O(overflow) | O(1) |
| Agent.run() | O(iterations × LLM latency) | O(messages) |

Where: n = documents, d = avg document length, vocab = unique terms
