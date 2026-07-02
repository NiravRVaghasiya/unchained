# System Architecture

## Overview

Unchained is a **single-file agentic AI framework** (417 lines) that provides everything needed to build intelligent AI agents — tool calling, memory, retrieval-augmented generation, multi-agent orchestration, and structured output.

## Design Principles

1. **Single-file core** — Everything in `unchained.py` (< 500 lines)
2. **Two dependencies** — `requests` + `pydantic` only
3. **Provider-agnostic** — Same code works with OpenAI, Anthropic, or local Ollama
4. **No magic** — No metaclasses, no runtime patching, no hidden state
5. **Composition over inheritance** — Agents compose tools, memory, and RAG

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UNCHAINED FRAMEWORK                              │
│                           (unchained.py — 417 lines)                          │
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
     ├── Extracts docstring → becomes tool description
     ├── Identifies optional params (has default value)
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
                   ├── Execute each tool
                   ├── Add tool call + result to memory
                   └── Continue loop (LLM sees results next iteration)
```

### 6. Router (Multi-Agent Orchestration)

```
router.route(query)              router.run_all(query)
     │                                │
     ├── Build agent descriptions     ├── For each agent:
     ├── Ask LLM: "which agent?"      │   └── agent.run(query)
     ├── Fuzzy-match agent name       │
     └── Delegate to chosen agent     └── Return {name: result}

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
User → Agent → LLM → [tool_call] → Tool → LLM → Response
                ↑                              │
                └──────────── loop ────────────┘
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
├── unchained.py                  ← THE framework (417 lines)
├── pyproject.toml               ← Package config + dependencies
├── README.md                    ← Star-attracting documentation
├── LICENSE                      ← MIT License
├── .gitignore
├── examples/
│   ├── researcher.py            ← Web research agent (30 lines)
│   ├── coder.py                 ← Code execution agent (25 lines)
│   ├── data_analyst.py          ← CSV analysis agent (20 lines)
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
- Entire framework readable in 15 minutes
- Copy `unchained.py` into any project — no install needed
- Easy to audit, understand, and modify
- Forces discipline: every line must earn its place

---

## Extension Points

Unchained is designed to be extended, not forked:

| Extension | How | Difficulty |
|---|---|---|
| New tools | `@tool` decorator on any function | Easy |
| New LLM provider | Add `_provider_name()` method to `LLM` | Easy |
| Better retrieval | Replace `RAG._rebuild_index()` + `search()` | Medium |
| Persistent memory | Subclass `Memory`, add SQLite backend | Medium |
| Custom routing | Subclass `Router`, override `route()` | Medium |
| Streaming | Modify `LLM.chat()` to yield chunks | Medium |
| Async execution | Wrap with `asyncio` | Medium |

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
