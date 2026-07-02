# Unchained User Manual

A friendly, plain-language guide to using Unchained. No deep background needed:
if you can run a Python file, you can build an agent.

## What is Unchained?

Unchained is a small toolkit for building "AI agents". An agent is a program
that uses a language model (like GPT-4o, Claude, or a local model) to answer
questions and get things done. Unlike a plain chatbot, an agent can also **use
tools** (call your functions), **remember** the conversation, and **look things
up** in your own documents.

Everything lives in a single file, `unchained.py`, and it only needs two extra
Python packages.

## Before you start

You need:

1. **Python 3.9 or newer** installed.
2. **A model to talk to.** Pick one:
   - **Ollama (free, runs on your computer):** install it from [ollama.com](https://ollama.com), then run `ollama pull llama3.1`. Nothing else to pay or configure.
   - **OpenAI or Anthropic (paid, runs in the cloud):** you'll need an API key from their website.

## Installation

Open a terminal in the project folder and run:

```bash
pip install requests pydantic
```

That's it. If you want the tests or the web UI later:

```bash
pip install pytest streamlit
```

## Trying it with no setup

Don't have a model yet? You can still see everything work. `MockLLM` is a
pretend model that needs no key and no server:

```python
from unchained import Agent, MockLLM

agent = Agent(MockLLM(reply="Hi! I'm a mock model."))
print(agent.run("hello"))
```

Or run the guided tour:

```bash
python examples/quickstart.py
```

When you're ready for real answers, swap `MockLLM(...)` for
`LLM(provider="ollama")` (or `openai` / `anthropic`).

## Your first agent

Create a file called `hello.py`:

```python
from unchained import LLM, Agent

agent = Agent(LLM(provider="ollama"))     # uses local Ollama by default
print(agent.run("Explain what an AI agent is, in two sentences."))
```

Run it:

```bash
python hello.py
```

To use a cloud model instead, change one line:

```python
agent = Agent(LLM(provider="openai", model="gpt-4o-mini"))
```

Set your key first so the agent can log in:

- macOS/Linux: `export OPENAI_API_KEY=sk-...`
- Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`

## Giving your agent tools

A tool is just a normal Python function with a short description. Unchained
reads the function for you; you don't write any schemas.

```python
from unchained import LLM, Agent, tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = Agent(LLM(provider="ollama"), tools=[multiply])
print(agent.run("What is 23 times 19?"))
```

The agent decides when to call `multiply`, reads the result, and answers.

Tips for good tools:
- Give the function a clear name and a one-line description (the text in triple quotes).
- Add type hints (`a: int`) so the model knows what to send.
- Keep each tool focused on one job.

## Remembering the conversation

By default an agent remembers the recent conversation. For long chats, add
memory that automatically summarises older messages so it never balloons:

```python
from unchained import LLM, Agent, Memory

llm = LLM(provider="ollama")
agent = Agent(llm, memory=Memory(max_messages=20, llm=llm))
```

## Answering from your own documents (RAG)

Want the agent to answer using your notes, docs, or FAQs? Load them once and
attach them. Unchained finds the most relevant pieces automatically.

```python
from unchained import LLM, Agent, RAG

kb = RAG()
kb.add_many([
    "Our support hours are 9am to 5pm, Monday to Friday.",
    "Refunds are processed within 5 business days.",
])

agent = Agent(LLM(provider="ollama"), rag=kb)
print(agent.run("When can I reach support?"))
```

## Getting structured answers

If you need the answer as clean, predictable data (not a paragraph), describe
the shape you want and Unchained validates it for you:

```python
from pydantic import BaseModel
from unchained import LLM, Agent

class Contact(BaseModel):
    name: str
    email: str

agent = Agent(LLM(provider="openai", model="gpt-4o-mini"))
contact = agent.run("Extract the contact: Jane Doe, jane@acme.com", response_format=Contact)
print(contact.name)   # "Jane Doe"
print(contact.email)  # "jane@acme.com"
```

## Using several agents together

For bigger problems you can have specialist agents and a coordinator. The
coordinator can either pick the best specialist or ask them all and combine the
answers. See the flagship example below.

## The flagship example: PickMyStack

PickMyStack recommends an AI technology stack for whatever you want to build.
Three specialists (cost, fit, trend) study the options and a final agent ranks
the best picks.

Run it from the command line:

```bash
python -m examples.pickmystack.app "Build a customer-support chatbot" --budget 200 --team 3
```

Or open the friendly web version:

```bash
streamlit run examples/pickmystack/ui/app_ui.py
```

Type what you want to build, set your budget and team size, and click
**Recommend a stack**.

## Streaming the answer as it's written

For a chat-like feel, stream the reply token by token instead of waiting for
the whole thing:

```python
for piece in agent.stream("Explain how RAG works."):
    print(piece, end="", flush=True)
```

## Seeing what your agent is doing

To watch each step (which tools it calls, how many tokens it uses), turn on
logging:

```python
import logging
from unchained import Agent, LoggingCallback

logging.basicConfig(level=logging.INFO)
agent = Agent(llm, tools=[...], callbacks=[LoggingCallback()])
```

You can also check how many tokens a run used:

```python
agent.run("Summarise today's standup notes.")
print(agent.usage)   # prompt/completion/total token counts
```

## Reliability

If a provider is briefly overloaded or rate-limits you, Unchained automatically
waits and retries a couple of times before giving up. You can tune this:

```python
llm = LLM(provider="openai", max_retries=3, backoff=0.5)
```

## Troubleshooting

**"Connection refused" or a timeout with Ollama**
Ollama isn't running. Start it, then confirm a model is available with
`ollama pull llama3.1`.

**"An API key is required"**
You chose OpenAI or Anthropic but didn't provide a key. Set the environment
variable (see "Your first agent") or paste the key into the web UI's sidebar.

**The agent ignores my tool**
Make sure the tool has a clear docstring and type hints, and that you passed it
in the `tools=[...]` list.

**The structured answer failed to parse**
Unchained already asks the model to fix invalid JSON once (tunable via
`structured_retries`). If it still fails, try a stronger model (for example
`gpt-4o-mini`) or simplify the schema.

**Responses cost more than expected**
Output text is the main cost driver. Use a cheaper model, keep prompts short,
or add RAG so a smaller model can still answer accurately.

## Where to go next

- [Overview](index.md) - a quick tour and feature list.
- [Architecture](ARCHITECTURE.md) - how everything works under the hood.
- `examples/` - short, self-contained agents you can copy and adapt.
