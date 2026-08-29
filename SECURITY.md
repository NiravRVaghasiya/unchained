# Security Policy

## Supported versions

Unchained is pre-1.0. Security fixes are applied to the latest release on the
`main` branch.

## Reporting a vulnerability

Please do not open a public issue for security vulnerabilities. Instead, use
GitHub's private ["Report a vulnerability"](https://github.com/NiravRVaghasiya/unchained/security/advisories/new)
advisory flow, or contact the maintainers directly.

We aim to acknowledge reports within a few days and will keep you updated on
remediation progress.

## Notes for users

- **API keys:** never commit real keys. Use environment variables or a local
  `.env` (see `.env.example`), both of which are gitignored.
- **`examples/coder.py`** executes model-generated Python in-process. It is a
  demo only. Do not expose it to untrusted input without a proper sandbox
  (separate process, container, resource limits).
- **The Streamlit UI** has no built-in authentication. Put it behind access
  control before exposing it publicly, or it will spend your API quota.
- Treat all model output as untrusted when feeding it into tools, shells, or
  file operations.

## Boundaries enforced in code

Unchained enforces these in Python, not by asking the model to behave. They
hold even when the model is confused, jailbroken, or adversarial.

- **An agent can only call the tools it was given.** `Agent` dispatches through
  its own `tools` dict; a call naming anything else returns an error string to
  the model. There is no dynamic lookup and no name-based import.
- **Routing fails closed.** `Router.route()` dispatches only on an exact agent
  name or an unambiguous whole-word mention of exactly one agent. Anything
  else — an empty reply, a refusal, a hallucinated name, or a reply naming two
  agents — raises `RoutingError`. This matters because agents differ in the
  tools, and therefore the privileges, they hold: a silent fallback would let
  a confused router hand a query to a more privileged agent than the one
  intended. Pass `Router(..., fallback=agent)` to choose a default explicitly.
- **Tool-call concurrency is bounded.** The number of tool calls in a turn is
  chosen by the model, so it is untrusted input. The executor is capped by
  `Agent(max_tool_workers=8)` rather than sized to the request, so one response
  cannot spawn an unbounded number of threads.
- **No placeholder credential is transmitted.** When no API key is configured
  the `Authorization` / `x-api-key` header is omitted rather than sent with a
  stringified `None`.

Known non-boundaries, by design:

- **Tool arguments are not validated against the tool's JSON schema.** The
  schema is advisory — it is what the model is shown. Arguments arrive as a
  dict and are passed as keyword arguments; a mismatch raises inside the tool
  and is returned to the model as an error string. Validate untrusted
  arguments inside the tool itself (a Pydantic model parameter is one way).
- **Tool error text is fed back to the model and stored in memory.** If a tool
  raises an exception whose message contains a secret, that secret enters the
  transcript. Catch and sanitise inside tools that handle credentials.
- **RAG context is injected into the prompt.** Retrieved documents are model
  input, so a poisoned corpus is a prompt-injection vector. Trust your corpus
  the way you trust user input.
