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
- **Every model-requested tool call passes a policy first.** `Agent._execute`
  is the single path from model output to a tool function, and its sequence is
  fixed: locate the tool, validate the arguments, `ToolPolicy.authorize()`,
  approval if required, execute, audit. `Agent.run()`, `Agent.stream()` and
  `Agent.arun()` all funnel through it. A tool marked `requires_approval` is
  gated even when no policy is configured, so the metadata is never
  decorative. See "Tool authorization" in the README.
- **Refusals fail closed, including when the machinery itself breaks.** A
  policy that raises an unexpected exception denies the call rather than
  falling through to execution; an approval callback that raises is a refusal,
  not a pass; and a tool needing approval with no approver configured is
  refused. The default answer whenever the system cannot get a clear yes is
  no.
- **Approval is application-controlled.** The `approve=` callback is supplied
  to `Agent()` by your code. Model output cannot set it, reach it, or change
  its answer, and it is serialised with a lock so concurrent tool calls in one
  turn cannot re-enter your prompt. Tool metadata (`permissions`,
  `requires_approval`, `side_effects`) is fixed at decoration time and is
  never sent to the model.
- **Argument shape is validated before the call.** `Tool.validate_arguments`
  rejects non-mappings, non-string argument names, unknown parameter names,
  and missing required parameters, so malformed model output cannot reach the
  function. Note the limit: this is structural, not type validation (see
  below).
- **Conversations are isolated by construction.** An `Agent` holds no
  conversation state; each `Session` owns its own memory and usage counters.
  One user's history cannot leak into another's through a shared agent, and
  the separation is structural rather than lock-based. `Agent.session(...)`
  per user; `agent.run()` is one persistent conversation, so do not use it to
  serve several. Session `metadata` reaches `ToolPolicy` as
  `context["metadata"]`, which is how a policy authorizes per user.
- **Authorization decisions are audited.** Each decision is emitted to
  `Callback.on_tool_audit` before the tool runs, so the record survives a tool
  that hangs or crashes, and refusals are written to the module logger even
  when no callback is attached. Audit events carry arguments verbatim — redact
  them in your sink if your tools take secrets.
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

- **Argument *types* are not enforced or coerced.** Validation is structural
  (names and required parameters), not semantic: a parameter annotated `int`
  will still receive `"3"` if the model sends a string, because silently
  repairing that would hide real model errors. Annotate the parameter with a
  Pydantic model when you need full validation, or check inside the tool.
- **`Tool.run()` and calling a tool directly are not policed.** The policy
  layer governs *model* intent. Application code holding a `Tool` object is
  already trusted and can call it however it likes; don't hand a raw `Tool` to
  something you wouldn't hand the underlying function.
- **A tool that declares no `permissions` requires none.** `PermissionPolicy`
  narrows the agent's tool list, it does not replace it: an unlabelled tool
  passes. Declare permissions on every tool you intend to gate — the audit log
  records the permissions of each call, which makes the gaps visible.
- **The policy layer is not a sandbox.** It decides *whether* a function runs,
  not what that function can then do. A tool that shells out or writes files
  still needs OS-level confinement (see `examples/coder.py`).
- **Tool error text is fed back to the model and stored in memory.** If a tool
  raises an exception whose message contains a secret, that secret enters the
  transcript. Catch and sanitise inside tools that handle credentials.
- **RAG context is injected into the prompt.** Retrieved documents are model
  input, so a poisoned corpus is a prompt-injection vector. Trust your corpus
  the way you trust user input.
