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
- **Arguments are validated and normalised before the call.**
  `Tool.validate_arguments` checks what actually arrived against a model built
  from the Python signature — *not* from the JSON schema shown to the model,
  which is advice the model is free to ignore. Missing required arguments,
  wrong types, malformed nested structures and unknown argument names are all
  rejected, and the function is not called. Every model-facing path validates:
  `Agent` validates before consulting the policy (so a policy sees normalised,
  correctly-typed arguments), and `Tool.run()` validates again for anyone
  calling it directly.
- **Validation errors do not echo the offending value.** Pydantic's own
  message embeds `input_value=...`, and this text is returned to the model,
  stored in conversation memory and written to the audit log. Unchained
  reports only the field path and the rule that failed, and raises outside the
  `except` block so neither `__cause__` nor `__context__` retains the original
  error — a tool taking a password or a customer record cannot leak it by
  failing validation.
- **Conversations are isolated by construction.** An `Agent` holds no
  conversation state; each `Session` owns its own memory and usage counters.
  One user's history cannot leak into another's through a shared agent, and
  the separation is structural rather than lock-based. `Agent.session(...)`
  per user; `agent.run()` is one persistent conversation, so do not use it to
  serve several. Session `metadata` reaches `ToolPolicy` as
  `context["metadata"]`, which is how a policy authorizes per user.
- **The response cache does not store tool-call decisions.** With
  `LLM(cache=True)` (policy `"final_only"`) a response asking for tool calls
  is returned to the caller but never cached, so an identical later prompt
  cannot replay a decision to act without the model being consulted — a
  replay that would otherwise be invisible, since no request leaves the
  process. `cache="all"` opts out; use it only when every tool in play is
  read-only. The cache never executes anything under any policy: it decides
  what the model is taken to have said, and every tool call still passes the
  policy and approval hook.
- **The cache key covers everything that changes an answer**, including full
  tool schemas rather than tool names. Keying on names alone let two
  different tools that happen to share one — a `search` over public documents
  and a `search` over internal records — serve each other's cached responses.
  Response formats are keyed by schema for the same reason.
- **Authorization decisions are audited.** Each decision is emitted to
  `Callback.on_tool_audit` before the tool runs, so the record survives a tool
  that hangs or crashes, and refusals are written to the module logger even
  when no callback is attached. Audit events carry arguments verbatim — redact
  them in your sink if your tools take secrets.
- **Routing fails closed, against a registry.** The routing decision is
  requested as JSON constrained to the registered agent names, then resolved
  by looking the answer up in that registry — so no reply, however crafted,
  can name an agent the `Router` does not hold. A decision is valid only if
  it identifies exactly one registered agent; an empty reply, a refusal, a
  hallucinated name, malformed JSON, or a reply naming two agents all raise
  `RoutingError`. This matters because agents differ in the tools, and
  therefore the privileges, they hold: a silent fallback would let a confused
  router hand a query to a more privileged agent than the one intended. Pass
  `Router(..., fallback=agent)` to choose a default explicitly.
- **Agent descriptions cannot forge the router's agent list.** Descriptions
  are interpolated into the routing prompt, so one containing newlines could
  present extra `- name:` entries — agents that do not exist, or instructions
  attached to one that does. Whitespace is collapsed and the text is capped,
  which removes that shape; the registry check removes its effect. Names are
  also validated at construction: blank names and post-normalisation
  duplicates are rejected rather than silently unreachable or ambiguous.
- **Tool-call concurrency is bounded.** The number of tool calls in a turn is
  chosen by the model, so it is untrusted input. The executor is capped by
  `Agent(max_tool_workers=8)` rather than sized to the request, so one response
  cannot spawn an unbounded number of threads.
- **No placeholder credential is transmitted.** When no API key is configured
  the `Authorization` / `x-api-key` header is omitted rather than sent with a
  stringified `None`.

Known non-boundaries, by design:

- **Validation is only as strict as your annotations.** A parameter with no
  annotation accepts anything (the plain function would too), a tool declaring
  `**kwargs` accepts unknown argument names by design, and a type Pydantic has
  no validator for is checked with `isinstance` only. `str` also accepts any
  string — annotate with a Pydantic model, or check inside the tool, when a
  value needs to be a path under a root, a positive number, or an id the
  caller owns. Type-correct is not the same as safe.
- **`Tool.__call__` is not validated.** `my_tool(1, 2)` is your own code
  calling your own function, and Python's argument handling applies.
  `Tool.run(dict)` — the model-shaped path — always validates.
- **`Tool.run()` and calling a tool directly are not policed.** The policy
  layer governs *model* intent. Application code holding a `Tool` object is
  already trusted and can call it however it likes; don't hand a raw `Tool` to
  something you wouldn't hand the underlying function.
- **A tool that declares no `permissions` requires none.** `PermissionPolicy`
  narrows the agent's tool list, it does not replace it: an unlabelled tool
  passes. Declare permissions on every tool you intend to gate — the audit log
  records the permissions of each call, which makes the gaps visible.
- **A cache hit means no request is made, and no audit trail from the
  provider.** Cached responses are still emitted to callbacks, but if you
  reconcile agent behaviour against provider-side logs, cached turns will not
  appear there. Use `cache="none"` where that matters, and `clear_cache()`
  when the underlying facts change.
- **Routing is not an authorization boundary on its own.** A prompt
  injection — in an agent description, or in the user's query — can still
  persuade the model to choose a different *registered* agent. Python can
  guarantee the choice is one of yours, not that it is the right one. What
  bounds the damage is that the chosen agent's own `ToolPolicy` still governs
  what it may do. Do not rely on routing to keep a privileged agent
  unreachable; give it a policy that refuses.
- **Text routing replies cannot read sense.** With the default (non-`strict`)
  `Router`, prose that mentions exactly one agent in order to *reject* it
  ("not cost") resolves to that agent. Use `strict=True` where that matters.
- **The policy layer is not a sandbox.** It decides *whether* a function runs,
  not what that function can then do. A tool that shells out or writes files
  still needs OS-level confinement (see `examples/coder.py`).
- **Tool timeouts bound the agent's wait, not the tool's work.** Python cannot
  cancel a running thread. `@tool(timeout=...)` and `Agent(tool_timeout=...)`
  stop the agent hanging, but the abandoned call keeps running, keeps its
  thread, and may still complete. **After a timeout on a side-effecting tool,
  treat the effect as unknown rather than as not having happened.** Hard
  cancellation of arbitrary Python requires process isolation — run the work
  in a subprocess and kill it. A timeout is a liveness guard, not a
  containment boundary, and not a defence against a hostile tool.
- **A tool timeout does not abort network I/O.** HTTP tools must still set
  their own timeout (`requests.get(url, timeout=20)`); otherwise the request
  outlives the timeout and holds a connection.
- **Tool output limits are a context and cost boundary, not a sandbox.**
  `@tool(max_output_size=...)` and `Agent(max_tool_output_size=...)` bound
  what a tool *sends onward* into the transcript. They do not stop a tool
  reading a file, querying a database, or transmitting data itself, and a
  secret that falls inside the retained prefix is still retained and still
  reaches the model. They reduce accidental bulk propagation of sensitive
  data; they are not a defence against a tool that means harm. For that you
  need a `ToolPolicy` to decide whether the tool runs at all, and OS-level
  isolation for what it does when it does.
- **Tool error text is fed back to the model and stored in memory.** If a tool
  raises an exception whose message contains a secret, that secret enters the
  transcript. Catch and sanitise inside tools that handle credentials.
- **RAG context is injected into the prompt.** Retrieved documents are model
  input, so a poisoned corpus is a prompt-injection vector. Trust your corpus
  the way you trust user input.
