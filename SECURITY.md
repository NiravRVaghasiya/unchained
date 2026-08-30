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

## Threat model

Read this before putting an agent in front of anything that matters. It
describes what Unchained does and does not defend against, and every claim
below is enforced in code — the boundaries are itemised in
[Boundaries enforced in code](#boundaries-enforced-in-code), and the gaps in
[Known non-boundaries](#known-non-boundaries-by-design).

### The shape of the problem

An agent takes instructions from you, input from a user, documents from a
corpus, and results from tools — then a language model decides what to do
next. The model is not a security boundary and cannot be made into one. So:

> **Assume the model will eventually be persuaded to request the wrong thing,
> and make sure the code refuses.**

That is the whole design. Every guarantee Unchained offers is a Python check
that holds regardless of what the model was told, believed, or produced.

### Trust levels

| Level | Source | Trusted for instructions? |
|---|---|---|
| 1 | System prompt (`system_prompt`) | yes — it is your code |
| 2 | Application config: tools, `ToolPolicy`, `Budget`, agent names | yes — it is your code |
| 3 | User input | as a *request*, never as configuration |
| 4 | **Retrieved documents, tool results** | **no — data only** |
| 5 | **Model output**, including the memory summary | **no — data only** |

Levels 4 and 5 are the dangerous ones. A document may have been written by
anyone who can add to your corpus; a tool result may relay text from a system
you do not control; a summary is generated *from* both.

### Ten things that are true of this framework

1. **Tools execute with the full privileges of the host Python process.** A
   tool is an ordinary Python function called in-process. It can read any file
   the process can read, open sockets, spawn processes, and reach into
   Unchained's own objects. Installing a tool is trusting code, exactly like
   adding a dependency.

2. **Unchained is not a sandbox.** Nothing here confines what a tool does once
   it runs. `ToolPolicy` decides *whether* a function is called;
   `max_output_size` bounds what it *sends onward*; `timeout` bounds how long
   the agent *waits*. None of them constrain the function's behaviour.

3. **An LLM tool call is an untrusted request, not a decision.** Every
   model-requested call takes one path — `Agent._execute` — which locates the
   tool, validates the arguments against the Python signature, asks the
   policy, takes approval if required, executes, and audits. A call naming a
   tool the agent does not hold never reaches a function.

4. **Retrieved documents and tool results are data, never instructions.** Both
   are wrapped in a fence carrying a random per-agent marker before reaching a
   provider, and any occurrence of that marker is stripped from the content,
   so it cannot close its own block. The memory summary is fenced too — it is
   spliced into the *system* message, so unfenced it would be a path from a
   tool result into your instructions.

5. **Prompt injection cannot be prevented by prompting, and this framework
   does not claim to prevent it.** The system prompt states the data boundary;
   that is the weakest layer and a persuasive document may still talk a model
   into something. Fencing constrains *structure*, not persuasion. What
   contains the damage is that a forged fence widens nothing: `permissions` is
   a `frozenset` fixed at decoration, and `ToolPolicy` sees the tool, the
   validated arguments and a context built from your code — never from
   retrieved or returned text.

6. **Least privilege is the application's job.** Give an agent the smallest
   set of tools that lets it do its work, and gate them:
   `PermissionPolicy(granted={"db:read"})`. A tool that declares no
   `permissions` requires none, so declare them on everything you intend to
   gate. Routing is *not* a privilege boundary — an injection can still steer
   a router to a different registered agent, so give a privileged agent a
   policy that refuses rather than relying on it being unreachable.

7. **Side-effecting tools need a policy, and usually approval.** Mark them
   `@tool(side_effects=True, permissions={...})` and, for anything
   irreversible, `requires_approval=True`. Approval is an application callback
   that model output cannot reach or influence; with no approver configured
   the call is refused rather than allowed. Note that `side_effects=True` is
   descriptive — it does **not** gate, serialise, or approve anything on its
   own.

8. **Network-facing tools must impose their own restrictions.** Unchained
   performs no URL validation, no allow-listing, and no SSRF protection
   anywhere. A tool that fetches a model-supplied URL will happily reach
   `localhost`, link-local metadata endpoints, and internal hosts. If a tool
   takes a URL, validate the scheme and resolved address inside the tool, and
   always pass an explicit `timeout=` — a tool timeout does not abort a socket
   read.

9. **Keep secrets out of prompts.** Unchained never puts your API key in a
   message: keys travel only in the `Authorization` / `x-api-key` headers, and
   no placeholder is sent when none is configured. But secrets reach the
   transcript by other routes you control — a tool that *returns* one, a tool
   exception whose message contains one, or `Agent(event_payloads=True)`.
   Audit events carry tool arguments verbatim by design; redact them in your
   sink. Note also that `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` /
   `OLLAMA_BASE_URL` redirect where your key is sent, and the endpoint appears
   in `LLMResponse.metadata` — treat that environment as sensitive.

10. **Running untrusted code requires external isolation.** If a tool executes
    model-generated or user-supplied code, a subprocess is not enough — it can
    still read files and reach the network. Use a container, gVisor/seccomp,
    network egress rules and resource limits. `examples/coder.py` runs
    snippets in an isolated subprocess with a timeout and says plainly that
    this is a demo, not a sandbox.

### Production checklist

- [ ] Every tool is one an attacker may invoke with arguments of their choosing.
- [ ] Agents hold the smallest tool set that does the job.
- [ ] A `ToolPolicy` is configured; `permissions` are declared on every tool
      you intend to gate.
- [ ] Irreversible tools are `requires_approval=True` with an `approve=`
      callback wired up.
- [ ] Tools taking a URL validate scheme and resolved address, and pass
      `timeout=`.
- [ ] Tools that could return a secret redact it; audit and event sinks redact
      arguments.
- [ ] Tools executing code run under OS-level isolation, not just a subprocess.
- [ ] `Budget` caps tokens, tool calls, wall clock and — if priced — spend.
- [ ] `Agent(tool_timeout=...)` and `max_tool_output_size` are set, and you
      accept that a timed-out side effect has an **unknown** outcome.
- [ ] Tools whose concurrent calls would race are
      `@tool(concurrency="exclusive")`.
- [ ] One `Session` per user; `agent.run()` is a single conversation.
- [ ] The RAG corpus is trusted the way user input is trusted.
- [ ] Audit events (`on_tool_audit`) are retained somewhere you can query.
- [ ] `Router(strict=True)` if routing decisions matter.
- [ ] The Streamlit UI, if exposed, is behind authentication.
- [ ] Keys come from the environment; `.env` and real keys are never committed.


## How untrusted text is kept separate

Retrieved documents, tool results and the memory summary are fenced before
they reach a provider:

```
<<document-9f2a1c4b7e8d0a35>>
[score=0.87]
...retrieved text...
<</document-9f2a1c4b7e8d0a35>>
```

The marker carries a random per-agent value and every occurrence of it is
stripped from the text being fenced, so content cannot close its own block and
continue at instruction level. Documents are stored *beside* the user's turn
rather than spliced into it, so memory records what the user actually said and
fencing happens per-send with the current agent's marker — a conversation
reloaded from disk never carries a dead agent's markers.

Two limits, stated rather than left to be discovered:

- The marker is per agent, not per run, so it appears in every prompt for that
  agent's life. An attacker who can both observe a response echoing it *and
  then* plant new content could forge a block. Rotating per run would close
  this and make the system prompt uncacheable.
- Fencing constrains structure, not persuasion. A document that simply argues
  convincingly is unaffected by any delimiter.

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
- **Event payloads are excluded by default.** `AgentEvent` carries shapes
  and sizes - message counts, character counts, durations, token usage - not
  prompts, tool arguments, tool results or answers. An event stream usually
  ends up somewhere with a longer retention and a wider audience than the
  application itself, and those fields are where personal data and
  credentials are. `Agent(event_payloads=True)` opts in deliberately; audit
  events (`on_tool_audit`) carry arguments verbatim regardless, as they
  always have.
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

## Known non-boundaries, by design

Things Unchained deliberately does not do. None of these is a bug; each is a
place where the guarantee stops and yours begins.


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
- **Concurrent tool calls are not serialised unless you say so.** A model can
  request several tools in one turn and they run concurrently by default. A
  tool whose concurrent calls would race — a read-modify-write, an append, a
  non-reentrant client — must be marked `@tool(concurrency="exclusive")`.
  `side_effects=True` does **not** do this: it describes the tool to a policy
  and the audit log, and says nothing about thread safety. Exclusivity holds
  within one turn; two concurrent sessions can still overlap, so
  process-wide exclusion needs a lock inside the tool.
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
- **Run budgets bound cost and runaway loops, not behaviour.** `Budget`
  limits what one run may consume — iterations, tool calls, tokens, tool
  output, wall clock, estimated spend — and stops the run deterministically
  when a limit is reached. Every model-requested tool call is claimed against
  it on the single path such calls take, before the tool is located or
  authorized, so unknown and denied calls count too. It is resource
  governance: it does not decide *whether* a tool may run (that is
  `ToolPolicy`) and it cannot interrupt a call already in flight.
- **Cost is an estimate, never an invoice.** It is computed from the
  provider's reported token counts and rates you supply. Unchained ships no
  price table, because a stale one would silently under-report. A model with
  no pricing entry raises rather than being costed as zero when `max_cost` is
  set, and sets `cost_is_complete = False` when it is not.
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
