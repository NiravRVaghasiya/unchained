"""Tool authorization: read-only, side-effecting, and approval-required tools.

An LLM choosing a tool is a *request*. This example shows the three kinds of
tool you actually end up with in production, and how a `ToolPolicy` grants or
refuses each request in Python - before the function runs.

Nothing here relies on the model behaving. The system prompt says nothing
about which tools are safe, and the agent is deliberately handed every tool,
including ones it is not allowed to use: the policy is what stops it.

Run (no API key needed - it uses MockLLM):
    python examples/policy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any, Dict

from unchained import Agent, Callback, MockLLM, PermissionPolicy, tool

# ---------------------------------------------------------------------------
# 1. A read-only tool. Needs no permission, changes nothing.
# ---------------------------------------------------------------------------
ORDERS = {"A-1": {"customer": "ada", "total": 42.00, "status": "shipped"}}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    order = ORDERS.get(order_id)
    return str(order) if order else f"no order {order_id}"


# ---------------------------------------------------------------------------
# 2. A side-effecting tool. Gated by a permission the agent must be granted.
#
# `side_effects=True` is descriptive - it tells a policy (and the audit log)
# that this call changes something. `permissions` is what actually gates it.
# ---------------------------------------------------------------------------
@tool(permissions={"billing:write"}, side_effects=True)
def apply_refund(order_id: str, amount: float) -> str:
    """Refund part or all of an order."""
    ORDERS[order_id]["status"] = "refunded"
    return f"refunded ${amount:.2f} on {order_id}"


# ---------------------------------------------------------------------------
# 3. An approval-required tool. Even a granted agent must ask a human.
#
# `allowed` adds a per-call check on the arguments themselves, so the rule can
# depend on *what* is being deleted, not just on who is asking.
# ---------------------------------------------------------------------------
@tool(
    permissions={"account:delete"},
    side_effects=True,
    requires_approval=True,
    allowed=lambda arguments, context: arguments["customer"] != "root",
)
def delete_account(customer: str) -> str:
    """Permanently delete a customer account."""
    return f"deleted {customer}"


TOOLS = [lookup_order, apply_refund, delete_account]


class AuditLog(Callback):
    """Record every authorization decision the agent makes."""

    def __init__(self) -> None:
        self.events: list = []

    def on_tool_audit(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        mark = "OK " if event["decision"] in ("allowed", "approved") else "NO "
        detail = f" - {event['reason']}" if event["reason"] else ""
        print(f"    {mark} {event['decision']:<18} {event['tool']}{detail}")


def approve_from_console(request: Dict[str, Any]) -> bool:
    """Approval hook. In a real app this prompts a human; here it is scripted.

    This callback is supplied by the application to `Agent(approve=...)`.
    Nothing the model emits can set it, reach it, or change its answer - which
    is the whole point: approval is application-controlled, never
    model-controlled.
    """
    print(
        f"    ?  approval requested: {request['tool']}({request['arguments']}) "
        f"permissions={request['permissions']}"
    )
    # A real implementation would block on input() or a UI event. We approve
    # anything that isn't touching the "ada" account, to show both outcomes.
    return request["arguments"].get("customer") != "ada"


def _call(name: str, **arguments: Any) -> Dict[str, Any]:
    """A tool call exactly as a model would emit it."""
    return {"name": name, "arguments": arguments, "id": f"call-{name}"}


def main() -> None:
    audit = AuditLog()

    # ---- A support agent that may read, but not write -------------------
    print("\nsupport agent - granted {'billing:read'}")
    support = Agent(
        MockLLM(),
        name="support",
        tools=TOOLS,  # handed every tool on purpose
        policy=PermissionPolicy(granted={"billing:read"}),
        callbacks=[audit],
    )
    print("   ", support._execute(_call("lookup_order", order_id="A-1")))
    print("   ", support._execute(_call("apply_refund", order_id="A-1", amount=42.0)))
    print("   ", support._execute(_call("delete_account", customer="ada")))

    # ---- A billing agent that may refund, with admin actions confirmed ---
    print("\nbilling agent - granted {'billing:write', 'account:delete'}, with approval")
    billing = Agent(
        MockLLM(),
        name="billing",
        tools=TOOLS,
        policy=PermissionPolicy(granted={"billing:write", "account:delete"}),
        approve=approve_from_console,
        callbacks=[audit],
    )
    print("   ", billing._execute(_call("apply_refund", order_id="A-1", amount=42.0)))
    print("   ", billing._execute(_call("delete_account", customer="bob")))
    print("   ", billing._execute(_call("delete_account", customer="ada")))
    print("   ", billing._execute(_call("delete_account", customer="root")))

    # ---- The model cannot argue its way past any of this ----------------
    print("\nwhat the model cannot do")
    print("   ", support._execute(_call("apply_refund", order_id="A-1", amount=0.01)))
    print("   ", support._execute(_call("drop_database")))
    print("   ", support._execute(_call("lookup_order", order_id="A-1", admin=True)))

    denied = [e for e in audit.events if e["decision"] not in ("allowed", "approved")]
    print(f"\n{len(audit.events)} decisions recorded, {len(denied)} refused.")
    print(f"order A-1 is now: {ORDERS['A-1']['status']}")


if __name__ == "__main__":
    main()
