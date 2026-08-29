"""One agent, many conversations: sessions for servers and concurrency.

An `Agent` is configuration and behaviour - the LLM, the tools, the prompt,
the policy. A `Session` is one conversation's state - its memory, its token
counters, its metadata. Build the agent once, give every user a session.

This is what makes an agent safe to hold in a module-level variable and serve
from many request handlers at once: the sessions have nothing to contend over,
because none of them shares any conversational state.

Run (no API key needed - it uses MockLLM):
    python examples/sessions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading
from typing import Any, Dict, List

from unchained import Agent, Memory, MockLLM, Session


def _fake_reply(messages: List[Dict[str, Any]], tools: Any) -> Dict[str, Any]:
    """Stand in for a provider, reporting token usage so the counters move."""
    return {
        "content": f"you said: {messages[-1]['content']}",
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


# Build the agent ONCE, at import time, like you would in a web app.
# Nothing about any particular conversation lives on it.
AGENT = Agent(
    MockLLM(handler=_fake_reply),
    name="assistant",
    system_prompt="You are a helpful assistant.",
    # Every new session gets its own Memory, configured this way. A factory,
    # not an instance - handing one Memory to two sessions would merge two
    # conversations.
    memory_factory=lambda: Memory(max_messages=10),
)

# A real app would use Redis, a database, or a SQLiteMemory per user. A dict
# is enough to show the shape.
SESSIONS: Dict[str, Session] = {}
_lock = threading.Lock()


def session_for(user: str) -> Session:
    """Return this user's conversation, creating it on first contact."""
    with _lock:
        if user not in SESSIONS:
            SESSIONS[user] = AGENT.session(metadata={"user": user})
        return SESSIONS[user]


def handle_request(user: str, message: str) -> str:
    """What a web request handler would do."""
    return session_for(user).run(message)


def main() -> None:
    print("two users, one agent")
    print("   ", handle_request("alice", "my name is Alice"))
    print("   ", handle_request("bob", "my name is Bob"))

    alice, bob = session_for("alice"), session_for("bob")
    print(f"\n    alice remembers {len(alice.memory.get())} messages, bob {len(bob.memory.get())}")
    print("    alice's history:", [m["content"] for m in alice.memory.get()])
    print("    bob's history:  ", [m["content"] for m in bob.memory.get()])
    print("    neither can see the other's conversation.")

    # ---- many at once ---------------------------------------------------
    print("\n50 concurrent requests across 10 users")
    errors: List[Exception] = []

    def worker(index: int) -> None:
        try:
            handle_request(f"user{index % 10}", f"request {index}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"    {len(errors)} errors, {len(SESSIONS)} sessions")
    for user in sorted(SESSIONS):
        history = [str(m["content"]) for m in SESSIONS[user].memory.get()]
        # Every line in this session was addressed to this user's session.
        assert all("request" in h or "name is" in h or "you said" in h for h in history)
    print("    every session's history contains only its own turns")

    # ---- usage is per conversation, not per agent ------------------------
    print("\ntoken usage is per session")
    print(f"    alice: {alice.usage}")
    print(f"    total across sessions: {sum(s.usage['total_tokens'] for s in SESSIONS.values())}")
    print("    (there is no AGENT.usage across sessions - that would be shared state)")

    # ---- the default session --------------------------------------------
    print("\nAGENT.run() still works, on one persistent default session")
    AGENT.run("hello")
    AGENT.run("again")
    print(f"    default session holds {len(AGENT.memory.get())} messages across 2 calls")
    print("    it is persistent, so it is for single-conversation scripts -")
    print("    serving many users means one session each, as above.")


if __name__ == "__main__":
    main()
