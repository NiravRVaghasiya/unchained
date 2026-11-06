"""Persistent memory backed by SQLite.

Demonstrates the `Memory` extension point: `SQLiteMemory` keeps the same
sliding-window behaviour as the base class, but every message is also written
to a SQLite database and reloaded on startup, so a conversation survives across
process restarts. Sessions are namespaced by `session_id`.

Run the demo:
    python examples/sqlite_memory.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unchained import LLM, Agent, Memory, MockLLM


class SQLiteMemory(Memory):
    """A `Memory` that persists messages to a SQLite database.

    The in-memory sliding window (inherited from `Memory`) still governs what
    is sent to the model each turn; the database keeps the full, durable
    history so it can be reloaded later or inspected.
    """

    def __init__(
        self,
        db_path: str = "unchained_memory.db",
        session_id: str = "default",
        max_messages: int = 20,
        llm: Optional[LLM] = None,
    ):
        super().__init__(max_messages=max_messages, llm=llm)
        self.db_path = db_path
        self.session_id = session_id
        # check_same_thread=False: Session.events() and arun() run a turn on a
        # worker thread, and a server may handle the next turn on another
        # thread. One Session must still never be driven by two threads at once.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                   id       INTEGER PRIMARY KEY AUTOINCREMENT,
                   session  TEXT NOT NULL,
                   role     TEXT NOT NULL,
                   content  TEXT NOT NULL,
                   extra    TEXT NOT NULL DEFAULT '{}'
               )"""
        )
        self._conn.commit()
        self._reload_window()

    def add(self, role: str, content: Any, **extra: Any) -> None:
        self._conn.execute(
            "INSERT INTO messages (session, role, content, extra) VALUES (?, ?, ?, ?)",
            (self.session_id, role, str(content), json.dumps(extra)),
        )
        self._conn.commit()
        super().add(role, content, **extra)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM messages WHERE session = ?", (self.session_id,))
        self._conn.commit()
        super().clear()

    def history(self) -> List[Dict[str, Any]]:
        """Return the full persisted history for this session (oldest first)."""
        rows = self._conn.execute(
            "SELECT role, content, extra FROM messages WHERE session = ? ORDER BY id",
            (self.session_id,),
        ).fetchall()
        return [
            {"role": role, "content": content, **json.loads(extra)} for role, content, extra in rows
        ]

    def _reload_window(self) -> None:
        """Load the most recent messages back into the sliding window."""
        history = self.history()
        # Like Memory itself, keep at least the most recent message.
        start = max(0, len(history) - max(1, self.max_messages))
        self.messages = history[self._tool_group_start(history, start) :]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()


def _demo() -> None:
    db = "unchained_memory_demo.db"
    Path(db).unlink(missing_ok=True)  # start clean for the demo

    memory = SQLiteMemory(db_path=db, session_id="alice")
    memory.add("user", "Remember that my favourite language is Python.")
    memory.add("assistant", "Got it - Python is your favourite.")
    print(f"Stored {len(memory.history())} messages in {db}")
    memory.close()

    # A fresh instance reloads the same session from disk.
    reloaded = SQLiteMemory(db_path=db, session_id="alice")
    print("Reloaded window:", [m["content"] for m in reloaded.get()])

    session = Agent(MockLLM(reply="Python, as you told me.")).session(memory=reloaded)
    print("Agent says:", session.run("What is my favourite language?"))
    print(f"{len(reloaded.history())} messages now persisted")
    reloaded.close()
    Path(db).unlink(missing_ok=True)


if __name__ == "__main__":
    _demo()
