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

from unchained import LLM, Agent, Memory


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
        # check_same_thread=False so a single connection can be shared across
        # the worker threads used by Router.run_all.
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
        self.messages = self.history()[-self.max_messages :]

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

    # Wire it into an agent exactly like the built-in Memory (needs a live LLM).
    _ = Agent(LLM(provider="ollama"), memory=reloaded)
    reloaded.close()
    Path(db).unlink(missing_ok=True)


if __name__ == "__main__":
    _demo()
