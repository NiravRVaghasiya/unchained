"""RAG retrieval tool for the PickMyStack agents.

Loads every Markdown file in ``../knowledge`` into an Unchained :class:`RAG`
index (split into paragraph-sized chunks) and exposes a search tool over it.
The index is built once and cached.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unchained import RAG, tool

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"

_rag: Optional[RAG] = None


def _chunk(text: str) -> list:
    """Split a document into paragraph chunks, keeping meaningful ones."""
    chunks, current = [], []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            chunks.append("\n".join(current).strip())
            current = []
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if len(c) > 40]


def get_rag() -> RAG:
    """Build (once) and return the shared knowledge-base RAG index."""
    global _rag
    if _rag is not None:
        return _rag

    rag = RAG()
    texts, metas = [], []
    if KNOWLEDGE_DIR.is_dir():
        for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
            content = md.read_text(encoding="utf-8")
            for chunk in _chunk(content):
                texts.append(chunk)
                metas.append({"source": md.name})
    if texts:
        rag.add_many(texts, metas)
    _rag = rag
    return _rag


@tool
def search_knowledge(query: str) -> str:
    """Search the AI-stack knowledge base (frameworks, models, deployment).

    Returns the most relevant passages with their source file. Use this to
    ground recommendations in documented facts rather than guessing.
    """
    hits = get_rag().search(query, top_k=3)
    if not hits:
        return "No relevant knowledge found."
    blocks = []
    for hit in hits:
        source = hit["metadata"].get("source", "?")
        blocks.append(f"[{source} | score {hit['score']:.2f}]\n{hit['text']}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(f"Indexed {len(get_rag())} chunks from {KNOWLEDGE_DIR}")
    print()
    print(search_knowledge("cheapest way to run an agent on a tight budget"))
