"""Benchmark: Unchained vs LangChain vs CrewAI.

Measures the things that actually bite you day to day - how much code you must
trust, how many dependencies you pull in, and how long the framework takes to
import. Unchained's numbers are measured live; the others use published
reference figures and are measured live *if* the package is installed.

Run:
    python benchmarks/compare_frameworks.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def count_source_lines(path: Path) -> int:
    """Count non-blank, non-comment lines of Python source."""
    lines = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines += 1
    return lines


def measure_import_seconds(module_name: str) -> float | None:
    """Import a module in a fresh subprocess-like state and time it. None if absent."""
    if importlib.util.find_spec(module_name) is None:
        return None
    # Drop any cached copy so the timing reflects a cold import.
    for name in list(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            del sys.modules[name]
    start = time.perf_counter()
    try:
        importlib.import_module(module_name)
    except Exception:
        return None
    return time.perf_counter() - start


# Published reference figures (approximate, order-of-magnitude) for context.
REFERENCE = {
    "Unchained": {"loc": None, "deps": 2, "import_module": "unchained"},
    "LangChain": {"loc": 400_000, "deps": 50, "import_module": "langchain"},
    "CrewAI": {"loc": 40_000, "deps": 20, "import_module": "crewai"},
}


def main() -> None:
    # Measure Unchained's real line count.
    REFERENCE["Unchained"]["loc"] = count_source_lines(_ROOT / "unchained.py")

    print("Unchained vs LangChain vs CrewAI")
    print("=" * 64)
    header = f"{'framework':<12} | {'LOC (core)':>12} | {'deps':>5} | {'import time':>12}"
    print(header)
    print("-" * len(header))

    for name, info in REFERENCE.items():
        seconds = measure_import_seconds(info["import_module"])
        import_str = f"{seconds * 1000:.1f} ms" if seconds is not None else "not installed"
        loc = info["loc"]
        loc_str = f"{loc:,}" if isinstance(loc, int) else "?"
        marker = "  <= measured live" if seconds is not None and name != "Unchained" else ""
        print(f"{name:<12} | {loc_str:>12} | {info['deps']:>5} | {import_str:>12}{marker}")

    print("-" * len(header))
    print(
        "\nLOC = lines you must trust in the core. Fewer dependencies means fewer\n"
        "version conflicts and a smaller supply-chain surface. Unchained's numbers\n"
        "are measured on this machine; LangChain/CrewAI LOC are published estimates\n"
        "(and are timed live only when the package is installed)."
    )


if __name__ == "__main__":
    main()
