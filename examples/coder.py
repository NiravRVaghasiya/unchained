"""Code-execution agent.

The agent writes small Python snippets to compute answers. Each snippet runs in
a separate process with a timeout, so an infinite loop or crash can't take down
the host process, and output is fed back into the loop.

WARNING: a subprocess is NOT a security sandbox. The code can still read files
and reach the network. Before exposing this to untrusted input, run it inside a
real sandbox (container, gVisor/seccomp, network egress rules, resource limits).

Run:
    python examples/coder.py "What are the first 10 Fibonacci numbers?"
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unchained import LLM, Agent, tool

EXEC_TIMEOUT_SECONDS = 10


@tool
def run_python(code: str) -> str:
    """Execute a snippet of Python in a separate process and return its output."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {EXEC_TIMEOUT_SECONDS}s."
    if proc.returncode != 0:
        return f"Error (exit {proc.returncode}):\n{(proc.stderr or proc.stdout).strip()}"
    output = (proc.stdout or "").strip()
    return output or "(no output - remember to print your result)"


def build_agent(provider: str = "ollama") -> Agent:
    return Agent(
        LLM(provider=provider),
        name="coder",
        description="Writes and runs Python to solve computational problems.",
        system_prompt=(
            "You are a Python coding assistant. When a question needs "
            "computation, write a short script that prints the answer and run "
            "it with run_python. Then explain the result."
        ),
        tools=[run_python],
    )


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What are the first 10 Fibonacci numbers?"
    print(build_agent().run(task))
