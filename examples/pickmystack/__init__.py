"""PickMyStack - the flagship Unchained multi-agent application.

Given a use case and constraints, three specialist agents (cost, fit, trend)
evaluate AI-stack options in parallel and a synthesizer ranks the best choices.
"""

import sys
from pathlib import Path

# Make the top-level `unchained.py` importable no matter where we're launched.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
