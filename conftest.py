"""Make the repository root importable so `pytest` works from any invocation.

Without this, `pytest tests/` fails with ModuleNotFoundError: no module named
'src', because pytest puts the test directory on sys.path rather than the
project root. Only `python -m pytest` would work otherwise.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running integration test (tiny-overfit); run with -m slow"
    )
