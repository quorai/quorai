"""Ensure src.tools._yfinance_fundamentals is importable in test environments where yfinance
may not be installed. We register a MagicMock stub before the module is collected."""

import sys
from unittest.mock import MagicMock

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = MagicMock()

import src.tools._yfinance_fundamentals  # noqa: F401
