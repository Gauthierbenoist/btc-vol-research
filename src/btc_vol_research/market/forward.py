"""Prix forward."""

from __future__ import annotations

import numpy as np


def forward_price(S: np.ndarray, T: np.ndarray, r: float, q: float) -> np.ndarray:
    """F = S · exp((r − q)·T)."""
    return S * np.exp((r - q) * T)
