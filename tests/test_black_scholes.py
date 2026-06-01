"""Tests Black-Scholes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.iv.black_scholes import bs_call_price, implied_volatility  # noqa: E402


def test_iv_roundtrip():
    S, K, T, r, q, sigma = 100_000.0, 105_000.0, 0.25, 0.0, 0.0, 0.65
    price = bs_call_price(S, K, T, r, q, sigma)
    iv = implied_volatility(
        np.array([price]),
        np.array([S]),
        np.array([K]),
        np.array([T]),
        r,
        q,
        np.array(["call"]),
    )[0]
    assert abs(iv - sigma) < 1e-4
