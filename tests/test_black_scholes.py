"""Tests Black-Scholes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.market.implied_vol import implied_volatility  # noqa: E402
from btc_vol_research.models.black_scholes import bs_call_price, bs_put_price  # noqa: E402


def test_iv_roundtrip_call():
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
    assert abs(iv - sigma) < 1e-6


def test_iv_roundtrip_put():
    S, K, T, r, q, sigma = 100_000.0, 95_000.0, 0.5, 0.0, 0.0, 0.55
    price = bs_put_price(S, K, T, r, q, sigma)
    iv = implied_volatility(
        np.array([price]),
        np.array([S]),
        np.array([K]),
        np.array([T]),
        r,
        q,
        np.array(["put"]),
    )[0]
    assert abs(iv - sigma) < 1e-6


def test_iv_vectorized_batch():
    S = np.array([100_000.0, 100_000.0])
    K = np.array([100_000.0, 110_000.0])
    T = np.array([0.5, 0.5])
    sigmas = np.array([0.65, 0.70])
    prices = np.array(
        [
            bs_call_price(S[0], K[0], T[0], 0.0, 0.0, sigmas[0]),
            bs_put_price(S[1], K[1], T[1], 0.0, 0.0, sigmas[1]),
        ]
    )
    ivs = implied_volatility(
        prices,
        S,
        K,
        T,
        0.0,
        0.0,
        np.array(["call", "put"]),
    )
    assert np.allclose(ivs, sigmas, atol=1e-6)
