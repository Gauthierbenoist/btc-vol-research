"""Tests Heston (prix positifs, Feller)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.models.heston.params import HestonParams  # noqa: E402
from btc_vol_research.models.heston.pricer import heston_call_price, heston_iv_grid  # noqa: E402


def test_heston_call_positive():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    assert p.feller_satisfied()
    c = heston_call_price(100_000.0, 100_000.0, 0.5, p)
    assert c > 0


def test_heston_iv_grid_reuses_engine():
    import numpy as np

    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    strikes = np.array([90_000.0, 100_000.0, 110_000.0])
    iv = heston_iv_grid(100_000.0, strikes, 0.5, p, 0.0, 0.0, np.array(["put", "call", "call"]))
    assert iv.shape == (3,)
    assert np.all(np.isfinite(iv))
    assert np.all(iv > 0)
