"""Tests Heston (prix positifs, Feller)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.models.heston.params import HestonParams  # noqa: E402
from btc_vol_research.models.heston.pricer import heston_call_price  # noqa: E402


def test_heston_call_positive():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    assert p.feller_satisfied()
    c = heston_call_price(100_000.0, 100_000.0, 0.5, p)
    assert c > 0
