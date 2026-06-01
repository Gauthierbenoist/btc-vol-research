"""Tests SVI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.models.svi.params import SVIParams  # noqa: E402
from btc_vol_research.models.svi.formula import svi_iv_from_log_moneyness, svi_total_variance  # noqa: E402


def test_svi_positive_variance():
    p = SVIParams(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    k = np.linspace(-0.5, 0.5, 20)
    w = svi_total_variance(k, p)
    assert np.all(w > 0)
    assert p.butterfly_ok()


def test_svi_iv_atm():
    p = SVIParams(a=0.08, b=0.05, rho=-0.3, m=0.0, sigma=0.15)
    T = 0.5
    iv = svi_iv_from_log_moneyness(np.array([0.0]), T, p)[0]
    w = 0.08  # approx at k=m for simple params
    assert 0.1 < iv < 1.5
