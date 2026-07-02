"""Tests Merton jump-diffusion."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.config import AppConfig, MertonBounds  # noqa: E402
from btc_vol_research.models.merton.calibrate import calibrate_global  # noqa: E402
from btc_vol_research.models.merton.params import MertonParams  # noqa: E402
from btc_vol_research.models.merton.pricer import merton_iv_row, merton_option_price  # noqa: E402


def test_merton_call_positive():
    p = MertonParams(sigma=0.5, lambda_jump=0.5, mu_jump=-0.1, sigma_jump=0.2)
    c = merton_option_price(100_000.0, 100_000.0, 0.5, p)
    assert c > 0


def test_merton_params_valid():
    p = MertonParams(sigma=0.4, lambda_jump=1.0, mu_jump=-0.05, sigma_jump=0.15)
    assert p.is_valid()
    assert p.jump_compensation() > -1


def test_merton_put_call_parity_and_iv_consistency():
    p = MertonParams(sigma=0.36, lambda_jump=0.15, mu_jump=-0.4, sigma_jump=0.37)
    S = 100_000.0
    K = 100_000.0
    T = 0.55
    r = 0.0
    q = 0.0
    call = merton_option_price(S, K, T, p, r, q, option_type="call")
    put = merton_option_price(S, K, T, p, r, q, option_type="put")
    parity_gap = call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert abs(parity_gap) < 1e-6

    iv_call = merton_iv_row(S, K, T, p, r, q, "call")
    iv_put = merton_iv_row(S, K, T, p, r, q, "put")
    assert abs(iv_call - iv_put) < 1e-6


def test_merton_global_calibration_toy_panel():
    rows = []
    for T, sid in [(0.08, "2026-07-01"), (0.25, "2026-10-01")]:
        for lm in np.linspace(-0.2, 0.2, 6):
            rows.append(
                {
                    "slice_id": sid,
                    "T": T,
                    "S": 100_000.0,
                    "K": 100_000.0 * np.exp(lm),
                    "log_moneyness": lm,
                    "iv_used": 0.45 + 0.1 * lm**2,
                    "option_type": "call" if lm >= 0 else "put",
                }
            )
    panel = pd.DataFrame(rows)
    cfg = AppConfig(merton_bounds=MertonBounds())
    result = calibrate_global(panel, cfg)
    assert result.n_points == len(panel)
    assert len(result.slice_results) == 2
    assert result.params.is_valid()
    assert np.isfinite(result.rmse_iv)
