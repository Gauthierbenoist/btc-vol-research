"""Tests pricer Merton vectorise."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.models.merton import (  # noqa: E402
    MertonParams,
    merton_iv_panel,
    merton_option_price,
    merton_option_prices,
)


def test_vectorized_prices_match_scalar():
    p = MertonParams(sigma=0.36, lambda_jump=0.15, mu_jump=-0.4, sigma_jump=0.37)
    cases = [
        (100_000.0, 95_000.0, 0.12, "call"),
        (100_000.0, 105_000.0, 0.55, "put"),
        (100_000.0, 100_000.0, 0.25, "call"),
    ]
    for s, k, t, opt in cases:
        scalar = merton_option_price(s, k, t, p, option_type=opt)
        vec = merton_option_prices(
            np.array([s]),
            np.array([k]),
            np.array([t]),
            p,
            option_types=np.array([opt]),
        )[0]
        assert abs(scalar - vec) < 1e-8


def test_vectorized_iv_panel_finite():
    p = MertonParams(sigma=0.4, lambda_jump=0.2, mu_jump=-0.3, sigma_jump=0.3)
    panel = pd.DataFrame(
        {
            "S": [100_000.0, 100_000.0, 100_000.0],
            "K": [90_000.0, 100_000.0, 110_000.0],
            "T": [0.2, 0.4, 0.6],
            "option_type": ["put", "call", "call"],
        }
    )
    iv = merton_iv_panel(panel, p, 0.0, 0.0)
    assert iv.shape == (3,)
    assert np.all(np.isfinite(iv))
    single = pd.DataFrame({"S": [100_000.0], "K": [100_000.0], "T": [0.4], "option_type": ["call"]})
    assert abs(iv[1] - float(merton_iv_panel(single, p, 0.0, 0.0)[0])) < 1e-6
