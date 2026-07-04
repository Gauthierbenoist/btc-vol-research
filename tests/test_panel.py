"""Tests panel marché."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.config import MarketConfig  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402


def _row(*, strike: float, option_type: str, bid=0.01, ask=0.012, mid=0.011) -> dict:
    return {
        "time_to_expiry_years": 0.25,
        "underlying_price": 100_000.0,
        "strike": strike,
        "bid_price": bid,
        "ask_price": ask,
        "mid_price": mid,
        "mark_iv": 0.5,
        "mark_price": mid,
        "option_type": option_type,
        "open_interest": 1.0,
        "maturity_date": pd.Timestamp("2026-09-01"),
    }


def test_drop_phantom_bid_ask_filters_nan_quotes():
    df = pd.DataFrame(
        [
            _row(strike=105_000.0, option_type="call"),
            {**_row(strike=105_000.0, option_type="call"), "bid_price": np.nan},
            {**_row(strike=95_000.0, option_type="put"), "ask_price": np.nan},
        ]
    )
    cfg_on = MarketConfig(drop_phantom_bid_ask=True)
    panel_on = build_market_panel(df, cfg_on)
    assert len(panel_on) == 1

    cfg_off = MarketConfig(drop_phantom_bid_ask=False)
    panel_off = build_market_panel(df, cfg_off)
    assert len(panel_off) == 3
