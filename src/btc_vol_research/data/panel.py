"""Préparation du panel marché pour smiles et calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.config import AppConfig, MarketConfig
from btc_vol_research.iv.black_scholes import forward_price, implied_volatility
from btc_vol_research.iv.conventions import apply_smile_convention, relative_spread


def build_market_panel(df: pd.DataFrame, cfg: AppConfig | MarketConfig | None = None) -> pd.DataFrame:
    """
    Enrichit le snapshot : forward, log-moneyness, IV mid (inversion BS), filtres qualité.
    """
    market = cfg.market if isinstance(cfg, AppConfig) else (cfg or MarketConfig())

    out = df.copy()
    out["option_price"] = out["mid_price"].where(out["mid_price"].notna(), out["mark_price"])
    out["T"] = out["time_to_expiry_years"].astype(float)
    out["S"] = out["underlying_price"].astype(float)
    out["K"] = out["strike"].astype(float)

    out["forward"] = forward_price(
        out["S"], out["T"], market.risk_free_rate, market.dividend_yield
    )
    out["log_moneyness"] = np.log(out["K"] / out["forward"])

    out["rel_spread"] = relative_spread(out["bid_price"], out["ask_price"], out["option_price"])
    out["iv_mark"] = out["mark_iv"].astype(float)
    out["iv_mid"] = implied_volatility(
        out["option_price"],
        out["S"],
        out["K"],
        out["T"],
        market.risk_free_rate,
        market.dividend_yield,
        out["option_type"],
    )
    # IV mid fiable si proche de mark_iv ; sinon on garde mark (Deribit)
    iv_mid_ok = out["iv_mid"].notna()
    if iv_mid_ok.any():
        rel_diff = (out["iv_mid"] - out["iv_mark"]).abs() / out["iv_mark"].clip(lower=1e-4)
        iv_mid_ok = iv_mid_ok & (rel_diff < 0.25)
    out["iv_used"] = out["iv_mid"].where(iv_mid_ok, out["iv_mark"])

    mask = (
        (out["T"] >= market.min_time_to_expiry_days / 365.25)
        & (out["T"] <= market.max_time_to_expiry_years)
        & (out["open_interest"] >= market.min_open_interest)
        & (out["iv_used"].notna())
        & (out["iv_used"] > 0)
        & (out["rel_spread"].isna() | (out["rel_spread"] <= market.max_relative_spread))
    )
    out = out.loc[mask].copy()
    out = apply_smile_convention(out, market.smile_convention)
    out["slice_id"] = out["maturity_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)
