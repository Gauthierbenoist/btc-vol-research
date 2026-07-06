"""Préparation du panel marché pour smiles et calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.config import AppConfig, MarketConfig
from btc_vol_research.data.filters import relative_spread, select_otm
from btc_vol_research.market.forward import forward_price


def build_market_panel(df: pd.DataFrame, cfg: AppConfig | MarketConfig | None = None) -> pd.DataFrame:
    """
    Enrichit le snapshot : forward, log-moneyness, utilise mark_iv de Deribit comme source.

    Deribit : primes bid/ask/mid/mark en BTC, underlying et strike en USD.
    """
    market = cfg.market if isinstance(cfg, AppConfig) else (cfg or MarketConfig())

    out = df.copy()
    out["T"] = out["time_to_expiry_years"].astype(float)
    out["S"] = out["underlying_price"].astype(float)
    out["K"] = out["strike"].astype(float)

    out["forward"] = forward_price(
        out["S"], out["T"], market.risk_free_rate, market.dividend_yield
    )
    out["log_moneyness"] = np.log(out["K"] / out["forward"])
    out["rel_spread"] = relative_spread(out["bid_price"], out["ask_price"], out["mid_price"])
    out["iv_used"] = out["mark_iv"].astype(float)

    mask = (
        (out["T"] >= market.min_time_to_expiry_days / 365.25)
        & (out["T"] <= market.max_time_to_expiry_years)
        & (out["open_interest"] >= market.min_open_interest)
        & (out["iv_used"].between(market.min_iv, market.max_iv))
        & (out["rel_spread"].isna() | (out["rel_spread"] <= market.max_relative_spread))
    )
    if market.drop_phantom_bid_ask:
        # options fantomes : pas de cote bid/ask exploitable
        mask &= out["bid_price"].notna() & out["ask_price"].notna()
    out = out.loc[mask].copy()
    out = select_otm(out)  # OTM systématique (calls K>F, puts K<F) — liquidité
    out["slice_id"] = out["maturity_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)
