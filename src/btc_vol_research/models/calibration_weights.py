"""Pondérations communes SVI / Heston."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.config import CalibrationConfig
from btc_vol_research.iv.black_scholes import bs_vega


def calibration_weights(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """w_i ∝ vega × √OI × gaussienne ATM, normalisées."""
    n = len(slice_df)
    w = np.ones(n)
    S0 = float(slice_df["S"].iloc[0])
    T = float(slice_df["T"].iloc[0])

    if cfg.use_vega_weight:
        vegas = np.array(
            [
                bs_vega(S0, float(row.K), T, r, q, float(row.iv_used))
                for row in slice_df.itertuples()
            ]
        )
        w *= np.maximum(vegas, 1e-8)

    if cfg.use_liquidity_weight:
        oi = slice_df["open_interest"].astype(float).values
        w *= np.sqrt(np.maximum(oi, 1e-8))

    lm = slice_df["log_moneyness"].values
    atm_w = np.exp(-0.5 * (lm / cfg.atm_log_moneyness_sigma) ** 2)
    # strength=0 : pas de surpoids ATM ; strength=1 : gaussienne pleine
    s = cfg.atm_gaussian_strength
    w *= (1.0 - s) + s * atm_w
    return w / (w.sum() + 1e-12) * n
