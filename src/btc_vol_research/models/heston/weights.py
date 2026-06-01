"""Pondérations pour la calibration (vega, liquidité, ATM)."""

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
    """
    w_i ∝ w_vega × w_liquidity × w_ATM, normalisées pour sommer à n.
    """
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
        vegas = np.maximum(vegas, 1e-8)
        w *= vegas

    if cfg.use_liquidity_weight:
        oi = slice_df["open_interest"].astype(float).values
        w *= np.sqrt(np.maximum(oi, 1e-8))

    lm = slice_df["log_moneyness"].values
    atm_w = np.exp(-0.5 * (lm / cfg.atm_log_moneyness_sigma) ** 2)
    w *= atm_w

    w = w / (w.sum() + 1e-12) * n
    return w
