"""Calibration Heston par tranche de maturité."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from btc_vol_research.config import AppConfig, HestonBounds
from btc_vol_research.models.calibration.errors import iv_rmse, iv_sse, iv_weighted_rmse
from btc_vol_research.models.calibration.filters import quality_filter
from btc_vol_research.models.heston.params import HestonParams
from btc_vol_research.models.heston.pricer import heston_iv_grid
from btc_vol_research.models.calibration_weights import WeightFn, calibration_weights


@dataclass
class CalibrationResult:
    slice_id: str
    params: HestonParams
    rmse_iv: float
    weighted_rmse_iv: float
    market_iv: np.ndarray
    model_iv: np.ndarray
    strikes: np.ndarray
    success: bool
    message: str


def _initial_guess(slice_df: pd.DataFrame, bounds: HestonBounds) -> HestonParams:
    atm_iv = float(slice_df.loc[slice_df["log_moneyness"].abs().idxmin(), "iv_used"])
    var = max(atm_iv**2, 0.01)
    return HestonParams(
        v0=np.clip(var, *bounds.v0),
        kappa=1.5,
        theta=np.clip(var, *bounds.theta),
        sigma=0.4,
        rho=-0.5,
    )


def calibrate_slice(
    slice_df: pd.DataFrame,
    cfg: AppConfig,
    slice_id: str | None = None,
    *,
    weight_fn: WeightFn | None = None,
) -> CalibrationResult:
    """Minimise Σ w_i (σ_model - σ_mkt)² sur une maturité."""
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.heston_bounds
    sid = slice_id or str(slice_df["slice_id"].iloc[0])

    if len(slice_df) < calib.min_strikes_per_slice:
        raise ValueError(f"Slice {sid}: seulement {len(slice_df)} strikes (min {calib.min_strikes_per_slice})")

    slice_df = quality_filter(slice_df, min_strikes=calib.min_strikes_per_slice)

    S0 = float(slice_df["S"].iloc[0])
    T = float(slice_df["T"].iloc[0])
    strikes = slice_df["K"].values.astype(float)
    market_iv = slice_df["iv_used"].values.astype(float)
    w_fn = weight_fn or calibration_weights
    weights = w_fn(slice_df, calib, market.risk_free_rate, market.dividend_yield)

    x0 = _initial_guess(slice_df, bounds).as_array()
    bnds = [bounds.v0, bounds.kappa, bounds.theta, bounds.sigma, bounds.rho]

    def objective(x: np.ndarray) -> float:
        params = HestonParams.from_array(x)
        if not params.feller_satisfied():
            return calib.feller_penalty
        try:
            model_iv = heston_iv_grid(
                S0,
                strikes,
                T,
                params,
                market.risk_free_rate,
                market.dividend_yield,
                slice_df["option_type"].values,
            )
        except Exception:
            return calib.feller_penalty
        if np.any(~np.isfinite(model_iv)):
            return calib.feller_penalty
        err = (model_iv - market_iv) ** 2
        return iv_sse(market_iv, model_iv, weights)

    res = minimize(objective, x0, method=calib.optimizer, bounds=bnds, options={"maxiter": 200})

    params_opt = HestonParams.from_array(res.x)
    model_iv = heston_iv_grid(
        S0,
        strikes,
        T,
        params_opt,
        market.risk_free_rate,
        market.dividend_yield,
        slice_df["option_type"].values,
    )
    err = model_iv - market_iv
    rmse = iv_rmse(market_iv, model_iv)
    w_rmse = iv_weighted_rmse(market_iv, model_iv, weights)

    return CalibrationResult(
        slice_id=sid,
        params=params_opt,
        rmse_iv=rmse,
        weighted_rmse_iv=w_rmse,
        market_iv=market_iv,
        model_iv=model_iv,
        strikes=strikes,
        success=bool(res.success),
        message=str(res.message),
    )


def calibrate_all_slices(
    panel: pd.DataFrame,
    cfg: AppConfig,
    *,
    weight_fn: WeightFn | None = None,
) -> list[CalibrationResult]:
    results: list[CalibrationResult] = []
    for sid, g in panel.groupby("slice_id"):
        if len(g) < cfg.calibration.min_strikes_per_slice:
            continue
        try:
            results.append(calibrate_slice(g, cfg, str(sid), weight_fn=weight_fn))
        except Exception:
            continue
    return results
