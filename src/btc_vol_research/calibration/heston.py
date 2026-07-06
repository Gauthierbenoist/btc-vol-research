"""Calibration Heston par tranche de maturité."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from btc_vol_research.calibration.errors import iv_rmse, sse_objective
from btc_vol_research.calibration.results import SliceCalibrationResult
from btc_vol_research.calibration.slices import atm_row, require_min_points
from btc_vol_research.calibration.slices import calibrate_all_slices as _run_all_slices
from btc_vol_research.calibration.weights import WeightFn, calibration_weights
from btc_vol_research.config import AppConfig, HestonBounds
from btc_vol_research.models.heston import HestonParams, heston_iv_grid


def _initial_guess(slice_df: pd.DataFrame, bounds: HestonBounds) -> HestonParams:
    atm_iv = float(atm_row(slice_df)["iv_used"])
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
) -> SliceCalibrationResult[HestonParams]:
    """Minimise Σ w_i (σ_model - σ_mkt)² sur une maturité."""
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.heston_bounds
    sid = slice_id or str(slice_df["slice_id"].iloc[0])

    slice_df = require_min_points(slice_df, calib.min_strikes_per_slice, sid)

    S0 = float(slice_df["S"].iloc[0])
    T = float(slice_df["T"].iloc[0])
    strikes = slice_df["K"].values.astype(float)
    market_iv = slice_df["iv_used"].values.astype(float)
    option_types = slice_df["option_type"].values
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
                option_types,
            )
        except Exception:
            return calib.feller_penalty
        if np.any(~np.isfinite(model_iv)):
            return calib.feller_penalty
        return sse_objective(market_iv, model_iv, weights)

    res = minimize(objective, x0, method=calib.optimizer, bounds=bnds, options={"maxiter": 200})

    params_opt = HestonParams.from_array(res.x)
    model_iv = heston_iv_grid(
        S0,
        strikes,
        T,
        params_opt,
        market.risk_free_rate,
        market.dividend_yield,
        option_types,
    )

    return SliceCalibrationResult(
        slice_id=sid,
        params=params_opt,
        rmse_iv=iv_rmse(market_iv, model_iv),
        market_iv=market_iv,
        model_iv=model_iv,
        log_moneyness=slice_df["log_moneyness"].values.astype(float),
        T=T,
        success=bool(res.success),
        message=str(res.message),
    )


def calibrate_all_slices(
    panel: pd.DataFrame,
    cfg: AppConfig,
    *,
    weight_fn: WeightFn | None = None,
) -> list[SliceCalibrationResult[HestonParams]]:
    return _run_all_slices(panel, cfg, calibrate_slice, weight_fn=weight_fn)
