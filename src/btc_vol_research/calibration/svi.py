"""Calibration SVI par tranche (baseline smile)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from btc_vol_research.calibration.results import SliceCalibrationResult
from btc_vol_research.calibration.slices import atm_row, require_min_points
from btc_vol_research.calibration.slices import calibrate_all_slices as _run_all_slices
from btc_vol_research.calibration.weights import WeightFn, calibration_weights
from btc_vol_research.config import AppConfig, SVIBounds
from btc_vol_research.models.svi import SVIParams, svi_iv_from_log_moneyness, svi_total_variance


def _initial_guess(slice_df: pd.DataFrame, T: float, bounds: SVIBounds) -> SVIParams:
    atm = atm_row(slice_df)
    atm_iv = float(atm["iv_used"])
    m = float(atm["log_moneyness"])
    w_atm = max(atm_iv**2 * T, 1e-4)
    return SVIParams(
        a=np.clip(0.8 * w_atm, *bounds.a),
        b=np.clip(0.15 * w_atm, *bounds.b),
        rho=-0.4,
        m=np.clip(m, *bounds.m),
        sigma=np.clip(0.15, *bounds.sigma),
    )


def calibrate_slice(
    slice_df: pd.DataFrame,
    cfg: AppConfig,
    slice_id: str | None = None,
    *,
    weight_fn: WeightFn | None = None,
) -> SliceCalibrationResult[SVIParams]:
    """Minimise Σ w_i (σ_SVI(k_i) - σ_mkt,i)²."""
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.svi_bounds
    sid = slice_id or str(slice_df["slice_id"].iloc[0])
    slice_df = require_min_points(slice_df, calib.min_strikes_per_slice, sid)

    T = float(slice_df["T"].iloc[0])
    k = slice_df["log_moneyness"].values.astype(float)
    market_iv = slice_df["iv_used"].values.astype(float)
    w_fn = weight_fn or calibration_weights
    weights = w_fn(slice_df, calib, market.risk_free_rate, market.dividend_yield)

    x0 = _initial_guess(slice_df, T, bounds).as_array()
    bnds = [bounds.a, bounds.b, bounds.rho, bounds.m, bounds.sigma]
    penalty = calib.feller_penalty  # réutilisé comme pénalité générique

    def objective(x: np.ndarray) -> float:
        params = SVIParams.from_array(x)
        if not params.butterfly_ok():
            return penalty
        w = svi_total_variance(k, params)
        if np.any(w < 0):
            return penalty
        model_iv = svi_iv_from_log_moneyness(k, T, params)
        if np.any(~np.isfinite(model_iv)):
            return penalty
        err = (model_iv - market_iv) ** 2
        return float(np.sum(weights * err))

    res = minimize(objective, x0, method=calib.optimizer, bounds=bnds, options={"maxiter": 300})

    params_opt = SVIParams.from_array(res.x)
    model_iv = svi_iv_from_log_moneyness(k, T, params_opt)
    err = model_iv - market_iv
    rmse = float(np.sqrt(np.mean(err**2)))

    return SliceCalibrationResult(
        slice_id=sid,
        params=params_opt,
        rmse_iv=rmse,
        market_iv=market_iv,
        model_iv=model_iv,
        log_moneyness=k,
        T=T,
        success=bool(res.success),
        message=str(res.message),
    )


def calibrate_all_slices(
    panel: pd.DataFrame,
    cfg: AppConfig,
    *,
    weight_fn: WeightFn | None = None,
) -> list[SliceCalibrationResult[SVIParams]]:
    return _run_all_slices(panel, cfg, calibrate_slice, weight_fn=weight_fn)
