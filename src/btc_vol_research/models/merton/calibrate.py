"""Calibration globale Merton sur toute la surface IV."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from btc_vol_research.config import AppConfig, MertonBounds
from btc_vol_research.models.calibration.errors import iv_rmse, iv_sse, iv_weighted_rmse
from btc_vol_research.models.calibration.filters import quality_filter
from btc_vol_research.models.calibration.results import GlobalCalibrationResult
from btc_vol_research.models.calibration.slice_fits import build_slice_fits
from btc_vol_research.models.calibration_weights import WeightFn, build_panel_weights
from btc_vol_research.models.merton.params import MertonParams
from btc_vol_research.models.merton.pricer import merton_iv_panel


def _initial_guess(panel: pd.DataFrame, bounds: MertonBounds) -> MertonParams:
    atm_row = panel.loc[panel["log_moneyness"].abs().idxmin()]
    atm_iv = float(atm_row["iv_used"])
    return MertonParams(
        sigma=np.clip(atm_iv * 0.85, *bounds.sigma),
        lambda_jump=np.clip(1.0, *bounds.lambda_jump),
        mu_jump=np.clip(-0.1, *bounds.mu_jump),
        sigma_jump=np.clip(0.2, *bounds.sigma_jump),
    )


def calibrate_global(
    panel: pd.DataFrame,
    cfg: AppConfig,
    *,
    weight_fn: WeightFn | None = None,
    weight_scheme: str = "uniform",
) -> GlobalCalibrationResult:
    """
    Un seul jeu de paramètres Merton sur toutes les maturités/strikes.
    Objectif : somme des w_i (sigma_model - sigma_mkt)^2 (w_i = 1 si uniforme).
    """
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.merton_bounds

    fit_df = quality_filter(panel, min_strikes=calib.min_strikes_per_slice).reset_index(drop=True)
    market_iv = fit_df["iv_used"].values.astype(float)
    weights = None
    if weight_fn is not None:
        weights = build_panel_weights(
            fit_df,
            weight_fn,
            calib,
            market.risk_free_rate,
            market.dividend_yield,
        )

    x0 = _initial_guess(fit_df, bounds).as_array()
    bnds = [bounds.sigma, bounds.lambda_jump, bounds.mu_jump, bounds.sigma_jump]
    penalty = calib.feller_penalty

    def objective(x: np.ndarray) -> float:
        params = MertonParams.from_array(x)
        if not params.is_valid():
            return penalty
        try:
            model_iv = merton_iv_panel(
                fit_df,
                params,
                market.risk_free_rate,
                market.dividend_yield,
            )
        except Exception:
            return penalty
        if np.any(~np.isfinite(model_iv)):
            return penalty
        return iv_sse(market_iv, model_iv, weights=weights)

    t0 = time.perf_counter()
    res = minimize(objective, x0, method=calib.optimizer, bounds=bnds, options={"maxiter": 300})
    calibration_time_s = time.perf_counter() - t0

    params_opt = MertonParams.from_array(res.x)
    model_iv = merton_iv_panel(
        fit_df,
        params_opt,
        market.risk_free_rate,
        market.dividend_yield,
    )
    slice_results = build_slice_fits(fit_df, model_iv, weights=weights)

    rmse = iv_rmse(market_iv, model_iv)
    w_rmse = iv_weighted_rmse(market_iv, model_iv, weights) if weights is not None else rmse

    return GlobalCalibrationResult(
        params=params_opt,
        rmse_iv=rmse,
        weighted_rmse_iv=w_rmse,
        slice_results=slice_results,
        n_points=len(fit_df),
        success=bool(res.success),
        message=str(res.message),
        weight_scheme=weight_scheme,
        calibration_time_s=calibration_time_s,
    )
