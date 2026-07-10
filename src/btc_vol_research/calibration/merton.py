"""Calibration globale Merton sur toute la surface IV."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from btc_vol_research.calibration.errors import iv_rmse, sse_objective
from btc_vol_research.calibration.results import GlobalCalibrationResult
from btc_vol_research.calibration.slices import atm_row, build_slice_fits, require_min_points
from btc_vol_research.calibration.weights import WeightFn, build_panel_weights
from btc_vol_research.config import AppConfig, MertonBounds, MertonInitial
from btc_vol_research.models.merton import MertonParams, merton_iv_panel


def _clip_params(params: MertonParams, bounds: MertonBounds) -> MertonParams:
    return MertonParams(
        sigma=np.clip(params.sigma, *bounds.sigma),
        lambda_jump=np.clip(params.lambda_jump, *bounds.lambda_jump),
        mu_jump=np.clip(params.mu_jump, *bounds.mu_jump),
        sigma_jump=np.clip(params.sigma_jump, *bounds.sigma_jump),
    )


def _initial_guess(
    panel: pd.DataFrame,
    bounds: MertonBounds,
    initial: MertonInitial,
) -> MertonParams:
    atm_iv = float(atm_row(panel)["iv_used"])
    sigma = initial.sigma if initial.sigma is not None else atm_iv * 0.85
    return _clip_params(
        MertonParams(
            sigma=sigma,
            lambda_jump=initial.lambda_jump,
            mu_jump=initial.mu_jump,
            sigma_jump=initial.sigma_jump,
        ),
        bounds,
    )


def calibrate_global(
    fit_df: pd.DataFrame,
    cfg: AppConfig,
    *,
    weight_fn: WeightFn | None = None,
    weight_scheme: str = "uniform",
) -> GlobalCalibrationResult[MertonParams]:
    """
    Un seul jeu de paramètres Merton sur toutes les maturités/strikes.
    Objectif : somme des w_i (sigma_model - sigma_mkt)^2 (w_i = 1 si uniforme).

    `fit_df` est déjà filtré (build_market_panel()) — même donnée que celle
    utilisée pour les plots, aucun re-filtrage ici.
    """
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.merton_bounds
    merton_cfg = cfg.merton

    fit_df = require_min_points(fit_df, calib.min_strikes_per_slice)
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
        return sse_objective(market_iv, model_iv, weights=weights)

    x0 = _initial_guess(fit_df, bounds, merton_cfg.initial).as_array()

    t0 = time.perf_counter()
    de_res = differential_evolution(
        objective,
        bounds=bnds,
        x0=x0,
        maxiter=40,
        popsize=12,
        polish=False,
        seed=42,
        updating="deferred",
        workers=1,
    )
    res = minimize(
        objective,
        de_res.x,
        method=calib.optimizer,
        bounds=bnds,
        options={"maxiter": 300},
    )
    calibration_time_s = time.perf_counter() - t0

    params_opt = MertonParams.from_array(res.x)
    model_iv = merton_iv_panel(
        fit_df,
        params_opt,
        market.risk_free_rate,
        market.dividend_yield,
    )

    return GlobalCalibrationResult(
        params=params_opt,
        rmse_iv=iv_rmse(market_iv, model_iv),
        slice_results=build_slice_fits(fit_df, model_iv),
        n_points=len(fit_df),
        success=bool(de_res.success and res.success),
        message=f"DE: {de_res.message} | local: {res.message}",
        weight_scheme=weight_scheme,
        calibration_time_s=calibration_time_s,
    )
