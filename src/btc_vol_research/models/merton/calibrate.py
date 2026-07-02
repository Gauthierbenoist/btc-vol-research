"""Calibration globale Merton sur toute la surface IV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from btc_vol_research.config import AppConfig, MertonBounds
from btc_vol_research.models.calibration.errors import iv_rmse, iv_sse, iv_weighted_rmse
from btc_vol_research.models.calibration.filters import quality_filter
from btc_vol_research.models.calibration.results import GlobalCalibrationResult
from btc_vol_research.models.calibration.slice_fits import build_slice_fits
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
) -> GlobalCalibrationResult:
    """
    Un seul jeu de paramètres Merton sur toutes les maturités/strikes.
    Objectif : somme des (sigma_model - sigma_mkt)^2 sans pondération.
    """
    calib = cfg.calibration
    market = cfg.market
    bounds = cfg.merton_bounds

    fit_df = quality_filter(panel, min_strikes=calib.min_strikes_per_slice).reset_index(drop=True)
    market_iv = fit_df["iv_used"].values.astype(float)

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
        return iv_sse(market_iv, model_iv, weights=None)

    res = minimize(objective, x0, method=calib.optimizer, bounds=bnds, options={"maxiter": 300})

    params_opt = MertonParams.from_array(res.x)
    model_iv = merton_iv_panel(
        fit_df,
        params_opt,
        market.risk_free_rate,
        market.dividend_yield,
    )
    slice_results = build_slice_fits(fit_df, model_iv, weights=None)

    return GlobalCalibrationResult(
        params=params_opt,
        rmse_iv=iv_rmse(market_iv, model_iv),
        weighted_rmse_iv=iv_rmse(market_iv, model_iv),
        slice_results=slice_results,
        n_points=len(fit_df),
        success=bool(res.success),
        message=str(res.message),
    )
