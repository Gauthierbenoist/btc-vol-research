"""Tableaux récapitulatifs de calibration (modèles globaux)."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.models.calibration.results import GlobalCalibrationResult, SliceFitResult
from btc_vol_research.models.merton.params import MertonParams


def slice_fit_summary_table(slice_results: list[SliceFitResult]) -> pd.DataFrame:
    rows = [
        {
            "slice_id": s.slice_id,
            "T_years": s.T,
            "rmse_iv": s.rmse_iv,
            "weighted_rmse_iv": s.weighted_rmse_iv,
            "n_strikes": len(s.market_iv),
        }
        for s in slice_results
    ]
    return pd.DataFrame(rows).sort_values("slice_id")


def merton_global_summary_table(result: GlobalCalibrationResult[MertonParams]) -> pd.DataFrame:
    p = result.params
    return pd.DataFrame(
        [
            {
                "weight_scheme": result.weight_scheme,
                "calibration_time_s": result.calibration_time_s,
                "rmse_iv": result.rmse_iv,
                "weighted_rmse_iv": result.weighted_rmse_iv,
                "n_points": result.n_points,
                "n_slices": len(result.slice_results),
                "sigma": p.sigma,
                "lambda_jump": p.lambda_jump,
                "mu_jump": p.mu_jump,
                "sigma_jump": p.sigma_jump,
                "success": result.success,
            }
        ]
    )
