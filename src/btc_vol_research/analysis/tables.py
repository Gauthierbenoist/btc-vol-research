"""Tables récapitulatives de calibration (SVI, Heston, Merton)."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.calibration.results import (
    GlobalCalibrationResult,
    SliceCalibrationResult,
    SliceFitResult,
)


def heston_summary_table(results: list[SliceCalibrationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "slice_id": r.slice_id,
                "rmse_iv": r.rmse_iv,
                "v0": r.params.v0,
                "kappa": r.params.kappa,
                "theta": r.params.theta,
                "sigma": r.params.sigma,
                "rho": r.params.rho,
                "feller": r.params.feller_satisfied(),
                "success": r.success,
            }
        )
    return pd.DataFrame(rows).sort_values("slice_id")


def svi_summary_table(results: list[SliceCalibrationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.params
        rows.append(
            {
                "slice_id": r.slice_id,
                "rmse_iv": r.rmse_iv,
                "a": p.a,
                "b": p.b,
                "rho": p.rho,
                "m": p.m,
                "sigma": p.sigma,
                "butterfly_ok": p.butterfly_ok(),
                "success": r.success,
            }
        )
    return pd.DataFrame(rows).sort_values("slice_id")


def svi_term_structure_table(results: list[SliceCalibrationResult]) -> pd.DataFrame:
    """Paramètres SVI en fonction de la maturité (pour ρ(T), etc.)."""
    rows = []
    for r in results:
        p = r.params
        rows.append(
            {
                "slice_id": r.slice_id,
                "T_years": r.T,
                "T_days": r.T * 365.25,
                "rho": p.rho,
                "a": p.a,
                "b": p.b,
                "m": p.m,
                "sigma": p.sigma,
            }
        )
    return pd.DataFrame(rows).sort_values("T_years")


def slice_fit_summary_table(slice_results: list[SliceFitResult]) -> pd.DataFrame:
    rows = [
        {
            "slice_id": s.slice_id,
            "T_years": s.T,
            "rmse_iv": s.rmse_iv,
            "n_strikes": len(s.market_iv),
        }
        for s in slice_results
    ]
    return pd.DataFrame(rows).sort_values("slice_id")


def merton_global_summary_table(result: GlobalCalibrationResult) -> pd.DataFrame:
    p = result.params
    return pd.DataFrame(
        [
            {
                "weight_scheme": result.weight_scheme,
                "calibration_time_s": result.calibration_time_s,
                "rmse_iv": result.rmse_iv,
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
