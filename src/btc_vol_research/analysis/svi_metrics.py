"""Métriques calibration SVI."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.models.svi.calibrate import SVICalibrationResult


def svi_summary_table(results: list[SVICalibrationResult]) -> pd.DataFrame:
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


def svi_term_structure_table(results: list[SVICalibrationResult]) -> pd.DataFrame:
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
