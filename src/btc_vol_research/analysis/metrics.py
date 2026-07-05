"""Métriques de qualité d'ajustement."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.models.heston.calibrate import CalibrationResult


def calibration_summary_table(results: list[CalibrationResult]) -> pd.DataFrame:
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
