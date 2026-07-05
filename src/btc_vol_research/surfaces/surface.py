"""Surface de volatilité implicite (grille moneyness × maturité)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def build_iv_surface_grid(
    panel: pd.DataFrame,
    *,
    n_moneyness: int = 40,
    n_maturities: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpole iv_used sur une grille (log_moneyness, T).

    Returns:
        lm_grid, T_grid, iv_matrix (NaN hors enveloppe convexe des points)
    """
    lm = panel["log_moneyness"].values
    T = panel["T"].values
    iv = panel["iv_used"].values

    lm_lin = np.linspace(lm.min(), lm.max(), n_moneyness)
    T_lin = np.linspace(T.min(), T.max(), n_maturities)
    lm_grid, T_grid = np.meshgrid(lm_lin, T_lin)
    points = np.column_stack([lm, T])
    iv_matrix = griddata(points, iv, (lm_grid, T_grid), method="linear")
    return lm_grid, T_grid, iv_matrix


def grid_surface_to_long_dataframe(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    value_matrix: np.ndarray,
    snapshot_date: str,
    *,
    value_col: str,
) -> pd.DataFrame:
    """Grille 2D (log-moneyness, T) -> DataFrame long (points finis uniquement)."""
    lm = np.asarray(lm_grid, dtype=float).ravel()
    t = np.asarray(t_grid, dtype=float).ravel()
    values = np.asarray(value_matrix, dtype=float).ravel()
    mask = np.isfinite(values)
    if not np.any(mask):
        return pd.DataFrame(columns=["snapshot_date", "log_moneyness", "T_years", value_col])
    return pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "log_moneyness": lm[mask],
            "T_years": t[mask],
            value_col: values[mask],
        }
    )
