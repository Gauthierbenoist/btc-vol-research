"""Export des grilles de surface en format long (CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grid_to_long_dataframe(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    values: np.ndarray,
    snapshot_date: str,
    value_col: str,
) -> pd.DataFrame:
    """Aplatit une grille (k, T) → valeur en DataFrame long ; les non-finis sont exclus."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "log_moneyness": np.asarray(lm_grid, dtype=float)[mask],
            "T_years": np.asarray(t_grid, dtype=float)[mask],
            value_col: values[mask],
        }
    )
