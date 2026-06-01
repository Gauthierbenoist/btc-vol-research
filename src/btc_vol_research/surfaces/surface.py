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
