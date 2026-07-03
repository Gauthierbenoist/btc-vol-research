"""Surface de volatilité implicite pour le modèle de Merton global."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.models.merton.params import MertonParams
from btc_vol_research.models.merton.pricer import merton_iv_panel


def build_merton_surface_grid(
    panel: pd.DataFrame,
    params: MertonParams,
    r: float,
    q: float,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    k_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une grille (log-moneyness, T) -> sigma_IV pour Merton.

    Le modèle étant global, on réévalue l'IV sur une grille abstraite (k, T)
    en reconstruisant des strikes à partir d'un spot de référence du snapshot.
    """
    if panel.empty:
        raise ValueError("Panel vide pour la surface Merton")

    k_lo = float(panel["log_moneyness"].min()) - k_pad
    k_hi = float(panel["log_moneyness"].max()) + k_pad
    t_lo = float(panel["T"].min())
    t_hi = float(panel["T"].max())

    k_lin = np.linspace(k_lo, k_hi, n_moneyness)
    t_lin = np.linspace(t_lo, t_hi, n_maturities)
    lm_grid, t_grid = np.meshgrid(k_lin, t_lin)

    s_ref = float(panel["S"].median())
    flat = pd.DataFrame(
        {
            "S": np.full(lm_grid.size, s_ref),
            "T": t_grid.ravel(),
            "log_moneyness": lm_grid.ravel(),
        }
    )
    flat["K"] = flat["S"] * np.exp((r - q) * flat["T"] + flat["log_moneyness"])
    # Parite put-call restauree : le choix call/put ne change plus l'IV.
    flat["option_type"] = np.where(flat["log_moneyness"] >= 0.0, "call", "put")

    iv = merton_iv_panel(flat, params, r, q).reshape(lm_grid.shape)
    return lm_grid, t_grid, iv


def surface_to_long_dataframe(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_matrix: np.ndarray,
    snapshot_date: str,
) -> pd.DataFrame:
    rows = []
    for i in range(lm_grid.shape[0]):
        for j in range(lm_grid.shape[1]):
            iv = iv_matrix[i, j]
            if not np.isfinite(iv):
                continue
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "log_moneyness": float(lm_grid[i, j]),
                    "T_years": float(t_grid[i, j]),
                    "iv_merton": float(iv),
                }
            )
    return pd.DataFrame(rows)


def build_merton_abs_error_surface_grid(
    fit_df: pd.DataFrame,
    model_iv: np.ndarray,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    k_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grille |IV marche - IV modele| (pts vol) interpolee sur (k, T)."""
    from scipy.interpolate import griddata

    if fit_df.empty:
        raise ValueError("Panel vide pour la surface d'erreur Merton")

    market_iv = fit_df["iv_used"].values.astype(float)
    model_iv = np.asarray(model_iv, dtype=float)
    abs_err_pts = np.abs(model_iv - market_iv) * 100.0

    k = fit_df["log_moneyness"].values.astype(float)
    t = fit_df["T"].values.astype(float)
    k_lo = float(k.min()) - k_pad
    k_hi = float(k.max()) + k_pad
    t_lo = float(t.min())
    t_hi = float(t.max())

    k_lin = np.linspace(k_lo, k_hi, n_moneyness)
    t_lin = np.linspace(t_lo, t_hi, n_maturities)
    lm_grid, t_grid = np.meshgrid(k_lin, t_lin)

    points = np.column_stack([k, t])
    err_grid = griddata(points, abs_err_pts, (lm_grid, t_grid), method="linear")
    if np.any(~np.isfinite(err_grid)):
        nearest = griddata(points, abs_err_pts, (lm_grid, t_grid), method="nearest")
        err_grid = np.where(np.isfinite(err_grid), err_grid, nearest)

    return lm_grid, t_grid, err_grid


def abs_error_surface_to_long_dataframe(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    err_matrix: np.ndarray,
    snapshot_date: str,
) -> pd.DataFrame:
    rows = []
    for i in range(lm_grid.shape[0]):
        for j in range(lm_grid.shape[1]):
            err = err_matrix[i, j]
            if not np.isfinite(err):
                continue
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "log_moneyness": float(lm_grid[i, j]),
                    "T_years": float(t_grid[i, j]),
                    "abs_error_iv_pts": float(err),
                }
            )
    return pd.DataFrame(rows)
