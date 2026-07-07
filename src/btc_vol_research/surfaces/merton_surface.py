"""Surfaces de volatilité implicite pour le modèle de Merton global.

Deux visualisations distinctes, sur la même grille (k, T) :

- Surface IV modèle : Merton est global (défini partout), on l'affiche sur
  TOUTE la grille — surface de vol extrapolée, lisse, au-delà des strikes cotés.
- Surface d'erreur absolue : |IV marché - IV modèle|, uniquement là où une IV
  marché existe réellement (enveloppe convexe des points observés). Aucune
  extrapolation — pas d'erreur inventée là où rien ne se traite.

La grille couvre la même donnée `fit_df` que les smiles et la calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from btc_vol_research.models.merton import MertonParams, merton_iv_panel


def _build_grid(
    k: np.ndarray,
    t: np.ndarray,
    n_moneyness: int,
    n_maturities: int,
    k_pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grille rectangulaire (log-moneyness, T) sur la plage des données observées."""
    k_lin = np.linspace(float(k.min()) - k_pad, float(k.max()) + k_pad, n_moneyness)
    t_lin = np.linspace(float(t.min()), float(t.max()), n_maturities)
    return np.meshgrid(k_lin, t_lin)


def build_merton_surface_grid(
    fit_df: pd.DataFrame,
    params: MertonParams,
    r: float,
    q: float,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    k_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grille (log-moneyness, T) -> sigma_IV Merton, évaluée sur TOUTE la grille.

    Le modèle étant global, la surface est extrapolée de façon lisse au-delà
    des strikes cotés — c'est la surface de vol que l'on veut visualiser en entier.
    """
    if fit_df.empty:
        raise ValueError("Panel vide pour la surface Merton")

    k = fit_df["log_moneyness"].values.astype(float)
    t = fit_df["T"].values.astype(float)
    lm_grid, t_grid = _build_grid(k, t, n_moneyness, n_maturities, k_pad)

    s_ref = float(fit_df["S"].median())
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


def build_merton_abs_error_surface_grid(
    fit_df: pd.DataFrame,
    model_iv: np.ndarray,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    k_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grille |IV marché - IV modèle| (pts vol), restreinte aux données observées.

    Interpolation linéaire uniquement : hors de l'enveloppe convexe des points
    marché, la valeur reste NaN (aucune extrapolation nearest-neighbor). On ne
    veut pas d'erreur affichée là où il n'y a pas d'IV marché.
    """
    if fit_df.empty:
        raise ValueError("Panel vide pour la surface d'erreur Merton")

    market_iv = fit_df["iv_used"].values.astype(float)
    model_iv = np.asarray(model_iv, dtype=float)
    abs_err_pts = np.abs(model_iv - market_iv) * 100.0

    k = fit_df["log_moneyness"].values.astype(float)
    t = fit_df["T"].values.astype(float)
    lm_grid, t_grid = _build_grid(k, t, n_moneyness, n_maturities, k_pad)

    err_grid = griddata(np.column_stack([k, t]), abs_err_pts, (lm_grid, t_grid), method="linear")
    return lm_grid, t_grid, err_grid
