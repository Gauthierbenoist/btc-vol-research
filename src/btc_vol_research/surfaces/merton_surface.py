"""Surface de volatilité implicite pour le modèle de Merton global.

Les deux surfaces (IV modèle et erreur absolue) sont restreintes à l'enveloppe
convexe des points marché réellement observés : on n'affiche rien là où aucune
option ne se traite à une maturité donnée (pas d'extrapolation). C'est la même
donnée `fit_df` que celle des smiles et de la calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from btc_vol_research.models.merton import MertonParams, merton_iv_panel


def _grid_and_envelope_mask(
    k: np.ndarray,
    t: np.ndarray,
    n_moneyness: int,
    n_maturities: int,
    k_pad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grille (k, T) + masque booléen de l'enveloppe convexe des points (k, t) observés.

    Le masque est True à l'intérieur de l'enveloppe des données réelles, False dehors.
    Construit via la même triangulation de Delaunay que griddata(method="linear"),
    donc identique à la région où l'interpolation de l'erreur est définie.
    """
    k_lin = np.linspace(float(k.min()) - k_pad, float(k.max()) + k_pad, n_moneyness)
    t_lin = np.linspace(float(t.min()), float(t.max()), n_maturities)
    lm_grid, t_grid = np.meshgrid(k_lin, t_lin)

    inside = griddata(
        np.column_stack([k, t]),
        np.ones(len(k)),
        (lm_grid, t_grid),
        method="linear",
    )
    mask = np.isfinite(inside)
    return lm_grid, t_grid, mask


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
    Grille (log-moneyness, T) -> sigma_IV Merton, masquée à l'enveloppe des données.

    Le modèle est global (défini partout), mais on ne montre l'IV que là où des
    options existent réellement — mêmes bornes que la surface d'erreur.
    """
    if fit_df.empty:
        raise ValueError("Panel vide pour la surface Merton")

    k = fit_df["log_moneyness"].values.astype(float)
    t = fit_df["T"].values.astype(float)
    lm_grid, t_grid, mask = _grid_and_envelope_mask(k, t, n_moneyness, n_maturities, k_pad)

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
    iv = np.where(mask, iv, np.nan)
    return lm_grid, t_grid, iv


def build_merton_abs_error_surface_grid(
    fit_df: pd.DataFrame,
    model_iv: np.ndarray,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    k_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grille |IV marché - IV modèle| (pts vol), restreinte à l'enveloppe des données.

    Interpolation linéaire uniquement : hors de l'enveloppe convexe des points
    observés, la valeur reste NaN (aucune extrapolation nearest-neighbor).
    """
    if fit_df.empty:
        raise ValueError("Panel vide pour la surface d'erreur Merton")

    market_iv = fit_df["iv_used"].values.astype(float)
    model_iv = np.asarray(model_iv, dtype=float)
    abs_err_pts = np.abs(model_iv - market_iv) * 100.0

    k = fit_df["log_moneyness"].values.astype(float)
    t = fit_df["T"].values.astype(float)
    lm_grid, t_grid, _ = _grid_and_envelope_mask(k, t, n_moneyness, n_maturities, k_pad)

    err_grid = griddata(np.column_stack([k, t]), abs_err_pts, (lm_grid, t_grid), method="linear")
    return lm_grid, t_grid, err_grid
