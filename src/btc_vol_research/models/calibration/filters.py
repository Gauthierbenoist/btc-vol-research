"""Filtres qualité communs aux calibrations."""

from __future__ import annotations

import pandas as pd

IV_MIN = 0.05
IV_MAX = 3.0
T_MIN_YEARS = 1 / 365


def quality_filter(
    df: pd.DataFrame,
    *,
    min_strikes: int = 1,
    iv_col: str = "iv_used",
    t_max_years: float | None = None,
) -> pd.DataFrame:
    """IV et maturité dans des plages raisonnables pour la calibration.

    Args:
        df: Panel avec colonnes iv_used et T.
        min_strikes: Nombre minimum de points après filtrage.
        iv_col: Colonne IV à filtrer (défaut: iv_used).
        t_max_years: Limite haute de maturité (années). Si None, aucune limite.

    Returns:
        DataFrame filtré.

    Raises:
        ValueError: Si trop peu de points après filtrage.
    """
    mask = df[iv_col].between(IV_MIN, IV_MAX) & df["T"].between(T_MIN_YEARS, df["T"].max())
    if t_max_years is not None:
        mask &= df["T"] <= t_max_years
    out = df.loc[mask].copy()
    if len(out) < min_strikes:
        raise ValueError(f"Pas assez de quotes après filtre qualité ({len(out)} < {min_strikes})")
    return out
