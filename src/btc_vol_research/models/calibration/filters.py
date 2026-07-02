"""Filtres qualité communs aux calibrations."""

from __future__ import annotations

import pandas as pd

IV_MIN = 0.05
IV_MAX = 3.0
T_MIN_YEARS = 1 / 365
T_MAX_YEARS = 2.0


def quality_filter(
    df: pd.DataFrame,
    *,
    min_strikes: int = 1,
    iv_col: str = "iv_used",
) -> pd.DataFrame:
    """IV et maturité dans des plages raisonnables pour la calibration."""
    out = df.loc[
        df[iv_col].between(IV_MIN, IV_MAX) & df["T"].between(T_MIN_YEARS, T_MAX_YEARS)
    ].copy()
    if len(out) < min_strikes:
        raise ValueError(f"Pas assez de quotes après filtre qualité ({len(out)} < {min_strikes})")
    return out
