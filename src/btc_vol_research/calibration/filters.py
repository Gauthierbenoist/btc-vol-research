"""Filtres qualité communs aux calibrations."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.config import AppConfig, MarketConfig


def quality_filter(
    df: pd.DataFrame,
    cfg: AppConfig | MarketConfig,
    *,
    min_strikes: int = 1,
    iv_col: str = "iv_used",
) -> pd.DataFrame:
    """IV et maturité dans des plages raisonnables pour la calibration.

    Les bornes (min_iv, max_iv, min_time_to_expiry_days, max_time_to_expiry_years)
    viennent de la config — source de vérité unique, partagée avec build_market_panel().

    Args:
        df: Panel avec colonnes iv_used et T.
        cfg: Configuration (AppConfig ou MarketConfig) fournissant les bornes.
        min_strikes: Nombre minimum de points après filtrage.
        iv_col: Colonne IV à filtrer (défaut: iv_used).

    Returns:
        DataFrame filtré.

    Raises:
        ValueError: Si trop peu de points après filtrage.
    """
    market = cfg.market if isinstance(cfg, AppConfig) else cfg
    t_min_years = market.min_time_to_expiry_days / 365.25
    mask = df[iv_col].between(market.min_iv, market.max_iv) & df["T"].between(
        t_min_years, market.max_time_to_expiry_years
    )
    out = df.loc[mask].copy()
    if len(out) < min_strikes:
        raise ValueError(f"Pas assez de quotes après filtre qualité ({len(out)} < {min_strikes})")
    return out
