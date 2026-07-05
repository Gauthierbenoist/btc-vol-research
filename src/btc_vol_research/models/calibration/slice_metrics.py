"""Métriques IV par tranche de maturité (uniformes et par zone)."""

from __future__ import annotations

import numpy as np

from btc_vol_research.analysis.zones import ZONE_ATM, ZONE_LEFT, ZONE_RIGHT, assign_moneyness_zone
from btc_vol_research.models.calibration.errors import iv_rmse


def _zone_rmse(
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    zones: np.ndarray,
    zone: str,
) -> float:
    mask = zones == zone
    if mask.sum() == 0:
        return float("nan")
    return iv_rmse(market_iv[mask], model_iv[mask])


def slice_iv_diagnostics(
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    atm_half_width: float,
) -> dict[str, float]:
    """RMSE uniforme, RMSE par zone, erreur maximale (IV décimale)."""
    lm = np.asarray(log_moneyness, dtype=float)
    mkt = np.asarray(market_iv, dtype=float)
    mdl = np.asarray(model_iv, dtype=float)
    zones = assign_moneyness_zone(lm, atm_half_width)
    abs_err = np.abs(mdl - mkt)

    return {
        "rmse_uniform": iv_rmse(mkt, mdl),
        "rmse_atm": _zone_rmse(mkt, mdl, zones, ZONE_ATM),
        "rmse_left_wing": _zone_rmse(mkt, mdl, zones, ZONE_LEFT),
        "rmse_right_wing": _zone_rmse(mkt, mdl, zones, ZONE_RIGHT),
        "max_error_iv": float(np.max(abs_err)),
    }
