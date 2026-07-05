"""Diagnostics IV : RMSE SVI par zone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.analysis.zones import ZONE_ORDER, assign_moneyness_zone
from btc_vol_research.models.svi.calibrate import SVICalibrationResult


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))


def svi_rmse_by_zone(
    results: list[SVICalibrationResult],
    atm_half_width: float = 0.10,
) -> pd.DataFrame:
    """RMSE marché vs SVI par zone et maturité."""
    rows = []
    for r in results:
        lm = r.log_moneyness
        zones = assign_moneyness_zone(lm, atm_half_width)
        mkt = r.market_iv
        mdl = r.model_iv
        for zone in ZONE_ORDER:
            mask = zones == zone
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "slice_id": r.slice_id,
                    "T_years": r.T,
                    "zone": zone,
                    "n": int(mask.sum()),
                    "rmse_svi": _rmse(mdl[mask], mkt[mask]),
                    "mae_svi": _mae(mdl[mask], mkt[mask]),
                    "bias_model_minus_mkt": float(np.nanmean(mdl[mask] - mkt[mask])),
                    "mean_mkt_iv_pct": float(np.nanmean(mkt[mask]) * 100),
                    "mean_model_iv_pct": float(np.nanmean(mdl[mask]) * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice_id", "zone"])
