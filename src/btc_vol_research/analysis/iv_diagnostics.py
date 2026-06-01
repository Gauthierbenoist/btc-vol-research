"""Diagnostics IV : mark vs mid, RMSE SVI par zone."""

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


def mark_vs_mid_table(panel: pd.DataFrame, atm_half_width: float = 0.10) -> pd.DataFrame:
    """Stats mark_iv vs iv_mid par maturité et zone."""
    rows = []
    for sid, g in panel.groupby("slice_id"):
        lm = g["log_moneyness"].values
        zones = assign_moneyness_zone(lm, atm_half_width)
        mark = g["iv_mark"].values
        mid = g["iv_mid"].values
        used_mid = g["iv_used"].values == g["iv_mid"].values

        for zone in ZONE_ORDER:
            mask = zones == zone
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "slice_id": sid,
                    "T_years": float(g["T"].iloc[0]),
                    "zone": zone,
                    "n": int(mask.sum()),
                    "rmse_mark_mid": _rmse(mark[mask], mid[mask]),
                    "mae_mark_mid": _mae(mark[mask], mid[mask]),
                    "mean_mark_minus_mid": float(np.nanmean(mark[mask] - mid[mask])),
                    "mean_mark_pct": float(np.nanmean(mark[mask]) * 100),
                    "mean_mid_pct": float(np.nanmean(mid[mask]) * 100),
                    "pct_points_using_mid": float(used_mid[mask].mean() * 100),
                    "n_mid_valid": int(np.isfinite(mid[mask]).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice_id", "zone"])


def mark_vs_mid_summary(panel: pd.DataFrame, atm_half_width: float = 0.10) -> pd.DataFrame:
    """Agrégat toutes maturités par zone."""
    detail = mark_vs_mid_table(panel, atm_half_width)
    rows = []
    for zone in ZONE_ORDER:
        z = detail[detail["zone"] == zone]
        if z.empty:
            continue
        rows.append(
            {
                "zone": zone,
                "n_slices": z["slice_id"].nunique(),
                "n_points": int(z["n"].sum()),
                "rmse_mark_mid_avg": float(z["rmse_mark_mid"].mean()),
                "mae_mark_mid_avg": float(z["mae_mark_mid"].mean()),
                "mean_mark_minus_mid_avg": float(z["mean_mark_minus_mid"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
