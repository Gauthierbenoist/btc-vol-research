"""Tableaux diagnostics Merton par maturité."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.calibration.errors import iv_rmse
from btc_vol_research.calibration.results import GlobalCalibrationResult, SliceFitResult
from btc_vol_research.calibration.weights import _vega_vector, _volume_series
from btc_vol_research.models.merton import MertonParams

_NAN_WEIGHT_COLS = {
    "vega_mean": float("nan"),
    "vega_min": float("nan"),
    "vega_max": float("nan"),
    "volume_mean": float("nan"),
    "volume_min": float("nan"),
    "volume_max": float("nan"),
    "weight_mean": float("nan"),
    "weight_min": float("nan"),
    "weight_max": float("nan"),
}


def _weight_stats(
    slice_df: pd.DataFrame,
    weights: np.ndarray,
    weight_scheme: str,
    r: float,
    q: float,
) -> dict[str, float]:
    w = np.asarray(weights, dtype=float)
    stats = {
        "weight_mean": float(np.mean(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "vega_mean": float("nan"),
        "vega_min": float("nan"),
        "vega_max": float("nan"),
        "volume_mean": float("nan"),
        "volume_min": float("nan"),
        "volume_max": float("nan"),
    }
    if weight_scheme == "vega":
        vegas = _vega_vector(slice_df, r, q)
        stats["vega_mean"] = float(np.mean(vegas))
        stats["vega_min"] = float(np.min(vegas))
        stats["vega_max"] = float(np.max(vegas))
    elif weight_scheme == "volume":
        vols = _volume_series(slice_df)
        stats["volume_mean"] = float(np.mean(vols))
        stats["volume_min"] = float(np.min(vols))
        stats["volume_max"] = float(np.max(vols))
    return stats


def merton_slice_metrics_row(
    sr: SliceFitResult,
    *,
    snapshot_date: str,
    weight_scheme: str,
    weight_scheme_label: str,
    calibration_time_s: float,
    slice_df: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
    r: float = 0.0,
    q: float = 0.0,
) -> dict:
    mkt = np.asarray(sr.market_iv, dtype=float)
    mdl = np.asarray(sr.model_iv, dtype=float)
    row = {
        "snapshot_date": snapshot_date,
        "weight_scheme": weight_scheme,
        "weight_scheme_label": weight_scheme_label,
        "slice_id": sr.slice_id,
        "T_years": sr.T,
        "n_strikes": len(sr.market_iv),
        "calibration_time_s": calibration_time_s,
        "rmse_uniform": iv_rmse(mkt, mdl) * 100.0,
        "max_error_iv": float(np.max(np.abs(mdl - mkt))) * 100.0,
    }
    if weights is not None and slice_df is not None:
        row.update(_weight_stats(slice_df, weights, weight_scheme, r, q))
    else:
        row.update(_NAN_WEIGHT_COLS)
    return row


def merton_slice_metrics_table(
    result: GlobalCalibrationResult[MertonParams],
    panel: pd.DataFrame,
    *,
    snapshot_date: str,
    weight_scheme_label: str,
    weights: np.ndarray | None = None,
    r: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for sr in result.slice_results:
        g = panel.loc[panel["slice_id"] == sr.slice_id]
        w_slice = None
        if weights is not None:
            pos = panel.index.get_indexer(g.index)
            w_slice = weights[pos]
        rows.append(
            merton_slice_metrics_row(
                sr,
                snapshot_date=snapshot_date,
                weight_scheme=result.weight_scheme,
                weight_scheme_label=weight_scheme_label,
                calibration_time_s=result.calibration_time_s,
                slice_df=g,
                weights=w_slice,
                r=r,
                q=q,
            )
        )
    return pd.DataFrame(rows).sort_values(["weight_scheme", "slice_id"])
