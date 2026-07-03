"""Tableaux diagnostics Merton par maturité."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.models.calibration.results import GlobalCalibrationResult, SliceFitResult
from btc_vol_research.models.calibration.slice_metrics import slice_iv_diagnostics
from btc_vol_research.models.calibration_weights import _vega_vector
from btc_vol_research.models.merton.params import MertonParams


def _vega_weight_stats(
    slice_df: pd.DataFrame,
    weights: np.ndarray,
    r: float,
    q: float,
) -> dict[str, float]:
    vegas = _vega_vector(slice_df, r, q)
    w = np.asarray(weights, dtype=float)
    return {
        "vega_mean": float(np.mean(vegas)),
        "vega_min": float(np.min(vegas)),
        "vega_max": float(np.max(vegas)),
        "weight_mean": float(np.mean(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
    }


def merton_slice_metrics_row(
    sr: SliceFitResult,
    *,
    snapshot_date: str,
    weight_scheme: str,
    weight_scheme_label: str,
    atm_half_width: float,
    calibration_time_s: float,
    slice_df: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
    r: float = 0.0,
    q: float = 0.0,
) -> dict:
    diag = slice_iv_diagnostics(sr.log_moneyness, sr.market_iv, sr.model_iv, atm_half_width)
    row = {
        "snapshot_date": snapshot_date,
        "weight_scheme": weight_scheme,
        "weight_scheme_label": weight_scheme_label,
        "slice_id": sr.slice_id,
        "T_years": sr.T,
        "n_strikes": len(sr.market_iv),
        "calibration_time_s": calibration_time_s,
        "rmse_uniform": diag["rmse_uniform"] * 100.0,
        "mae_uniform": diag["mae_uniform"] * 100.0,
        "rmse_atm": diag["rmse_atm"] * 100.0,
        "rmse_left_wing": diag["rmse_left_wing"] * 100.0,
        "rmse_right_wing": diag["rmse_right_wing"] * 100.0,
        "max_error_iv": diag["max_error_iv"] * 100.0,
    }
    if weights is not None and slice_df is not None:
        row.update(_vega_weight_stats(slice_df, weights, r, q))
    else:
        row.update(
            {
                "vega_mean": float("nan"),
                "vega_min": float("nan"),
                "vega_max": float("nan"),
                "weight_mean": float("nan"),
                "weight_min": float("nan"),
                "weight_max": float("nan"),
            }
        )
    return row


def merton_slice_metrics_table(
    result: GlobalCalibrationResult[MertonParams],
    panel: pd.DataFrame,
    *,
    snapshot_date: str,
    weight_scheme_label: str,
    atm_half_width: float,
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
                atm_half_width=atm_half_width,
                calibration_time_s=result.calibration_time_s,
                slice_df=g,
                weights=w_slice,
                r=r,
                q=q,
            )
        )
    return pd.DataFrame(rows).sort_values(["weight_scheme", "slice_id"])
