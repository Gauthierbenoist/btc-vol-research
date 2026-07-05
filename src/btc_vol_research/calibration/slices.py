"""Runner générique par maturité et découpage des fits globaux."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from btc_vol_research.calibration.errors import iv_rmse
from btc_vol_research.calibration.results import SliceFitResult
from btc_vol_research.config import AppConfig


def atm_row(df: pd.DataFrame) -> pd.Series:
    """Ligne la plus proche de la monnaie (|log_moneyness| minimal)."""
    return df.loc[df["log_moneyness"].abs().idxmin()]


def calibrate_all_slices(
    panel: pd.DataFrame,
    cfg: AppConfig,
    calibrate_slice_fn: Callable,
    *,
    weight_fn=None,
) -> list:
    """Calibre chaque maturité éligible ; les tranches qui échouent sont ignorées."""
    results = []
    for sid, g in panel.groupby("slice_id"):
        if len(g) < cfg.calibration.min_strikes_per_slice:
            continue
        try:
            results.append(calibrate_slice_fn(g, cfg, str(sid), weight_fn=weight_fn))
        except Exception:
            continue
    return results


def build_slice_fits(panel: pd.DataFrame, model_iv: np.ndarray) -> list[SliceFitResult]:
    """RMSE par tranche à partir d'un fit global sur tout le panel."""
    model = np.asarray(model_iv, dtype=float)
    market = panel["iv_used"].values.astype(float)

    results: list[SliceFitResult] = []
    for sid, g in panel.groupby("slice_id", sort=True):
        pos = panel.index.get_indexer(g.index)
        mkt = market[pos]
        mdl = model[pos]
        results.append(
            SliceFitResult(
                slice_id=str(sid),
                T=float(g["T"].iloc[0]),
                rmse_iv=iv_rmse(mkt, mdl),
                market_iv=mkt,
                model_iv=mdl,
                log_moneyness=g["log_moneyness"].values.astype(float),
            )
        )
    return results
