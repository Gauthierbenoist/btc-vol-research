"""Découpage des fits globaux par maturité."""

from __future__ import annotations

import pandas as pd

from btc_vol_research.models.calibration.errors import iv_rmse
from btc_vol_research.models.calibration.results import SliceFitResult


def build_slice_fits(
    panel: pd.DataFrame,
    model_iv: pd.Series | "np.ndarray",
    *,
    weights: "np.ndarray | None" = None,
) -> list[SliceFitResult]:
    """RMSE par tranche à partir d'un fit global sur tout le panel."""
    import numpy as np

    model = np.asarray(model_iv, dtype=float)
    market = panel["iv_used"].values.astype(float)
    w_all = None if weights is None else np.asarray(weights, dtype=float)

    results: list[SliceFitResult] = []
    for sid, g in panel.groupby("slice_id", sort=True):
        idx = g.index
        pos = panel.index.get_indexer(idx)
        mkt = market[pos]
        mdl = model[pos]
        w = None if w_all is None else w_all[pos]
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
