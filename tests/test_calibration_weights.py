"""Pondérations de calibration."""

import numpy as np
import pandas as pd

from btc_vol_research.calibration.weights import (
    _vega_vector,
    calibration_weights,
    calibration_weights_v2,
)
from btc_vol_research.config import CalibrationConfig


def _slice_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "S": [100_000.0, 100_000.0],
            "T": [0.25, 0.25],
            "K": [90_000.0, 110_000.0],
            "iv_used": [0.5, 0.45],
            "open_interest": [0.0, 3.0],
            "volume_24h": [0.0, 15.0],
        }
    )


def test_vega_vector_matches_scalar_bs_vega():
    from btc_vol_research.market.greeks import bs_vega

    df = _slice_df()
    vec = _vega_vector(df, 0.0, 0.0)
    expected = np.array(
        [
            bs_vega(float(row.S), float(row.K), float(row.T), 0.0, 0.0, float(row.iv_used))
            for row in df.itertuples()
        ]
    )
    assert np.allclose(vec, expected)


def test_calibration_weights_v2_uses_one_plus_oi_and_volume():
    cfg = CalibrationConfig(use_vega_weight=False)
    w = calibration_weights_v2(_slice_df(), cfg, 0.0, 0.0)
    assert w[0] / w[1] == (1.0 * 1.0) / (np.sqrt(4.0) * np.sqrt(16.0))


def test_v1_and_v2_differ_with_vega_off():
    cfg = CalibrationConfig(use_vega_weight=False, use_liquidity_weight=True)
    w1 = calibration_weights(_slice_df(), cfg, 0.0, 0.0)
    w2 = calibration_weights_v2(_slice_df(), cfg, 0.0, 0.0)
    assert not np.allclose(w1, w2)
