"""Tests métriques et pondération Merton."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.config import AppConfig, CalibrationConfig, MertonBounds, MertonConfig  # noqa: E402
from btc_vol_research.models.calibration_weights import vega_only_weights  # noqa: E402
from btc_vol_research.models.merton.calibrate import calibrate_global  # noqa: E402
from btc_vol_research.models.merton.weight_schemes import MERTON_WEIGHT_SCHEMES  # noqa: E402


def _toy_panel() -> pd.DataFrame:
    rows = []
    for T, sid in [(0.08, "2026-07-01"), (0.25, "2026-10-01")]:
        for lm in np.linspace(-0.2, 0.2, 6):
            rows.append(
                {
                    "slice_id": sid,
                    "T": T,
                    "S": 100_000.0,
                    "K": 100_000.0 * np.exp(lm),
                    "log_moneyness": lm,
                    "iv_used": 0.45 + 0.1 * lm**2,
                    "option_type": "call" if lm >= 0 else "put",
                    "open_interest": 1.0 + abs(lm),
                }
            )
    return pd.DataFrame(rows)


def test_vega_only_weights_positive():
    df = _toy_panel().iloc[:6]
    w = vega_only_weights(df, CalibrationConfig(), 0.0, 0.0)
    assert len(w) == len(df)
    assert np.all(w > 0)
    assert abs(w.sum() - len(df)) < 1e-6


def test_merton_weight_schemes_registry():
    from btc_vol_research.models.merton.weight_schemes import get_merton_weight_scheme

    for sid in ("uniform", "vega", "volume"):
        assert get_merton_weight_scheme(sid).scheme_id == sid


def test_volume_only_weights_positive():
    df = _toy_panel().iloc[:6]
    from btc_vol_research.models.calibration_weights import volume_only_weights

    w = volume_only_weights(df, CalibrationConfig(), 0.0, 0.0)
    assert len(w) == len(df)
    assert np.all(w > 0)
    assert abs(w.sum() - len(df)) < 1e-6


def test_merton_global_calibration_vega_weighted():
    panel = _toy_panel()
    cfg = AppConfig(merton=MertonConfig(bounds=MertonBounds()))
    vega_scheme = next(s for s in MERTON_WEIGHT_SCHEMES if s.scheme_id == "vega")
    result = calibrate_global(panel, cfg, weight_fn=vega_scheme.weight_fn, weight_scheme="vega")
    assert result.weight_scheme == "vega"
    assert result.calibration_time_s >= 0
    assert result.n_points == len(panel)
    assert len(result.slice_results) == 2
    assert np.isfinite(result.rmse_iv)
