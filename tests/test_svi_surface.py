"""Tests surface SVI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.models.svi.calibrate import SVICalibrationResult  # noqa: E402
from btc_vol_research.models.svi.params import SVIParams  # noqa: E402
from btc_vol_research.surfaces.svi_surface import build_svi_surface_grid  # noqa: E402


def _fake_result(T: float, a: float) -> SVICalibrationResult:
    p = SVIParams(a=a, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
    k = np.linspace(-0.3, 0.3, 10)
    return SVICalibrationResult(
        slice_id=str(T),
        params=p,
        rmse_iv=0.01,
        weighted_rmse_iv=0.01,
        market_iv=np.full(10, 0.5),
        model_iv=np.full(10, 0.5),
        log_moneyness=k,
        T=T,
        success=True,
        message="ok",
    )


def test_build_svi_surface_grid():
    r1, r2 = _fake_result(0.1, 0.04), _fake_result(0.5, 0.08)
    lm, T, iv = build_svi_surface_grid([r1, r2], n_moneyness=20, n_maturities=10)
    assert lm.shape == iv.shape == T.shape
    assert np.all(np.isfinite(iv))


def test_grid_surface_to_long_dataframe():
    from btc_vol_research.surfaces.surface import grid_surface_to_long_dataframe
    from btc_vol_research.surfaces.svi_surface import surface_to_long_dataframe

    lm = np.array([[-1.0, 0.0], [1.0, np.nan]])
    t = np.array([[0.1, 0.1], [0.5, 0.5]])
    iv = np.array([[0.4, 0.5], [0.6, np.nan]])

    direct = grid_surface_to_long_dataframe(lm, t, iv, "2026-06-01", value_col="iv_svi")
    wrapped = surface_to_long_dataframe(lm, t, iv, "2026-06-01")
    assert len(direct) == 3
    assert direct.equals(wrapped)
    assert np.allclose(direct["iv_svi"].values, [0.4, 0.5, 0.6])
