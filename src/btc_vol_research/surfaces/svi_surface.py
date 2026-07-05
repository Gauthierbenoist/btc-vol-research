"""Surface de volatilité implicite à partir des calibrations SVI par maturité."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btc_vol_research.calibration.results import SliceCalibrationResult
from btc_vol_research.models.svi import svi_iv_from_log_moneyness, svi_total_variance


def _sorted_results(results: list[SliceCalibrationResult]) -> list[SliceCalibrationResult]:
    return sorted(results, key=lambda r: r.T)


def _k_bounds(
    results: list[SliceCalibrationResult],
    panel: pd.DataFrame | None,
    k_pad: float,
) -> tuple[float, float]:
    if panel is not None and len(panel):
        lo = float(panel["log_moneyness"].min()) - k_pad
        hi = float(panel["log_moneyness"].max()) + k_pad
    else:
        lo = min(float(r.log_moneyness.min()) for r in results) - k_pad
        hi = max(float(r.log_moneyness.max()) for r in results) + k_pad
    return lo, hi


def svi_iv_at_point(
    k: np.ndarray | float,
    T: float,
    results: list[SliceCalibrationResult],
    *,
    atol_T: float = 1e-8,
) -> np.ndarray:
    """
    IV SVI en un point (ou vecteur k) et maturité T.

    Entre deux tenors calibrés : interpolation linéaire de la variance totale w(k)
    (plus stable pour la surface que l'interpolation des paramètres bruts).
    """
    k = np.asarray(k, dtype=float)
    ordered = _sorted_results(results)
    Ts = np.array([r.T for r in ordered])

    if T <= Ts[0] + atol_T:
        return svi_iv_from_log_moneyness(k, T, ordered[0].params)
    if T >= Ts[-1] - atol_T:
        return svi_iv_from_log_moneyness(k, T, ordered[-1].params)

    for r in ordered:
        if abs(r.T - T) <= atol_T:
            return svi_iv_from_log_moneyness(k, T, r.params)

    idx = int(np.searchsorted(Ts, T))
    r0, r1 = ordered[idx - 1], ordered[idx]
    alpha = (T - r0.T) / (r1.T - r0.T)
    w0 = svi_total_variance(k, r0.params)
    w1 = svi_total_variance(k, r1.params)
    w = (1.0 - alpha) * w0 + alpha * w1
    w = np.maximum(w, 1e-12)
    return np.sqrt(w / max(T, 1e-10))


def build_svi_surface_grid(
    results: list[SliceCalibrationResult],
    panel: pd.DataFrame | None = None,
    *,
    n_moneyness: int = 50,
    n_maturities: int | None = 30,
    k_pad: float = 0.05,
    interpolate_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une grille (log-moneyness, T) → σ_IV.

    Returns:
        lm_grid, T_grid, iv_matrix  (shapes compatibles meshgrid : T × k)
    """
    if len(results) < 2:
        raise ValueError("Au moins 2 tranches SVI calibrées pour une surface")

    ordered = _sorted_results(results)
    k_lo, k_hi = _k_bounds(ordered, panel, k_pad)
    k_lin = np.linspace(k_lo, k_hi, n_moneyness)
    T_vals = np.array([r.T for r in ordered])

    if n_maturities is None or n_maturities <= len(ordered):
        T_lin = T_vals
    else:
        T_lin = np.linspace(T_vals.min(), T_vals.max(), n_maturities)

    lm_grid, T_grid = np.meshgrid(k_lin, T_lin)
    iv_matrix = np.zeros_like(lm_grid)

    for i, T in enumerate(T_lin):
        if not interpolate_time and len(T_lin) == len(ordered):
            iv_matrix[i, :] = svi_iv_from_log_moneyness(k_lin, float(T), ordered[i].params)
        else:
            iv_matrix[i, :] = svi_iv_at_point(k_lin, float(T), ordered)

    return lm_grid, T_grid, iv_matrix
