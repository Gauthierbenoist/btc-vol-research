"""Formules SVI (variance totale et vol implicite)."""

from __future__ import annotations

import numpy as np

from btc_vol_research.models.svi.params import SVIParams


def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """w(k) en variance totale (σ²T)."""
    km = np.asarray(k, dtype=float) - params.m
    return params.a + params.b * (params.rho * km + np.sqrt(km**2 + params.sigma**2))


def svi_iv_from_log_moneyness(
    k: np.ndarray,
    T: float,
    params: SVIParams,
) -> np.ndarray:
    """σ_BS(k, T) à partir de w(k) = σ²T."""
    w = svi_total_variance(k, params)
    w = np.maximum(w, 1e-12)
    T = max(T, 1e-10)
    return np.sqrt(w / T)
