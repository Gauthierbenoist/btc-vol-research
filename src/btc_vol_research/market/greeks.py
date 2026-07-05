"""Greeks Black-Scholes (vega scalaire et vectorisé)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from btc_vol_research.models.black_scholes import d1


def bs_vega_vec(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    T = np.maximum(np.asarray(T, dtype=float), 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    d_1 = d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d_1) * np.sqrt(T)


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    return float(bs_vega_vec(S, K, T, r, q, sigma))
