"""Black-Scholes-Merton (européen) — prix vectorisés et scalaires."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """d1 vectorisé (T et sigma bornés à 1e-10 pour la stabilité numérique)."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call_price_vec(
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
    vol_sqrt_t = sigma * np.sqrt(T)
    d_1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d_2 = d_1 - vol_sqrt_t
    return S * np.exp(-q * T) * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def bs_put_price_vec(
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
    vol_sqrt_t = sigma * np.sqrt(T)
    d_1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d_2 = d_1 - vol_sqrt_t
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * np.exp(-q * T) * norm.cdf(-d_1)


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return float(bs_call_price_vec(S, K, T, r, q, sigma))


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    return float(bs_put_price_vec(S, K, T, r, q, sigma))
