"""Black-Scholes-Merton (européen) — prix et inversion de volatilité."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray) -> np.ndarray:
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def forward_price(S: np.ndarray, T: np.ndarray, r: float, q: float) -> np.ndarray:
    return S * np.exp((r - q) * T)


def _scalar_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    T = max(T, 1e-10)
    sigma = max(sigma, 1e-10)
    return float((np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = _scalar_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = _scalar_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _scalar_d1(S, K, T, r, q, sigma)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def implied_volatility(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    option_type: np.ndarray,
    *,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> np.ndarray:
    """Inversion Newton-Raphson vectorisée (NaN si échec)."""
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    opt = np.asarray(option_type)
    n = len(price)
    iv = np.full(n, np.nan)

    for i in range(n):
        p, s, k, t = price[i], S[i], K[i], T[i]
        if not np.isfinite(p) or p <= 0 or t <= 0 or s <= 0 or k <= 0:
            continue
        is_call = str(opt[i]).lower() == "call"
        intrinsic = max(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0) if is_call else max(
            k * np.exp(-r * t) - s * np.exp(-q * t), 0.0
        )
        if p < intrinsic * 0.999:
            continue
        sigma = 0.8
        for _ in range(max_iter):
            if is_call:
                model = bs_call_price(s, k, t, r, q, sigma)
            else:
                model = bs_put_price(s, k, t, r, q, sigma)
            diff = model - p
            if abs(diff) < tol:
                iv[i] = sigma
                break
            v = bs_vega(s, k, t, r, q, sigma)
            if v < 1e-12:
                break
            sigma = max(min(sigma - diff / v, 5.0), 1e-4)
        else:
            if abs(diff) < tol * 10:
                iv[i] = sigma
    return iv
