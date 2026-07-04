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
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


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
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


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
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


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
    """Inversion Newton-Raphson vectorisee (NaN si echec)."""
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    opt = np.asarray(option_type)
    n = len(price)
    iv = np.full(n, np.nan)
    is_call = np.char.lower(opt.astype(str)) == "call"

    valid = np.isfinite(price) & (price > 0) & (T > 0) & (S > 0) & (K > 0)
    intrinsic = np.where(
        is_call,
        np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0),
        np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0),
    )
    valid &= price >= intrinsic * 0.999

    sigma = np.full(n, 0.8)
    active = valid.copy()

    for _ in range(max_iter):
        if not np.any(active):
            break
        model = np.where(
            is_call,
            bs_call_price_vec(S, K, T, r, q, sigma),
            bs_put_price_vec(S, K, T, r, q, sigma),
        )
        diff = model - price
        converged = active & (np.abs(diff) < tol)
        iv[converged] = sigma[converged]
        active &= ~converged
        if not np.any(active):
            break
        vega = bs_vega_vec(S, K, T, r, q, sigma)
        sigma = np.where(
            active,
            np.clip(sigma - diff / np.maximum(vega, 1e-12), 1e-4, 5.0),
            sigma,
        )

    loose = valid & np.isnan(iv)
    if np.any(loose):
        model = np.where(
            is_call,
            bs_call_price_vec(S, K, T, r, q, sigma),
            bs_put_price_vec(S, K, T, r, q, sigma),
        )
        diff = model - price
        iv[loose & (np.abs(diff) < tol * 10)] = sigma[loose & (np.abs(diff) < tol * 10)]

    return iv
