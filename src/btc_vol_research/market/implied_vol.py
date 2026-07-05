"""Inversion de volatilité implicite (Newton-Raphson vectorisé)."""

from __future__ import annotations

import numpy as np

from btc_vol_research.market.greeks import bs_vega_vec
from btc_vol_research.models.black_scholes import bs_call_price_vec, bs_put_price_vec


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
