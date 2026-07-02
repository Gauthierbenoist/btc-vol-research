"""Prix européens Merton jump-diffusion (série de Poisson × Black–Scholes)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from btc_vol_research.iv.black_scholes import implied_volatility
from btc_vol_research.models.merton.params import MertonParams

_MAX_POISSON_TERMS = 60


def _bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def merton_option_price(
    S0: float,
    K: float,
    T: float,
    params: MertonParams,
    r: float = 0.0,
    q: float = 0.0,
    *,
    option_type: str = "call",
) -> float:
    """Prix via expansion de Merton (1976)."""
    if T <= 0:
        is_call = str(option_type).lower() == "call"
        return _bs_call(S0, K, T, r, q, params.sigma) if is_call else _bs_put(S0, K, T, r, q, params.sigma)

    k = params.jump_compensation()
    lam_t = params.lambda_jump * T
    jump_drift = params.mu_jump + 0.5 * params.sigma_jump**2

    price = 0.0
    is_call = str(option_type).lower() == "call"
    pricer = _bs_call if is_call else _bs_put

    for n in range(_MAX_POISSON_TERMS):
        pois_prob = np.exp(-lam_t) * (lam_t**n) / math.factorial(n)
        if pois_prob < 1e-14 and n > 0:
            break
        sigma_n = np.sqrt(params.sigma**2 + n * params.sigma_jump**2 / T)
        r_n = r - q - params.lambda_jump * k + n * jump_drift / T
        price += pois_prob * pricer(S0, K, T, r_n, 0.0, sigma_n)

    return float(price)


def merton_iv_row(
    S0: float,
    K: float,
    T: float,
    params: MertonParams,
    r: float,
    q: float,
    option_type: str,
) -> float:
    price = merton_option_price(S0, K, T, params, r, q, option_type=option_type)
    iv = implied_volatility(
        np.array([price]),
        np.array([S0]),
        np.array([K]),
        np.array([T]),
        r,
        q,
        np.array([option_type]),
    )
    return float(iv[0])


def merton_iv_panel(
    panel: pd.DataFrame,
    params: MertonParams,
    r: float,
    q: float,
) -> np.ndarray:
    """IV modèle pour chaque ligne du panel (surface globale)."""
    out = np.empty(len(panel), dtype=float)
    for i, row in enumerate(panel.itertuples()):
        out[i] = merton_iv_row(
            float(row.S),
            float(row.K),
            float(row.T),
            params,
            r,
            q,
            str(row.option_type),
        )
    return out
