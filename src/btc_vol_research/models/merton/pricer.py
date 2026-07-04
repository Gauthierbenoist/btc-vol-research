"""Prix européens Merton jump-diffusion (série de Poisson × Black–Scholes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import factorial

from btc_vol_research.iv.black_scholes import bs_call_price_vec, bs_put_price_vec, implied_volatility
from btc_vol_research.models.merton.params import MertonParams

_MAX_POISSON_TERMS = 60
_POISSON_FACTORS = factorial(np.arange(_MAX_POISSON_TERMS, dtype=float))


def _as_option_types(option_types: np.ndarray | None, n: int) -> np.ndarray:
    if option_types is None:
        return np.full(n, "call", dtype=object)
    return np.asarray(option_types, dtype=object)


def merton_option_prices(
    S0: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    params: MertonParams,
    r: float = 0.0,
    q: float = 0.0,
    *,
    option_types: np.ndarray | None = None,
) -> np.ndarray:
    """Prix Merton vectorises sur (S, K, T)."""
    s0 = np.asarray(S0, dtype=float)
    k = np.asarray(K, dtype=float)
    t = np.asarray(T, dtype=float)
    if s0.ndim == 0:
        s0 = np.array([float(s0)])
        k = np.array([float(k)])
        t = np.array([float(t)])
    n_pts = s0.size
    opt = _as_option_types(option_types, n_pts)
    is_call = np.char.lower(opt.astype(str)) == "call"

    t_safe = np.maximum(t, 1e-10)
    jump_k = params.jump_compensation()
    lam_t = params.lambda_jump * t_safe
    jump_drift = params.mu_jump + 0.5 * params.sigma_jump**2

    prices = np.zeros(n_pts, dtype=float)
    for n in range(_MAX_POISSON_TERMS):
        pois_prob = np.exp(-lam_t) * np.power(lam_t, n) / _POISSON_FACTORS[n]
        if n > 0 and float(np.max(pois_prob)) < 1e-14:
            break
        sigma_n = np.sqrt(params.sigma**2 + n * params.sigma_jump**2 / t_safe)
        s_n = s0 * np.exp(-params.lambda_jump * jump_k * t_safe + n * jump_drift)
        bs_call = bs_call_price_vec(s_n, k, t_safe, r, q, sigma_n)
        bs_put = bs_put_price_vec(s_n, k, t_safe, r, q, sigma_n)
        prices += pois_prob * np.where(is_call, bs_call, bs_put)

    short = t <= 0
    if np.any(short):
        intrinsic_call = np.maximum(s0 * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
        intrinsic_put = np.maximum(k * np.exp(-r * t) - s0 * np.exp(-q * t), 0.0)
        prices = np.where(short, np.where(is_call, intrinsic_call, intrinsic_put), prices)

    return prices


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
    return float(
        merton_option_prices(
            np.array([S0]),
            np.array([K]),
            np.array([T]),
            params,
            r,
            q,
            option_types=np.array([option_type]),
        )[0]
    )


def merton_iv_row(
    S0: float,
    K: float,
    T: float,
    params: MertonParams,
    r: float,
    q: float,
    option_type: str,
) -> float:
    iv = merton_iv_panel(
        pd.DataFrame(
            {
                "S": [S0],
                "K": [K],
                "T": [T],
                "option_type": [option_type],
            }
        ),
        params,
        r,
        q,
    )
    return float(iv[0])


def merton_iv_panel(
    panel: pd.DataFrame,
    params: MertonParams,
    r: float,
    q: float,
) -> np.ndarray:
    """IV modele pour chaque ligne du panel (surface globale, vectorise)."""
    prices = merton_option_prices(
        panel["S"].values.astype(float),
        panel["K"].values.astype(float),
        panel["T"].values.astype(float),
        params,
        r,
        q,
        option_types=panel["option_type"].values,
    )
    return implied_volatility(
        prices,
        panel["S"].values.astype(float),
        panel["K"].values.astype(float),
        panel["T"].values.astype(float),
        r,
        q,
        panel["option_type"].values,
    )
