"""Métriques d'erreur IV et prix communes."""

from __future__ import annotations

import numpy as np


def iv_sse(market_iv: np.ndarray, model_iv: np.ndarray, weights: np.ndarray | None = None) -> float:
    err2 = (np.asarray(model_iv) - np.asarray(market_iv)) ** 2
    if weights is None:
        return float(np.sum(err2))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * err2))


def iv_rmse(market_iv: np.ndarray, model_iv: np.ndarray) -> float:
    err = np.asarray(model_iv) - np.asarray(market_iv)
    return float(np.sqrt(np.mean(err**2)))


def iv_mae(market_iv: np.ndarray, model_iv: np.ndarray) -> float:
    err = np.asarray(model_iv) - np.asarray(market_iv)
    return float(np.mean(np.abs(err)))


def iv_error_variance(market_iv: np.ndarray, model_iv: np.ndarray) -> float:
    """Variance des ecarts model - market (meme unite que les IV d'entree)."""
    err = np.asarray(model_iv) - np.asarray(market_iv)
    return float(np.var(err))


def price_rmse(
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
) -> float:
    """RMSE entre les prix BS implicites (marché vs modèle), en USD.

    Reflète l'erreur réelle en trading/hedging, indépendante des poids de calibration.
    """
    from btc_vol_research.iv.black_scholes import bs_call_price_vec

    market_px = bs_call_price_vec(S, K, T, r, q, np.asarray(market_iv)) * S
    model_px = bs_call_price_vec(S, K, T, r, q, np.asarray(model_iv)) * S
    err = model_px - market_px
    return float(np.sqrt(np.mean(err**2)))
