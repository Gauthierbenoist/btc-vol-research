"""Métriques d'erreur IV et prix communes."""

from __future__ import annotations

import numpy as np


def sse_objective(market_iv: np.ndarray, model_iv: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Somme des erreurs au carré (non normalisée) — fonction de coût pour scipy.optimize."""
    err2 = (np.asarray(model_iv) - np.asarray(market_iv)) ** 2
    if weights is None:
        return float(np.sum(err2))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * err2))


def iv_rmse(market_iv: np.ndarray, model_iv: np.ndarray) -> float:
    err = np.asarray(model_iv) - np.asarray(market_iv)
    return float(np.sqrt(np.mean(err**2)))


def price_rmse(
    market_price: np.ndarray,
    model_iv: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    option_types: np.ndarray,
) -> float:
    """RMSE entre le prix marché observé et le prix modèle (BS à partir de model_iv), en USD.

    Utilise directement market_price (ex: mark_price * S, déjà fourni par Deribit) plutôt que
    de reconstruire un "prix marché" via Black-Scholes(market_iv) — cela évite un aller-retour
    BS inutile et reflète l'erreur réelle de trading/hedging, indépendante des poids de calibration.
    """
    from btc_vol_research.models.black_scholes import bs_call_price_vec, bs_put_price_vec

    S = np.asarray(S, dtype=float)
    model_iv = np.asarray(model_iv, dtype=float)
    is_call = np.char.lower(np.asarray(option_types).astype(str)) == "call"
    call_px = bs_call_price_vec(S, K, T, r, q, model_iv)
    put_px = bs_put_price_vec(S, K, T, r, q, model_iv)
    model_px = np.where(is_call, call_px, put_px) * S
    err = model_px - np.asarray(market_price, dtype=float)
    return float(np.sqrt(np.mean(err**2)))
