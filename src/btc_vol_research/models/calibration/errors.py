"""Métriques d'erreur IV communes."""

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


def iv_error_variance(market_iv: np.ndarray, model_iv: np.ndarray) -> float:
    """Variance des ecarts model - market (meme unite que les IV d'entree)."""
    err = np.asarray(model_iv) - np.asarray(market_iv)
    return float(np.var(err))


def iv_weighted_rmse(
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    weights: np.ndarray,
) -> float:
    err2 = (np.asarray(model_iv) - np.asarray(market_iv)) ** 2
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.mean(w * err2 / (w.sum() + 1e-12))))
