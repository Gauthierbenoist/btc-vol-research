"""Inversion de volatilité implicite (Peter Jäckel — Let's Be Rational)."""

from __future__ import annotations

import numpy as np
from py_lets_be_rational.exceptions import AboveMaximumException, BelowIntrinsicException
from py_lets_be_rational.lets_be_rational import implied_volatility_from_a_transformed_rational_guess


def _implied_volatility_lbr_scalar(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
) -> float:
    """BSM -> forward Black, puis inversion LBR (prix non actualise)."""
    F = S * np.exp((r - q) * T)
    undiscounted_price = price * np.exp(r * T)
    q_flag = 1.0 if is_call else -1.0
    return float(
        implied_volatility_from_a_transformed_rational_guess(
            undiscounted_price,
            F,
            K,
            T,
            q_flag,
        )
    )


def implied_volatility(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    option_type: np.ndarray,
    *,
    intrinsic_tol: float = 0.999,
) -> np.ndarray:
    """
    Inversion vectorisee via Let's Be Rational (Jäckel).

    Entree/sortie en tableaux numpy de meme longueur ; NaN si inversion impossible.
    """
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
    valid &= price >= intrinsic * intrinsic_tol

    for i in np.flatnonzero(valid):
        try:
            iv[i] = _implied_volatility_lbr_scalar(
                float(price[i]),
                float(S[i]),
                float(K[i]),
                float(T[i]),
                r,
                q,
                bool(is_call[i]),
            )
        except (BelowIntrinsicException, AboveMaximumException):
            continue

    return iv
