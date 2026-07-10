"""Inversion de volatilité implicite (Let's Be Rational, vectorisé via py_vollib_vectorized)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _ensure_py_vollib_vectorized_no_jit() -> None:
    """Charge py_vollib_vectorized sans JIT numba (LBR incompatible avec nopython)."""
    if "py_vollib_vectorized._iv_models" in sys.modules:
        return

    spec = importlib.util.find_spec("py_vollib_vectorized")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "py_vollib_vectorized is required. Install with: pip install py-vollib-vectorized"
        )

    root = Path(spec.submodule_search_locations[0])
    for pkg_name, pkg_path in (
        ("py_vollib_vectorized", root),
        ("py_vollib_vectorized.util", root / "util"),
    ):
        if pkg_name not in sys.modules:
            mod = types.ModuleType(pkg_name)
            mod.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = mod

    import py_vollib_vectorized.util.jit_helper as jh

    jh.use_jit = False


_ensure_py_vollib_vectorized_no_jit()
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility  # noqa: E402


def _option_type_to_flags(option_type: np.ndarray) -> np.ndarray:
    is_call = np.char.lower(np.asarray(option_type, dtype=str)) == "call"
    return np.where(is_call, "c", "p").astype("U1")


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
    Inversion vectorisee via Let's Be Rational (Jäckel), via py_vollib_vectorized.

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

    if not np.any(valid):
        return iv

    flags = _option_type_to_flags(opt[valid])
    iv[valid] = vectorized_implied_volatility(
        price[valid],
        S[valid],
        K[valid],
        T[valid],
        r,
        flags,
        q=q,
        model="black_scholes_merton",
        return_as="numpy",
        on_error="ignore",
    )
    return iv
