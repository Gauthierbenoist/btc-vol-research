"""Pondérations de calibration (vega, liquidité, volume, 1/spread) et schémas nommés."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btc_vol_research.config import CalibrationConfig
from btc_vol_research.market.greeks import bs_vega_vec

WeightFn = Callable[[pd.DataFrame, CalibrationConfig, float, float], np.ndarray]


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    n = len(w)
    return w / (w.sum() + 1e-12) * n


def _vega_vector(slice_df: pd.DataFrame, r: float, q: float) -> np.ndarray:
    n = len(slice_df)
    s0 = float(slice_df["S"].iloc[0])
    t = float(slice_df["T"].iloc[0])
    k = slice_df["K"].values.astype(float)
    sigma = slice_df["iv_used"].values.astype(float)
    return bs_vega_vec(
        np.full(n, s0),
        k,
        np.full(n, t),
        r,
        q,
        sigma,
    )


def _volume_series(slice_df: pd.DataFrame) -> np.ndarray:
    if "volume_24h" in slice_df.columns:
        return slice_df["volume_24h"].astype(float).fillna(0).values
    if "volume" in slice_df.columns:
        return slice_df["volume"].astype(float).fillna(0).values
    return np.zeros(len(slice_df))


def _spread_series(slice_df: pd.DataFrame) -> np.ndarray:
    """Spread relatif par ligne ; NaN (cote sans bid/ask exploitable) -> spread max de la tranche."""
    if "rel_spread" not in slice_df.columns:
        return np.ones(len(slice_df))
    s = slice_df["rel_spread"].astype(float).values
    finite = s[np.isfinite(s)]
    fallback = float(np.max(finite)) if finite.size else 1.0
    return np.where(np.isfinite(s), s, fallback)


def calibration_weights(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """v1 : w_i ∝ vega × √OI (normalisées)."""
    n = len(slice_df)
    w = np.ones(n)

    if cfg.use_vega_weight:
        w *= np.maximum(_vega_vector(slice_df, r, q), 1e-8)

    if cfg.use_liquidity_weight:
        oi = slice_df["open_interest"].astype(float).values
        w *= np.sqrt(np.maximum(oi, 1e-8))

    return _normalize_weights(w)


def calibration_weights_v2(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """v2 : w_i ∝ vega × √(1+OI) × √(1+volume) (normalisées)."""
    n = len(slice_df)
    w = np.ones(n)

    if cfg.use_vega_weight:
        w *= np.maximum(_vega_vector(slice_df, r, q), 1e-8)

    oi = (
        slice_df["open_interest"].astype(float).fillna(0).values
        if "open_interest" in slice_df.columns
        else np.zeros(n)
    )
    vol = _volume_series(slice_df)
    w *= np.sqrt(oi + 1.0) * np.sqrt(vol + 1.0)

    return _normalize_weights(w)


def vega_only_weights(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """Poids ∝ vega (normalisés), sans composante liquidité."""
    del cfg
    w = np.maximum(_vega_vector(slice_df, r, q), 1e-8)
    return _normalize_weights(w)


def volume_only_weights(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """Poids ∝ √(1 + volume) (normalisés)."""
    del cfg, r, q
    vol = _volume_series(slice_df)
    w = np.sqrt(vol + 1.0)
    return _normalize_weights(np.maximum(w, 1e-8))


def spread_only_weights(
    slice_df: pd.DataFrame,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """Poids ∝ 1/spread relatif (normalisés) — cotes serrées = plus fiables."""
    del cfg, r, q
    s = _spread_series(slice_df)
    w = 1.0 / np.maximum(s, 1e-4)
    return _normalize_weights(w)


def build_panel_weights(
    panel: pd.DataFrame,
    weight_fn: WeightFn,
    cfg: CalibrationConfig,
    r: float,
    q: float,
) -> np.ndarray:
    """Assemble les poids par tranche sur l'ordre du panel."""
    w = np.zeros(len(panel), dtype=float)
    for _, g in panel.groupby("slice_id", sort=True):
        pos = panel.index.get_indexer(g.index)
        w[pos] = weight_fn(g, cfg, r, q)
    return w


# --- Schémas nommés (Merton global) -------------------------------------------------


@dataclass(frozen=True)
class MertonWeightScheme:
    scheme_id: str
    label: str
    weight_fn: WeightFn | None


MERTON_WEIGHT_SCHEMES: tuple[MertonWeightScheme, ...] = (
    MertonWeightScheme("uniform", "sans ponderation", None),
    MertonWeightScheme("vega", "vega", vega_only_weights),
    MertonWeightScheme("volume", "volume", volume_only_weights),
    MertonWeightScheme("spread", "1/spread", spread_only_weights),
)

_SCHEME_BY_ID = {s.scheme_id: s for s in MERTON_WEIGHT_SCHEMES}


def get_merton_weight_scheme(scheme_id: str) -> MertonWeightScheme:
    key = scheme_id.strip().lower()
    if key not in _SCHEME_BY_ID:
        known = ", ".join(_SCHEME_BY_ID)
        raise ValueError(f"Ponderation Merton inconnue: {scheme_id!r} (attendu: {known})")
    return _SCHEME_BY_ID[key]
