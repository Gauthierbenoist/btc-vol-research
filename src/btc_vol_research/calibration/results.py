"""Structures de résultats de calibration partagées."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

ParamsT = TypeVar("ParamsT")


@dataclass
class SliceCalibrationResult(Generic[ParamsT]):
    """Résultat d'une calibration par tranche de maturité (SVI, Heston)."""

    slice_id: str
    params: ParamsT
    rmse_iv: float
    market_iv: np.ndarray
    model_iv: np.ndarray
    log_moneyness: np.ndarray
    T: float
    success: bool
    message: str


@dataclass
class SliceFitResult:
    """Fit par maturité redécoupé à partir d'une calibration globale."""

    slice_id: str
    T: float
    rmse_iv: float
    market_iv: np.ndarray
    model_iv: np.ndarray
    log_moneyness: np.ndarray


@dataclass
class GlobalCalibrationResult(Generic[ParamsT]):
    """Résultat d'une calibration globale sur toute la surface (Merton)."""

    params: ParamsT
    rmse_iv: float
    slice_results: list[SliceFitResult]
    n_points: int
    success: bool
    message: str
    weight_scheme: str = "uniform"
    calibration_time_s: float = 0.0
