"""SVI brut (Gatheral) — paramètres et formules de variance totale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVIParams:
    """
    Variance totale w(k) = a + b (ρ(k-m) + √((k-m)² + σ²)).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "SVIParams":
        return cls(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])

    def butterfly_ok(self, eps: float = 1e-8) -> bool:
        """Condition suffisante de non-arbitrage papillon (Gatheral)."""
        if self.b < 0 or self.sigma <= 0 or abs(self.rho) >= 1:
            return False
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) >= -eps


def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """w(k) en variance totale (σ²T)."""
    km = np.asarray(k, dtype=float) - params.m
    return params.a + params.b * (params.rho * km + np.sqrt(km**2 + params.sigma**2))


def svi_iv_from_log_moneyness(
    k: np.ndarray,
    T: float,
    params: SVIParams,
) -> np.ndarray:
    """σ_BS(k, T) à partir de w(k) = σ²T."""
    w = svi_total_variance(k, params)
    w = np.maximum(w, 1e-12)
    T = max(T, 1e-10)
    return np.sqrt(w / T)
