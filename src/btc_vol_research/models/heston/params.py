"""Paramètres du modèle de Heston."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def as_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        return cls(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

    def feller_satisfied(self, eps: float = 1e-8) -> bool:
        return 2 * self.kappa * self.theta > self.sigma**2 + eps
