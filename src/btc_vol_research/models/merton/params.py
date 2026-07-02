"""Paramètres Merton jump-diffusion (1976)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MertonParams:
    """
    Sauts log-normaux : ln(S+/S) ~ N(mu_jump, sigma_jump^2).

    - sigma : volatilité de diffusion
    - lambda_jump : intensité des sauts (par an)
    - mu_jump : moyenne du saut log
    - sigma_jump : écart-type du saut log
    """

    sigma: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    def as_array(self) -> np.ndarray:
        return np.array([self.sigma, self.lambda_jump, self.mu_jump, self.sigma_jump])

    @classmethod
    def from_array(cls, x: np.ndarray) -> MertonParams:
        return cls(sigma=x[0], lambda_jump=x[1], mu_jump=x[2], sigma_jump=x[3])

    def jump_compensation(self) -> float:
        """k = E[exp(J) - 1] pour la dérive compensée."""
        return float(np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0)

    def is_valid(self, eps: float = 1e-8) -> bool:
        return (
            self.sigma > eps
            and self.lambda_jump >= 0.0
            and self.sigma_jump >= 0.0
            and np.isfinite(self.as_array()).all()
        )
