"""Fonction caractéristique Heston (convention Wikipedia / Heston 1993)."""

from __future__ import annotations

import numpy as np


def heston_characteristic(
    nu: complex,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    q: float,
    S0: float,
) -> complex:
    """φ(ν) = E[exp(i ν ln S_T)] avec spot S0."""
    a = kappa * theta
    d = np.sqrt((rho * sigma * 1j * nu - kappa) ** 2 + sigma**2 * (1j * nu + nu**2))
    g = (kappa - rho * sigma * 1j * nu - d) / (kappa - rho * sigma * 1j * nu + d)
    exp_dt = np.exp(-d * T)
    C = 1j * nu * (r - q) * T + (a / sigma**2) * (
        (kappa - rho * sigma * 1j * nu - d) * T - 2 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = (kappa - rho * sigma * 1j * nu - d) / sigma**2 * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(C + D * v0 + 1j * nu * np.log(S0))
