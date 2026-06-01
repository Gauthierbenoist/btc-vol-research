"""Zones de moneyness pour diagnostics (ailes / ATM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_LEFT = "left_wing"
ZONE_ATM = "atm"
ZONE_RIGHT = "right_wing"
ZONE_ORDER = [ZONE_LEFT, ZONE_ATM, ZONE_RIGHT]


def assign_moneyness_zone(log_moneyness: np.ndarray | pd.Series, atm_half_width: float) -> np.ndarray:
    """left : k < -w ; atm : |k| <= w ; right : k > w."""
    lm = np.asarray(log_moneyness, dtype=float)
    zones = np.full(len(lm), ZONE_ATM, dtype=object)
    zones[lm < -atm_half_width] = ZONE_LEFT
    zones[lm > atm_half_width] = ZONE_RIGHT
    return zones


def zone_label_fr(zone: str) -> str:
    return {
        ZONE_LEFT: "Aile gauche",
        ZONE_ATM: "ATM",
        ZONE_RIGHT: "Aile droite",
    }.get(zone, zone)
