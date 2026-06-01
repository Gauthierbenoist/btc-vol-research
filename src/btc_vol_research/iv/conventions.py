"""Conventions de smile (OTM, spreads)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_spread(bid: pd.Series, ask: pd.Series, mid: pd.Series) -> pd.Series:
    spread = ask - bid
    ref = mid.where(mid > 0, (ask + bid) / 2)
    return (spread / ref).where((bid > 0) & (ask > 0) & (ask >= bid))


def apply_smile_convention(df: pd.DataFrame, convention: str) -> pd.DataFrame:
    """Filtre les options selon la convention de construction du smile."""
    c = convention.lower()
    if c == "all":
        return df
    if c == "calls":
        return df.loc[df["option_type"].str.lower() == "call"].copy()
    if c == "puts":
        return df.loc[df["option_type"].str.lower() == "put"].copy()
    # otm (défaut)
    is_call = df["option_type"].str.lower() == "call"
    otm = (is_call & (df["K"] > df["forward"])) | ((~is_call) & (df["K"] < df["forward"]))
    return df.loc[otm].copy()
