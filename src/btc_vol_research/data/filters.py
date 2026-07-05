"""Filtres et métriques qualité au niveau des données marché (préparation du panel)."""

from __future__ import annotations

import pandas as pd


def relative_spread(bid: pd.Series, ask: pd.Series, mid: pd.Series) -> pd.Series:
    """Spread relatif (ask - bid) / prix de référence.

    NaN si la cote est inexploitable (bid/ask <= 0 ou ask < bid).
    """
    spread = ask - bid
    ref = mid.where(mid > 0, (ask + bid) / 2)
    return (spread / ref).where((bid > 0) & (ask > 0) & (ask >= bid))


def select_otm(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les options hors de la monnaie (calls K > F, puts K < F).

    Les OTM sont les plus liquides sur Deribit : on les prend systématiquement
    pour construire le smile. Requiert les colonnes option_type, K et forward.
    """
    is_call = df["option_type"].str.lower() == "call"
    otm = (is_call & (df["K"] > df["forward"])) | ((~is_call) & (df["K"] < df["forward"]))
    return df.loc[otm].copy()
