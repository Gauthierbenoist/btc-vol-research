"""Smiles de volatilité par maturité."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def smile_slices(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Découpe le panel en tranches par maturité (slice_id)."""
    return {sid: g.sort_values("log_moneyness") for sid, g in panel.groupby("slice_id")}


def iter_slices(panel: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    for sid, g in panel.groupby("slice_id"):
        yield sid, g.sort_values("log_moneyness")


def summarize_moneyness_maturity(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques par maturité : niveau ATM, pentes skew, term structure du ATM.
    """
    rows = []
    for sid, g in panel.groupby("slice_id"):
        atm_idx = g["log_moneyness"].abs().idxmin()
        atm = g.loc[atm_idx]
        otm_calls = g.loc[g["option_type"].str.lower() == "call"]
        otm_puts = g.loc[g["option_type"].str.lower() == "put"]
        skew = np.nan
        if len(otm_puts) >= 3 and len(otm_calls) >= 3:
            # proxy skew : put OTM 25d - call OTM 25d (approx par quantiles de log-moneyness)
            q_put = otm_puts.nsmallest(max(1, len(otm_puts) // 4), "log_moneyness")["iv_used"].mean()
            q_call = otm_calls.nlargest(max(1, len(otm_calls) // 4), "log_moneyness")["iv_used"].mean()
            skew = float(q_put - q_call)
        rows.append(
            {
                "slice_id": sid,
                "T_years": float(g["T"].iloc[0]),
                "n_strikes": len(g),
                "iv_atm": float(atm["iv_used"]),
                "iv_mark_atm": float(atm["iv_mark"]),
                "skew_proxy": skew,
                "underlying": float(atm["S"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("T_years")
    return summary
