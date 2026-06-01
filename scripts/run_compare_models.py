#!/usr/bin/env python
"""Compare SVI (baseline) vs Heston sur les mêmes tranches."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_vol_research.config import load_config  # noqa: E402
from btc_vol_research.data.loader import load_snapshot  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402
from btc_vol_research.models.svi.calibrate import calibrate_all_slices as calibrate_svi  # noqa: E402
from btc_vol_research.models.heston.calibrate import calibrate_all_slices as calibrate_heston  # noqa: E402
from btc_vol_research.analysis.svi_metrics import svi_summary_table  # noqa: E402
from btc_vol_research.analysis.metrics import calibration_summary_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVI vs Heston — tableau comparatif RMSE")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument("--max-slices", type=int, default=4, help="Maturités (Heston est lent)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    snap = date.fromisoformat(args.date) if args.date else None
    raw = load_snapshot(snap)
    snap_str = str(raw["snapshot_date"].iloc[0])
    panel = build_market_panel(raw, cfg)

    counts = panel.groupby("slice_id").size().sort_values(ascending=False)
    keep = counts.head(args.max_slices).index.tolist()
    panel_sub = panel.loc[panel["slice_id"].isin(keep)].copy()

    print("Calibration SVI (baseline)...")
    svi_res = calibrate_svi(panel_sub, cfg)
    svi_tbl = svi_summary_table(svi_res)[["slice_id", "rmse_iv", "weighted_rmse_iv"]].rename(
        columns={"rmse_iv": "rmse_svi", "weighted_rmse_iv": "w_rmse_svi"}
    )

    print("Calibration Heston...")
    heston_res = calibrate_heston(panel_sub, cfg)
    h_tbl = calibration_summary_table(heston_res)[["slice_id", "rmse_iv", "weighted_rmse_iv"]].rename(
        columns={"rmse_iv": "rmse_heston", "weighted_rmse_iv": "w_rmse_heston"}
    )

    cmp_df = pd.merge(svi_tbl, h_tbl, on="slice_id", how="outer")
    cmp_df["delta_rmse"] = cmp_df["rmse_heston"] - cmp_df["rmse_svi"]
    out = cfg.reports_dir / f"model_compare_{snap_str}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(out, index=False)
    print(f"\nComparaison — {snap_str}\n")
    print(cmp_df.to_string(index=False))
    print(f"\nRapport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
