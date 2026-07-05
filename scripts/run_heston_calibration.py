#!/usr/bin/env python
"""Calibration Heston pondérée par maturité (Neon → smiles)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from btc_vol_research.analysis.report import write_heston_calibration_report
from btc_vol_research.analysis.tables import heston_summary_table
from btc_vol_research.calibration.heston import calibrate_all_slices
from btc_vol_research.config import load_config
from btc_vol_research.data.loader import load_snapshot
from btc_vol_research.data.panel import build_market_panel
from btc_vol_research.surfaces.plots import plot_calibration_fit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration Heston sur données Neon")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument("--max-slices", type=int, default=6, help="Nombre max de maturités à calibrer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    fig_dir = cfg.figures_dir / "heston"
    snap = date.fromisoformat(args.date) if args.date else None
    if snap is None and cfg.snapshot_date:
        snap = date.fromisoformat(str(cfg.snapshot_date))

    raw = load_snapshot(snap)
    snapshot_date = raw["snapshot_date"].iloc[0]
    snap_str = str(snapshot_date)
    panel = build_market_panel(raw, cfg)

    # Tranches les plus liquides (plus de strikes)
    counts = panel.groupby("slice_id").size().sort_values(ascending=False)
    keep = counts.head(args.max_slices).index.tolist()
    panel_sub = panel.loc[panel["slice_id"].isin(keep)].copy()

    results = calibrate_all_slices(panel_sub, cfg)
    if not results:
        print("Aucune tranche calibrée — vérifiez les filtres ou MIN strikes.", file=sys.stderr)
        return 1

    table = heston_summary_table(results)
    report_path = write_heston_calibration_report(results, cfg.reports_dir, snap_str)
    print(table.to_string(index=False))
    print(f"\nRapport: {report_path}")

    for r in results:
        g = panel_sub.loc[panel_sub["slice_id"] == r.slice_id].sort_values("log_moneyness")
        plot_calibration_fit(
            g,
            r.model_iv,
            fig_dir,
            snap_str,
            r.slice_id,
            model_name="Heston",
            file_prefix="heston",
        )
        print(f"  {r.slice_id}: RMSE IV={r.rmse_iv:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
