#!/usr/bin/env python
"""Calibration Merton jump-diffusion globale (toute la surface IV)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_vol_research.config import load_config  # noqa: E402
from btc_vol_research.data.loader import load_snapshot  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402
from btc_vol_research.models.calibration.filters import quality_filter  # noqa: E402
from btc_vol_research.models.merton.calibrate import calibrate_global  # noqa: E402
from btc_vol_research.analysis.calibration_tables import (  # noqa: E402
    merton_global_summary_table,
    slice_fit_summary_table,
)
from btc_vol_research.analysis.report import write_merton_calibration_report  # noqa: E402
from btc_vol_research.surfaces.plots import plot_calibration_fit  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration Merton globale sur surface IV")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="Limite les PNG par maturite (0 = toutes les tranches calibrees)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    snap = date.fromisoformat(args.date) if args.date else None
    if snap is None and cfg.snapshot_date:
        snap = date.fromisoformat(str(cfg.snapshot_date))

    raw = load_snapshot(snap)
    snapshot_date = raw["snapshot_date"].iloc[0]
    snap_str = str(snapshot_date)
    panel = build_market_panel(raw, cfg)

    result = calibrate_global(panel, cfg)
    fit_df = quality_filter(panel, min_strikes=cfg.calibration.min_strikes_per_slice).reset_index(drop=True)

    global_tbl = merton_global_summary_table(result)
    slice_tbl = slice_fit_summary_table(result.slice_results)
    global_path, slice_path = write_merton_calibration_report(result, cfg.reports_dir, snap_str)

    print(f"Snapshot {snapshot_date} — Merton global : {result.n_points} points, "
          f"{len(result.slice_results)} maturites")
    print("\n=== Parametres globaux ===")
    print(global_tbl.to_string(index=False))
    print(f"\nRapports: {global_path.name}, {slice_path.name}")
    print("\n=== RMSE par maturite ===")
    print(slice_tbl.to_string(index=False))

    plot_slices = result.slice_results
    if args.max_slices > 0:
        top = slice_tbl.sort_values("n_strikes", ascending=False).head(args.max_slices)["slice_id"]
        plot_slices = [s for s in result.slice_results if s.slice_id in set(top)]

    for sr in plot_slices:
        g = fit_df.loc[fit_df["slice_id"] == sr.slice_id].sort_values("log_moneyness")
        plot_calibration_fit(
            g,
            sr.model_iv,
            cfg.figures_dir,
            snap_str,
            sr.slice_id,
            model_name="Merton",
            file_prefix="merton",
        )
        print(f"  {sr.slice_id}: RMSE IV={sr.rmse_iv:.4f}")

    print(f"\nFigures: merton_fit_{snap_str}_*.png ({len(plot_slices)} tranches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
