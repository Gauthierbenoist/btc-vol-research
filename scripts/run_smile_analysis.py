#!/usr/bin/env python
"""Analyse des smiles et surface IV à partir de Neon."""

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
from btc_vol_research.data.loader import load_snapshot, list_snapshot_dates  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402
from btc_vol_research.surfaces.smile import summarize_moneyness_maturity  # noqa: E402
from btc_vol_research.surfaces.plots import plot_smiles, plot_iv_surface  # noqa: E402
from btc_vol_research.analysis.report import write_smile_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smiles et surface IV BTC (Neon)")
    p.add_argument("--date", type=str, help="YYYY-MM-DD (défaut: dernière date Neon)")
    p.add_argument("--list-dates", action="store_true", help="Lister les snapshots disponibles")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    fig_dir = cfg.figures_dir / "svi"

    if args.list_dates:
        for d in list_snapshot_dates():
            print(d)
        return 0

    snap = date.fromisoformat(args.date) if args.date else None
    if snap is None and cfg.snapshot_date:
        snap = date.fromisoformat(str(cfg.snapshot_date))

    raw = load_snapshot(snap)
    snapshot_date = raw["snapshot_date"].iloc[0]
    panel = build_market_panel(raw, cfg)
    summary = summarize_moneyness_maturity(panel)

    snap_str = str(snapshot_date)
    paths = plot_smiles(panel, fig_dir, snap_str)
    surface_path = plot_iv_surface(panel, fig_dir, snap_str)

    report_dir = cfg.reports_dir
    write_smile_report(summary, report_dir / f"smile_summary_{snap_str}.csv", snap_str)

    print(f"Snapshot: {snapshot_date} | {len(panel)} options après filtres")
    print(summary.to_string(index=False))
    print(f"\nFigures: {len(paths)} smiles + {surface_path}")
    print(f"Rapport: {report_dir / f'smile_summary_{snap_str}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
