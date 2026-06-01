#!/usr/bin/env python
"""Calibration SVI (baseline) par maturité — Neon → smiles."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_vol_research.config import load_config  # noqa: E402
from btc_vol_research.data.loader import load_snapshot  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402
from btc_vol_research.models.svi.calibrate import calibrate_all_slices  # noqa: E402
from btc_vol_research.models.svi.formula import svi_iv_from_log_moneyness  # noqa: E402
from btc_vol_research.surfaces.plots import (  # noqa: E402
    plot_calibration_fit,
    plot_svi_rho_term_structure,
    plot_svi_surface,
)
from btc_vol_research.analysis.svi_metrics import svi_summary_table, svi_term_structure_table  # noqa: E402
from btc_vol_research.surfaces.svi_surface import build_svi_surface_grid, surface_to_long_dataframe  # noqa: E402
from btc_vol_research.analysis.report import write_svi_calibration_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration SVI baseline sur données Neon")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument("--max-slices", type=int, default=8, help="Nombre max de maturités")
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

    counts = panel.groupby("slice_id").size().sort_values(ascending=False)
    keep = counts.head(args.max_slices).index.tolist()
    panel_sub = panel.loc[panel["slice_id"].isin(keep)].copy()

    results = calibrate_all_slices(panel_sub, cfg)
    if not results:
        print("Aucune tranche SVI calibrée.", file=sys.stderr)
        return 1
    if len(results) < 2:
        print("Surface SVI : au moins 2 maturités requises (augmentez --max-slices).", file=sys.stderr)

    table = svi_summary_table(results)
    report_path = write_svi_calibration_report(results, cfg.reports_dir, snap_str)
    print(f"Snapshot {snapshot_date} — baseline SVI ({len(results)} tranches)\n")
    print(table.to_string(index=False))
    print(f"\nRapport: {report_path}")

    ts = svi_term_structure_table(results)
    rho_csv = cfg.reports_dir / f"svi_rho_term_{snap_str}.csv"
    rho_csv.parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(rho_csv, index=False)
    rho_path = plot_svi_rho_term_structure(results, cfg.figures_dir, snap_str)
    print(f"rho(T) : {rho_path.name} | CSV : {rho_csv.name}")

    for r in results:
        g = panel_sub.loc[panel_sub["slice_id"] == r.slice_id].sort_values("log_moneyness")
        k_fine = np.linspace(g["log_moneyness"].min(), g["log_moneyness"].max(), 150)
        iv_fine = svi_iv_from_log_moneyness(k_fine, r.T, r.params)
        plot_calibration_fit(
            g,
            r.model_iv,
            cfg.figures_dir,
            snap_str,
            r.slice_id,
            model_name="SVI",
            file_prefix="svi",
            smooth_curve=(k_fine, iv_fine),
        )
        rmse = r.rmse_iv if np.isfinite(r.rmse_iv) else float("nan")
        print(f"  {r.slice_id}: RMSE IV={rmse:.4f} (pondéré {r.weighted_rmse_iv:.4f})")

    if len(results) >= 2:
        path_3d, path_contour = plot_svi_surface(results, panel_sub, cfg.figures_dir, snap_str)
        lm_g, T_g, iv_g = build_svi_surface_grid(results, panel_sub)
        surface_csv = cfg.reports_dir / f"svi_surface_{snap_str}.csv"
        surface_csv.parent.mkdir(parents=True, exist_ok=True)
        surface_to_long_dataframe(lm_g, T_g, iv_g, snap_str).to_csv(surface_csv, index=False)
        print(f"\nSurface SVI : {path_3d.name}, {path_contour.name}")
        print(f"Grille CSV : {surface_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
