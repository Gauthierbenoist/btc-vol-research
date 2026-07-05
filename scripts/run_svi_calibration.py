#!/usr/bin/env python
"""Calibration SVI (baseline) par maturité — Neon → smiles."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from btc_vol_research.config import load_config
from btc_vol_research.data.loader import load_snapshot
from btc_vol_research.data.panel import build_market_panel
from btc_vol_research.models.calibration_weights import calibration_weights_v2
from btc_vol_research.models.svi.calibrate import SVICalibrationResult, calibrate_all_slices
from btc_vol_research.models.svi.formula import svi_iv_from_log_moneyness
from btc_vol_research.surfaces.plots import (
    plot_calibration_fit,
    plot_svi_rmse_by_zone,
    plot_svi_rmse_zones_heatmap,
    plot_svi_rho_term_structure,
    plot_svi_surface,
)
from btc_vol_research.analysis.iv_diagnostics import (
    svi_rmse_by_zone,
)
from btc_vol_research.analysis.svi_metrics import svi_summary_table, svi_term_structure_table
from btc_vol_research.surfaces.svi_surface import build_svi_surface_grid, surface_to_long_dataframe
from btc_vol_research.analysis.report import write_svi_calibration_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration SVI baseline sur données Neon")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument(
        "--max-slices",
        type=int,
        default=8,
        help="Maturités pour les PNG smile individuels uniquement (0 = toutes). "
        "rho(T), surface et CSV utilisent toutes les maturités éligibles.",
    )
    return p.parse_args()


def _emit_svi_figures(
    results_all: list[SVICalibrationResult],
    results_plots: list[SVICalibrationResult],
    panel,
    cfg,
    snap_str: str,
    *,
    file_prefix: str,
    surface_stem: str,
    rho_stem: str,
    model_label: str,
    title_suffix: str,
) -> None:
    """Fits par maturité, surface 3D/contour, rho(T), CSV calibration."""
    fig_dir = cfg.figures_dir / "svi"
    for r in results_plots:
        g = panel.loc[panel["slice_id"] == r.slice_id].sort_values("log_moneyness")
        k_fine = np.linspace(g["log_moneyness"].min(), g["log_moneyness"].max(), 150)
        iv_fine = svi_iv_from_log_moneyness(k_fine, r.T, r.params)
        plot_calibration_fit(
            g,
            r.model_iv,
            fig_dir,
            snap_str,
            r.slice_id,
            model_name=model_label,
            file_prefix=file_prefix,
            smooth_curve=(k_fine, iv_fine),
        )

    if len(results_all) >= 2:
        path_3d, path_contour = plot_svi_surface(
            results_all,
            panel,
            fig_dir,
            snap_str,
            file_stem=surface_stem,
            title_suffix=title_suffix,
        )
        lm_g, T_g, iv_g = build_svi_surface_grid(results_all, panel)
        surface_csv = cfg.reports_dir / f"{surface_stem}_{snap_str}.csv"
        surface_csv.parent.mkdir(parents=True, exist_ok=True)
        surface_to_long_dataframe(lm_g, T_g, iv_g, snap_str).to_csv(surface_csv, index=False)
        print(f"  Surface : {path_3d.name}, {path_contour.name} | CSV {surface_csv.name}")

    ts = svi_term_structure_table(results_all)
    rho_csv = cfg.reports_dir / f"{rho_stem}_{snap_str}.csv"
    ts.to_csv(rho_csv, index=False)
    plot_svi_rho_term_structure(
        results_all,
        fig_dir,
        snap_str,
        file_stem=rho_stem,
        title_suffix=title_suffix,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config()
    fig_dir = cfg.figures_dir / "svi"
    snap = date.fromisoformat(args.date) if args.date else None
    if snap is None and cfg.snapshot_date:
        snap = date.fromisoformat(str(cfg.snapshot_date))

    raw = load_snapshot(snap)
    snapshot_date = raw["snapshot_date"].iloc[0]
    snap_str = str(snapshot_date)
    panel = build_market_panel(raw, cfg)

    results_all = calibrate_all_slices(panel, cfg)
    if not results_all:
        print("Aucune tranche SVI calibrée.", file=sys.stderr)
        return 1

    n_eligible = panel["slice_id"].nunique()
    if args.max_slices > 0:
        keep = set(panel.groupby("slice_id").size().sort_values(ascending=False).head(args.max_slices).index)
        results_plots = [r for r in results_all if r.slice_id in keep]
    else:
        results_plots = results_all

    if len(results_all) < 2:
        print("Surface SVI : au moins 2 maturités éligibles requises.", file=sys.stderr)

    table = svi_summary_table(results_all)
    report_path = write_svi_calibration_report(results_all, cfg.reports_dir, snap_str)
    print(f"Snapshot {snapshot_date} — SVI : {len(results_all)}/{n_eligible} maturités calibrées")
    print(f"  (smiles PNG : {len(results_plots)})\n")
    print(table.to_string(index=False))
    print(f"\nRapport: {report_path}")

    atm_w = cfg.calibration.atm_zone_half_width
    svi_zones = svi_rmse_by_zone(results_all, atm_w)
    diag_dir = cfg.reports_dir
    svi_zones.to_csv(diag_dir / f"svi_rmse_by_zone_{snap_str}.csv", index=False)
    plot_svi_rmse_by_zone(svi_zones, fig_dir, snap_str)
    plot_svi_rmse_zones_heatmap(svi_zones, fig_dir, snap_str)
    print("\n=== RMSE SVI par zone (moyenne sur maturites) ===")
    print(
        svi_zones.groupby("zone")[["rmse_svi", "bias_model_minus_mkt"]]
        .mean()
        .assign(rmse_svi_pct=lambda d: d["rmse_svi"] * 100)
        .to_string()
    )
    print(f"\nDiagnostics : svi_rmse_by_zone_{snap_str}.png")

    for r in results_plots:
        rmse = r.rmse_iv if np.isfinite(r.rmse_iv) else float("nan")
        print(f"  {r.slice_id}: RMSE IV={rmse:.4f} (pondere {r.weighted_rmse_iv:.4f})")

    print("\n--- SVI v1 (poids vega * sqrt(OI)) ---")
    _emit_svi_figures(
        results_all,
        results_plots,
        panel,
        cfg,
        snap_str,
        file_prefix="svi",
        surface_stem="svi_surface",
        rho_stem="svi_rho_term",
        model_label="SVI",
        title_suffix="",
    )

    results_w2 = calibrate_all_slices(panel, cfg, weight_fn=calibration_weights_v2)
    if results_w2:
        results_plots_w2 = (
            [r for r in results_w2 if r.slice_id in keep] if args.max_slices > 0 else results_w2
        )
        report_w2 = write_svi_calibration_report(
            results_w2, cfg.reports_dir, snap_str, filename_prefix="svi_calibration_v2"
        )
        print(f"\n--- SVI v2 (poids vega * sqrt(1+OI) * sqrt(1+volume)) : {len(results_w2)} maturites ---")
        print(f"  Rapport: {report_w2.name}")
        for r in results_plots_w2:
            rmse = r.rmse_iv if np.isfinite(r.rmse_iv) else float("nan")
            print(f"  {r.slice_id}: RMSE IV={rmse:.4f} (pondere {r.weighted_rmse_iv:.4f})")
        _emit_svi_figures(
            results_w2,
            results_plots_w2,
            panel,
            cfg,
            snap_str,
            file_prefix="svi_v2",
            surface_stem="svi_surface_v2",
            rho_stem="svi_rho_term_v2",
            model_label="SVI v2",
            title_suffix=" (poids v2)",
        )
    else:
        print("\nAucune tranche SVI v2 calibree.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
