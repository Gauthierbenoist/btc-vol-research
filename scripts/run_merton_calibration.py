#!/usr/bin/env python
"""Calibration Merton jump-diffusion globale (toute la surface IV)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from btc_vol_research.analysis.merton_diagnostics import merton_slice_metrics_table
from btc_vol_research.analysis.report import write_merton_calibration_report
from btc_vol_research.analysis.tables import merton_global_summary_table, slice_fit_summary_table
from btc_vol_research.calibration.merton import calibrate_global
from btc_vol_research.calibration.slices import require_min_points
from btc_vol_research.calibration.weights import build_panel_weights, get_merton_weight_scheme
from btc_vol_research.config import load_config
from btc_vol_research.data.loader import load_snapshot
from btc_vol_research.data.panel import build_market_panel
from btc_vol_research.models.merton import merton_iv_panel
from btc_vol_research.surfaces.export import grid_to_long_dataframe
from btc_vol_research.surfaces.merton_surface import (
    build_merton_abs_error_surface_grid,
    build_merton_surface_grid,
)
from btc_vol_research.surfaces.plots import (
    plot_calibration_fit,
    plot_merton_iv_and_abs_error_plotly,
    plot_merton_surface_plotly,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration Merton globale sur surface IV")
    p.add_argument("--date", type=str, help="YYYY-MM-DD")
    p.add_argument(
        "--weight-scheme",
        type=str,
        default="",
        help="Ponderation erreur: uniform | vega | volume (defaut: configs/default.yaml merton.weight_scheme)",
    )
    p.add_argument(
        "--max-slices",
        type=int,
        default=20,
        help="Limite les PNG par maturite (0 = toutes les tranches calibrees)",
    )
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Calibration + CSV uniquement (pas de figures)",
    )
    return p.parse_args()


def _plot_slices(
    result,
    fit_df: pd.DataFrame,
    fig_dir: Path,
    snap_str: str,
    *,
    scheme_id: str,
    scheme_label: str,
    max_slices: int,
) -> int:
    slice_tbl = slice_fit_summary_table(result.slice_results)
    plot_slices = result.slice_results
    if max_slices > 0:
        top = slice_tbl.sort_values("n_strikes", ascending=False).head(max_slices)["slice_id"]
        plot_slices = [s for s in result.slice_results if s.slice_id in set(top)]

    file_prefix = f"merton_{scheme_id}"
    for sr in plot_slices:
        g = fit_df.loc[fit_df["slice_id"] == sr.slice_id].sort_values("log_moneyness")
        plot_calibration_fit(
            g,
            sr.model_iv,
            fig_dir,
            snap_str,
            sr.slice_id,
            model_name=f"Merton ({scheme_label})",
            file_prefix=file_prefix,
        )
        print(f"  {sr.slice_id}: RMSE IV={sr.rmse_iv:.4f}")
    return len(plot_slices)


def _generate_surfaces(
    result,
    fit_df: pd.DataFrame,
    fig_dir: Path,
    reports_dir: Path,
    snap_str: str,
    cfg,
    *,
    scheme_id: str,
    scheme_label: str,
) -> None:
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    model_iv = merton_iv_panel(fit_df, result.params, r, q)

    lm_grid, t_grid, iv_grid = build_merton_surface_grid(fit_df, result.params, r, q)
    _, _, err_grid = build_merton_abs_error_surface_grid(fit_df, model_iv)

    rmse_pct = result.rmse_iv * 100.0
    file_stem = f"merton_{scheme_id}"
    title_suffix = f" ({scheme_label})"

    surface_html = plot_merton_surface_plotly(
        lm_grid,
        t_grid,
        iv_grid,
        fig_dir,
        snap_str,
        file_stem=file_stem,
        title_suffix=title_suffix,
    )
    dual_html = plot_merton_iv_and_abs_error_plotly(
        lm_grid,
        t_grid,
        iv_grid,
        err_grid,
        fig_dir,
        snap_str,
        rmse_pct=rmse_pct,
        file_stem=file_stem,
        title_suffix=title_suffix,
    )
    surface_csv = reports_dir / f"merton_surface_{scheme_id}_{snap_str}.csv"
    err_csv = reports_dir / f"merton_abs_error_surface_{scheme_id}_{snap_str}.csv"
    grid_to_long_dataframe(lm_grid, t_grid, iv_grid, snap_str, "iv_merton").to_csv(surface_csv, index=False)
    grid_to_long_dataframe(lm_grid, t_grid, err_grid, snap_str, "abs_error_iv_pts").to_csv(err_csv, index=False)

    print(f"  Surface 3D: {surface_html.name}")
    print(f"  IV + erreur absolue: {dual_html.name}")
    print(f"  RMSE global: {rmse_pct:.3f} pts vol")
    print(f"  Temps calibration: {result.calibration_time_s:.2f} s")


def _merge_metrics_csv(metrics_path: Path, metrics_tbl: pd.DataFrame, scheme_id: str) -> pd.DataFrame:
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        prev = prev[prev["weight_scheme"] != scheme_id]
        combined = pd.concat([prev, metrics_tbl], ignore_index=True)
    else:
        combined = metrics_tbl
    combined.sort_values(["weight_scheme", "slice_id"]).to_csv(metrics_path, index=False)
    return combined


def main() -> int:
    args = parse_args()
    cfg = load_config()
    scheme_id = args.weight_scheme.strip() or cfg.merton.weight_scheme
    scheme = get_merton_weight_scheme(scheme_id)

    fig_root = cfg.figures_dir / "merton" / scheme.scheme_id
    snap = date.fromisoformat(args.date) if args.date else None
    if snap is None and cfg.snapshot_date:
        snap = date.fromisoformat(str(cfg.snapshot_date))

    raw = load_snapshot(snap)
    snapshot_date = raw["snapshot_date"].iloc[0]
    snap_str = str(snapshot_date)
    panel = build_market_panel(raw, cfg)
    fit_df = require_min_points(panel, cfg.calibration.min_strikes_per_slice).reset_index(drop=True)
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    metrics_path = cfg.reports_dir / f"merton_slice_metrics_{snap_str}.csv"

    print(f"Snapshot {snapshot_date} — Merton scheme={scheme.scheme_id} ({scheme.label})")

    result = calibrate_global(
        fit_df,
        cfg,
        weight_fn=scheme.weight_fn,
        weight_scheme=scheme.scheme_id,
    )

    global_tbl = merton_global_summary_table(result)
    global_path, slice_path = write_merton_calibration_report(
        result,
        cfg.reports_dir,
        snap_str,
        scheme_id=scheme.scheme_id,
    )
    print(global_tbl.to_string(index=False))
    print(f"Rapports: {global_path.name}, {slice_path.name}")

    weights = None
    if scheme.weight_fn is not None:
        weights = build_panel_weights(fit_df, scheme.weight_fn, cfg.calibration, r, q)

    metrics_tbl = merton_slice_metrics_table(
        result,
        fit_df,
        snapshot_date=snap_str,
        weight_scheme_label=scheme.label,
        weights=weights,
        r=r,
        q=q,
    )
    full_metrics = _merge_metrics_csv(metrics_path, metrics_tbl, scheme.scheme_id)

    if not args.metrics_only:
        n_plots = _plot_slices(
            result,
            fit_df,
            fig_root,
            snap_str,
            scheme_id=scheme.scheme_id,
            scheme_label=scheme.label,
            max_slices=args.max_slices,
        )
        print(f"  Smiles traces: {n_plots}")
        _generate_surfaces(
            result,
            fit_df,
            fig_root,
            cfg.reports_dir,
            snap_str,
            cfg,
            scheme_id=scheme.scheme_id,
            scheme_label=scheme.label,
        )

    print(f"\nCSV metriques par maturite: {metrics_path.name}")
    preview = full_metrics.loc[full_metrics["weight_scheme"] == scheme.scheme_id][
        ["weight_scheme", "slice_id", "rmse_uniform", "calibration_time_s"]
    ]
    print("\n=== Metriques (RMSE uniforme, pts vol) ===")
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
