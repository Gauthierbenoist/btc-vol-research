"""Rapports CSV de calibration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from btc_vol_research.analysis.tables import (
    heston_summary_table,
    merton_global_summary_table,
    slice_fit_summary_table,
    svi_summary_table,
)
from btc_vol_research.calibration.results import GlobalCalibrationResult, SliceCalibrationResult


def write_heston_calibration_report(
    results: list[SliceCalibrationResult],
    out_dir: Path,
    snapshot_date: date | str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = heston_summary_table(results)
    path = out_dir / f"heston_calibration_{snapshot_date}.csv"
    table.to_csv(path, index=False)
    return path


def write_svi_calibration_report(
    results: list[SliceCalibrationResult],
    out_dir: Path,
    snapshot_date: date | str,
    *,
    filename_prefix: str = "svi_calibration",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = svi_summary_table(results)
    path = out_dir / f"{filename_prefix}_{snapshot_date}.csv"
    table.to_csv(path, index=False)
    return path


def write_merton_calibration_report(
    result: GlobalCalibrationResult,
    out_dir: Path,
    snapshot_date: date | str,
    *,
    scheme_id: str = "uniform",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    global_path = out_dir / f"merton_calibration_{scheme_id}_{snapshot_date}.csv"
    slice_path = out_dir / f"merton_calibration_by_slice_{scheme_id}_{snapshot_date}.csv"
    merton_global_summary_table(result).to_csv(global_path, index=False)
    slice_fit_summary_table(result.slice_results).to_csv(slice_path, index=False)
    return global_path, slice_path
