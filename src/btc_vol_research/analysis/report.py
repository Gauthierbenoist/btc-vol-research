"""Rapports texte / CSV."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from btc_vol_research.models.heston.calibrate import CalibrationResult
from btc_vol_research.models.merton.calibrate import GlobalCalibrationResult
from btc_vol_research.models.merton.params import MertonParams
from btc_vol_research.analysis.calibration_tables import (
    merton_global_summary_table,
    slice_fit_summary_table,
)
from btc_vol_research.models.svi.calibrate import SVICalibrationResult
from btc_vol_research.analysis.metrics import calibration_summary_table
from btc_vol_research.analysis.svi_metrics import svi_summary_table


def write_smile_report(summary: pd.DataFrame, out_path: Path, snapshot_date: date | str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    lines = [f"# Smile summary — {snapshot_date}", "", summary.to_string(index=False)]
    out_path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


def write_calibration_report(
    results: list[CalibrationResult],
    out_dir: Path,
    snapshot_date: date | str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = calibration_summary_table(results)
    path = out_dir / f"heston_calibration_{snapshot_date}.csv"
    table.to_csv(path, index=False)
    return path


def write_svi_calibration_report(
    results: list[SVICalibrationResult],
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
    result: GlobalCalibrationResult[MertonParams],
    out_dir: Path,
    snapshot_date: date | str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    global_path = out_dir / f"merton_calibration_{snapshot_date}.csv"
    slice_path = out_dir / f"merton_calibration_by_slice_{snapshot_date}.csv"
    merton_global_summary_table(result).to_csv(global_path, index=False)
    slice_fit_summary_table(result.slice_results).to_csv(slice_path, index=False)
    return global_path, slice_path
