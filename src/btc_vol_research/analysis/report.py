"""Rapports texte / CSV."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from btc_vol_research.models.heston.calibrate import CalibrationResult
from btc_vol_research.analysis.metrics import calibration_summary_table


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
