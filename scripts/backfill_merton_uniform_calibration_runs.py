#!/usr/bin/env python
"""Backfill Merton uniform calibration runs into Postgres."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from btc_vol_research.calibration.merton import calibrate_global  # noqa: E402
from btc_vol_research.calibration.weights import get_merton_weight_scheme  # noqa: E402
from btc_vol_research.config import load_config  # noqa: E402
from btc_vol_research.data.loader import get_connection, list_snapshot_dates, load_snapshot  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Merton uniform runs into calibration_runs")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of snapshot dates")
    return parser.parse_args()


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _filter_dates(dates: list[date], start: date | None, end: date | None, limit: int) -> list[date]:
    out = [d for d in dates if (start is None or d >= start) and (end is None or d <= end)]
    if limit > 0:
        out = out[:limit]
    return out


def _py_float(value) -> float | None:
    return None if value is None else float(value)


def _py_int(value) -> int | None:
    return None if value is None else int(value)


def _replace_run(cur, *, snapshot_date: date, result, total_runtime_s: float, optimizer_local: str) -> None:
    cur.execute(
        """
        delete from calibration_runs
        where snapshot_date = %s
          and model_name = 'merton'
          and weight_scheme = 'uniform'
          and scope = 'global'
          and slice_id is null
        """,
        (snapshot_date,),
    )
    cur.execute(
        """
        insert into calibration_runs (
            snapshot_date,
            model_name,
            weight_scheme,
            scope,
            slice_id,
            success,
            optimizer_global,
            optimizer_local,
            objective_name,
            n_points,
            n_slices,
            rmse_iv,
            calibration_time_s,
            total_runtime_s,
            sigma,
            lambda_jump,
            mu_jump,
            sigma_jump
        )
        values (
            %s, 'merton', 'uniform', 'global', null,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            snapshot_date,
            result.success,
            "differential_evolution",
            optimizer_local,
            "iv_sse",
            _py_int(result.n_points),
            _py_int(len(result.slice_results)),
            _py_float(result.rmse_iv),
            _py_float(result.calibration_time_s),
            _py_float(total_runtime_s),
            _py_float(result.params.sigma),
            _py_float(result.params.lambda_jump),
            _py_float(result.params.mu_jump),
            _py_float(result.params.sigma_jump),
        ),
    )


def main() -> int:
    args = parse_args()
    cfg = load_config()
    scheme = get_merton_weight_scheme("uniform")
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    conn = get_connection(cfg.postgres)
    ok = 0
    failed = 0
    try:
        dates = _filter_dates(list_snapshot_dates(conn), start_date, end_date, args.limit)
        print(f"Backfill Merton uniform sur {len(dates)} snapshot(s)")
        for idx, snap_date in enumerate(dates, start=1):
            t0 = time.perf_counter()
            try:
                raw = load_snapshot(snap_date, conn=conn)
                panel = build_market_panel(raw, cfg)
                result = calibrate_global(
                    panel,
                    cfg,
                    weight_fn=scheme.weight_fn,
                    weight_scheme=scheme.scheme_id,
                )
                total_runtime_s = time.perf_counter() - t0
                with conn.cursor() as cur:
                    _replace_run(
                        cur,
                        snapshot_date=snap_date,
                        result=result,
                        total_runtime_s=total_runtime_s,
                        optimizer_local=cfg.calibration.optimizer,
                    )
                conn.commit()
                ok += 1
                print(
                    f"[{idx}/{len(dates)}] {snap_date} ok "
                    f"rmse={result.rmse_iv * 100:.3f}pts "
                    f"time={result.calibration_time_s:.2f}s "
                    f"mu={result.params.mu_jump:.4f}"
                )
            except Exception as exc:
                conn.rollback()
                failed += 1
                print(f"[{idx}/{len(dates)}] {snap_date} failed: {exc}")
        print(f"Termine: {ok} succes, {failed} echec(s)")
        return 0 if failed == 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
