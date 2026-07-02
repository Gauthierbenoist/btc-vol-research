#!/usr/bin/env python
"""Compare calibrations SVI : pondération v1 vs v2."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from btc_vol_research.config import load_config  # noqa: E402
from btc_vol_research.data.loader import load_snapshot  # noqa: E402
from btc_vol_research.data.panel import build_market_panel  # noqa: E402
from btc_vol_research.models.calibration_weights import calibration_weights, calibration_weights_v2  # noqa: E402
from btc_vol_research.models.svi.calibrate import calibrate_all_slices  # noqa: E402
from btc_vol_research.analysis.iv_diagnostics import svi_rmse_by_zone  # noqa: E402

WEIGHT_SCHEMES = [
    ("w_v1_vega_sqrt_oi", calibration_weights, "vega * sqrt(OI)"),
    ("w_v2_vega_sqrt_1plus", calibration_weights_v2, "vega * sqrt(1+OI) * sqrt(1+volume)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SVI calibré avec pondération v1 vs v2")
    p.add_argument("--date", type=str, default="2026-06-01", help="YYYY-MM-DD")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    snap = date.fromisoformat(args.date)
    cfg = load_config()
    panel = build_market_panel(load_snapshot(snap), cfg)
    atm_w = cfg.calibration.atm_zone_half_width
    snap_str = snap.isoformat()

    rows_global = []
    rows_zone = []

    for name, w_fn, label in WEIGHT_SCHEMES:
        results = calibrate_all_slices(panel, cfg, weight_fn=w_fn)
        if not results:
            print(f"Aucun résultat pour {name}", file=sys.stderr)
            continue

        rmse_g = float(np.sqrt(np.mean([r.rmse_iv**2 for r in results])))
        w_rmse_g = float(np.mean([r.weighted_rmse_iv for r in results]))
        rows_global.append(
            {
                "scheme": name,
                "formula": label,
                "rmse_global_pct": rmse_g * 100,
                "weighted_rmse_pct": w_rmse_g * 100,
                "n_slices": len(results),
            }
        )

        zdf = svi_rmse_by_zone(results, atm_w)
        for zone, g in zdf.groupby("zone"):
            rows_zone.append(
                {
                    "scheme": name,
                    "formula": label,
                    "zone": zone,
                    "rmse_pct": float(g["rmse_svi"].mean() * 100),
                    "mae_pct": float(g["mae_svi"].mean() * 100),
                    "bias_pts": float(g["bias_model_minus_mkt"].mean() * 100),
                }
            )

    global_df = pd.DataFrame(rows_global)
    zone_df = pd.DataFrame(rows_zone)
    out_dir = cfg.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    global_path = out_dir / f"svi_weights_v1_v2_global_{snap_str}.csv"
    zone_path = out_dir / f"svi_weights_v1_v2_by_zone_{snap_str}.csv"
    global_df.to_csv(global_path, index=False)
    zone_df.to_csv(zone_path, index=False)

    print(f"Snapshot {snap_str} — comparaison pondérations SVI\n")
    print("=== RMSE / RMSE pondéré global ===")
    print(global_df.to_string(index=False))
    print("\n=== RMSE par zone (moyenne maturités, pts vol) ===")
    print(zone_df.pivot(index="zone", columns="scheme", values="rmse_pct").to_string())
    print("\n=== MAE par zone (moyenne maturités, pts vol) ===")
    print(zone_df.pivot(index="zone", columns="scheme", values="mae_pct").to_string())
    print(f"\nCSV : {global_path.name}, {zone_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
