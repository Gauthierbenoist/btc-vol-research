#!/usr/bin/env python
"""Compare RMSE SVI : gaussienne ATM pleine / réduite / absente."""

from __future__ import annotations

import sys
from dataclasses import replace
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
from btc_vol_research.models.svi.calibrate import calibrate_all_slices  # noqa: E402
from btc_vol_research.analysis.iv_diagnostics import svi_rmse_by_zone  # noqa: E402
from btc_vol_research.models import calibration_weights as cw  # noqa: E402
from btc_vol_research.models.svi import calibrate as svi_calibrate  # noqa: E402

SCHEMES = [
    ("gauss_pleine", 0.15, 1.0),
    ("gauss_reduite", 0.25, 0.35),
    ("sans_gauss", None, 0.0),
]

_orig_weights = cw.calibration_weights


def _weights_with_atm(slice_df, cfg, r, q, sigma, strength):
    w = _orig_weights(slice_df, cfg, r, q)
    if sigma is None or strength <= 0:
        return w
    lm = slice_df["log_moneyness"].values
    atm_w = np.exp(-0.5 * (lm / sigma) ** 2)
    factor = (1.0 - strength) + strength * atm_w
    w = w * factor
    return w / (w.sum() + 1e-12) * len(w)


def main() -> int:
    snap = date(2026, 6, 1)
    cfg = load_config()
    panel = build_market_panel(load_snapshot(snap), cfg)
    atm_w = cfg.calibration.atm_zone_half_width

    rows_global = []
    rows_zone = []

    for name, sigma, strength in SCHEMES:
        svi_calibrate.calibration_weights = lambda sdf, c, r, q, s=sigma, st=strength: _weights_with_atm(
            sdf, c, r, q, s, st
        )
        results = calibrate_all_slices(panel, cfg)
        svi_calibrate.calibration_weights = _orig_weights

        rmse_g = float(np.sqrt(np.mean([r.rmse_iv**2 for r in results])))
        rows_global.append({"scheme": name, "rmse_global_pct": rmse_g * 100, "n_slices": len(results)})

        zdf = svi_rmse_by_zone(results, atm_w)
        for zone, g in zdf.groupby("zone"):
            rows_zone.append(
                {
                    "scheme": name,
                    "zone": zone,
                    "rmse_pct": float(g["rmse_svi"].mean() * 100),
                    "bias_pts": float(g["bias_model_minus_mkt"].mean() * 100),
                }
            )

    global_df = pd.DataFrame(rows_global)
    zone_df = pd.DataFrame(rows_zone)
    out = cfg.reports_dir / "svi_weight_scheme_compare_2026-06-01.csv"
    zone_df.to_csv(out, index=False)

    print("=== RMSE global SVI (2026-06-01) ===")
    print(global_df.to_string(index=False))
    print("\n=== RMSE par zone (pts vol) ===")
    pivot = zone_df.pivot(index="zone", columns="scheme", values="rmse_pct")
    print(pivot.to_string())
    print(f"\nCSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
