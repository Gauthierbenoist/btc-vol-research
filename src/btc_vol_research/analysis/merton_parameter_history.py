"""Historique temporel des parametres Merton depuis la base SQL."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from btc_vol_research.config import AppConfig
from btc_vol_research.data.loader import get_connection


def load_merton_parameter_history(
    cfg: AppConfig,
    *,
    weight_scheme: str = "uniform",
) -> pd.DataFrame:
    """Charge l'historique global Merton depuis `calibration_runs`."""
    query = """
        select
            snapshot_date,
            sigma,
            lambda_jump,
            mu_jump,
            sigma_jump,
            rmse_iv,
            calibration_time_s,
            success
        from calibration_runs
        where model_name = 'merton'
          and weight_scheme = %s
          and scope = 'global'
          and slice_id is null
        order by snapshot_date
    """
    conn = get_connection(cfg.postgres)
    try:
        df = pd.read_sql(query, conn, params=(weight_scheme,))
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError(f"Aucune calibration Merton trouvee pour weight_scheme={weight_scheme!r}")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def plot_merton_parameter_history(
    history: pd.DataFrame,
    out_dir: Path,
    *,
    weight_scheme: str = "uniform",
) -> Path:
    """
    Affiche sur une meme figure l'evolution temporelle des parametres Merton.

    `lambda_jump` est place sur un axe secondaire pour rester lisible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_right = ax_left.twinx()

    x = history["snapshot_date"]
    line_kw = {"linewidth": 1.8, "marker": "o", "markersize": 4}

    l1 = ax_left.plot(x, history["sigma"], label="sigma", color="tab:blue", **line_kw)
    l2 = ax_left.plot(x, history["mu_jump"], label="mu_jump", color="tab:red", **line_kw)
    l3 = ax_left.plot(x, history["sigma_jump"], label="sigma_jump", color="tab:green", **line_kw)
    l4 = ax_right.plot(x, history["lambda_jump"], label="lambda_jump", color="tab:orange", **line_kw)

    failed = history.loc[~history["success"].fillna(False)]
    if not failed.empty:
        ax_left.scatter(
            failed["snapshot_date"],
            failed["sigma"],
            color="black",
            marker="x",
            s=40,
            label="run non converge",
            zorder=5,
        )

    ax_left.set_title(f"Evolution temporelle des parametres Merton ({weight_scheme})")
    ax_left.set_xlabel("snapshot_date")
    ax_left.set_ylabel("sigma / mu_jump / sigma_jump")
    ax_right.set_ylabel("lambda_jump")
    ax_left.grid(True, alpha=0.3)

    ax_left.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    if not failed.empty:
        failed_handle = ax_left.collections[-1]
        lines.append(failed_handle)
        labels.append("run non converge")
    ax_left.legend(lines, labels, loc="best")

    path = out_dir / f"merton_parameter_history_{weight_scheme}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
