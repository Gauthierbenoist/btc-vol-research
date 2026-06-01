"""Visualisations smiles et surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btc_vol_research.surfaces.smile import iter_slices
from btc_vol_research.surfaces.surface import build_iv_surface_grid


def plot_smiles(
    panel: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
    *,
    iv_col: str = "iv_used",
    compare_mark: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sid, g in iter_slices(panel):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(g["log_moneyness"], g[iv_col] * 100, s=12, alpha=0.7, label="IV utilisée")
        if compare_mark and "iv_mark" in g.columns:
            ax.scatter(
                g["log_moneyness"],
                g["iv_mark"] * 100,
                s=10,
                alpha=0.4,
                marker="x",
                label="mark_iv Deribit",
            )
        ax.set_xlabel("log(K/F)")
        ax.set_ylabel("Vol implicite (%)")
        ax.set_title(f"Smile BTC — {snapshot_date} — échéance {sid}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"smile_{snapshot_date}_{sid}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_iv_surface(
    panel: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_grid, T_grid, iv_matrix = build_iv_surface_grid(panel)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(lm_grid, T_grid, iv_matrix * 100, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("T (années)")
    ax.set_zlabel("IV (%)")
    ax.set_title(f"Surface IV implicite — {snapshot_date}")
    path = out_dir / f"surface_iv_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration_fit(
    slice_df: pd.DataFrame,
    model_iv: np.ndarray,
    out_dir: Path,
    snapshot_date: str,
    slice_id: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(slice_df["log_moneyness"], slice_df["iv_used"] * 100, label="Marché", s=20)
    ax.plot(
        slice_df["log_moneyness"],
        model_iv * 100,
        "r-",
        lw=2,
        label="Heston calibré",
    )
    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("IV (%)")
    ax.set_title(f"Calibration Heston — {snapshot_date} — {slice_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / f"heston_fit_{snapshot_date}_{slice_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
