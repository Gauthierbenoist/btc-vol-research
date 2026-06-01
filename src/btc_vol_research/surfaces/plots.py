"""Visualisations smiles et surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

from btc_vol_research.surfaces.smile import iter_slices
from btc_vol_research.surfaces.surface import build_iv_surface_grid

if TYPE_CHECKING:
    from btc_vol_research.models.svi.calibrate import SVICalibrationResult


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


def plot_svi_surface(
    results: "list[SVICalibrationResult]",
    panel: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
) -> tuple[Path, Path]:
    """
    Surface 3D + carte de chaleur (contour) de la vol SVI calibrée.

    Returns:
        (path_3d, path_contour)
    """
    from btc_vol_research.surfaces.svi_surface import build_svi_surface_grid

    out_dir.mkdir(parents=True, exist_ok=True)
    lm_grid, T_grid, iv_matrix = build_svi_surface_grid(
        results,
        panel,
        n_moneyness=n_moneyness,
        n_maturities=n_maturities,
    )

    # 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        lm_grid,
        T_grid,
        iv_matrix * 100,
        cmap="plasma",
        edgecolor="none",
        alpha=0.92,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, label="IV (%)")
    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("T (années)")
    ax.set_zlabel("IV (%)")
    ax.set_title(f"Surface SVI — {snapshot_date}")
    path_3d = out_dir / f"svi_surface_3d_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path_3d, dpi=120)
    plt.close(fig)

    # Contour (souvent plus lisible)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    cf = ax2.contourf(lm_grid, T_grid, iv_matrix * 100, levels=25, cmap="plasma")
    fig2.colorbar(cf, ax=ax2, label="IV (%)")
    ax2.set_xlabel("log(K/F)")
    ax2.set_ylabel("T (années)")
    ax2.set_title(f"Surface SVI (vue contour) — {snapshot_date}")
    path_contour = out_dir / f"svi_surface_contour_{snapshot_date}.png"
    fig2.tight_layout()
    fig2.savefig(path_contour, dpi=120)
    plt.close(fig2)

    return path_3d, path_contour


def plot_calibration_fit(
    slice_df: pd.DataFrame,
    model_iv: np.ndarray,
    out_dir: Path,
    snapshot_date: str,
    slice_id: str,
    *,
    model_name: str = "Heston",
    file_prefix: str | None = None,
    smooth_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = file_prefix or model_name.lower()
    order = np.argsort(slice_df["log_moneyness"].values)
    lm = slice_df["log_moneyness"].values[order]
    miv = np.asarray(model_iv)[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(slice_df["log_moneyness"], slice_df["iv_used"] * 100, label="Marché", s=20)
    if smooth_curve is not None:
        k_fine, iv_fine = smooth_curve
        ax.plot(k_fine, iv_fine * 100, "r-", lw=2, label=f"{model_name} calibré")
    else:
        ax.plot(lm, miv * 100, "r-", lw=2, label=f"{model_name} calibré")
    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("IV (%)")
    ax.set_title(f"Calibration {model_name} — {snapshot_date} — {slice_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{prefix}_fit_{snapshot_date}_{slice_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
