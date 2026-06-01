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


def plot_svi_rho_term_structure(
    results: "list[SVICalibrationResult]",
    out_dir: Path,
    snapshot_date: str,
) -> Path:
    """
    Trace ρ(T) — paramètre de corrélation/skew SVI par maturité calibrée.
    """
    from btc_vol_research.analysis.svi_metrics import svi_term_structure_table

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = svi_term_structure_table(results)
    T = ts["T_years"].values
    rho = ts["rho"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, rho, "o-", color="steelblue", lw=2, markersize=8, label=r"$\rho(T)$ SVI")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    for _, row in ts.iterrows():
        ax.annotate(
            row["slice_id"],
            (row["T_years"], row["rho"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            alpha=0.85,
        )
    ax.set_xlabel("Maturité T (années)")
    ax.set_ylabel(r"$\rho$ (paramètre SVI)")
    ax.set_title(f"Structure terme du skew SVI — ρ(T) — {snapshot_date}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out_dir / f"svi_rho_term_{snapshot_date}.png"
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


def plot_mark_vs_mid(
    panel: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
    atm_half_width: float = 0.10,
) -> Path:
    """mark_iv vs iv_mid (scatter + écart) coloré par zone."""
    from btc_vol_research.analysis.zones import ZONE_ATM, ZONE_LEFT, ZONE_RIGHT, assign_moneyness_zone, zone_label_fr

    out_dir.mkdir(parents=True, exist_ok=True)
    lm = panel["log_moneyness"].values
    zones = assign_moneyness_zone(lm, atm_half_width)
    mark = panel["iv_mark"].values * 100
    mid = panel["iv_mid"].values * 100
    colors = {ZONE_LEFT: "C0", ZONE_ATM: "C2", ZONE_RIGHT: "C3"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for z in (ZONE_LEFT, ZONE_ATM, ZONE_RIGHT):
        m = zones == z
        if m.sum() == 0:
            continue
        ax.scatter(mark[m], mid[m], s=14, alpha=0.55, c=colors[z], label=zone_label_fr(z))
    lim_lo = min(np.nanmin(mark), np.nanmin(mid))
    lim_hi = max(np.nanmax(mark), np.nanmax(mid))
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, label="mark = mid")
    ax.set_xlabel("mark_iv (%)")
    ax.set_ylabel("iv_mid inversion BS (%)")
    ax.set_title(f"mark_iv vs iv_mid — {snapshot_date}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    diff = mark - mid
    for z in (ZONE_LEFT, ZONE_ATM, ZONE_RIGHT):
        m = zones == z
        if m.sum() == 0:
            continue
        ax2.scatter(lm[m], diff[m], s=14, alpha=0.55, c=colors[z], label=zone_label_fr(z))
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(-atm_half_width, color="gray", ls=":", lw=0.7)
    ax2.axvline(atm_half_width, color="gray", ls=":", lw=0.7)
    ax2.set_xlabel("log(K/F)")
    ax2.set_ylabel("mark_iv − iv_mid (pts vol)")
    ax2.set_title("Écart mark − mid par moneyness")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    path = out_dir / f"mark_vs_mid_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_svi_rmse_by_zone(
    rmse_df: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
) -> Path:
    """Barres RMSE SVI par zone (groupées par maturité)."""
    from btc_vol_research.analysis.zones import ZONE_ORDER, zone_label_fr

    out_dir.mkdir(parents=True, exist_ok=True)
    zones = [z for z in ZONE_ORDER if z in rmse_df["zone"].values]
    slices = sorted(rmse_df["slice_id"].unique())
    x = np.arange(len(slices))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(slices) * 0.9), 5))

    for i, zone in enumerate(zones):
        vals = []
        for sid in slices:
            row = rmse_df[(rmse_df["slice_id"] == sid) & (rmse_df["zone"] == zone)]
            vals.append(float(row["rmse_svi"].iloc[0] * 100) if len(row) else 0.0)
        ax.bar(x + (i - 1) * width, vals, width=width, label=zone_label_fr(zone))

    ax.set_xticks(x)
    ax.set_xticklabels(slices, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE SVI vs marché (pts vol)")
    ax.set_title(f"RMSE SVI par zone — {snapshot_date}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = out_dir / f"svi_rmse_by_zone_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_svi_rmse_zones_heatmap(
    rmse_df: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
) -> Path:
    """Heatmap maturité × zone (RMSE %)."""
    from btc_vol_research.analysis.zones import ZONE_ORDER, zone_label_fr

    out_dir.mkdir(parents=True, exist_ok=True)
    pivot = rmse_df.pivot(index="slice_id", columns="zone", values="rmse_svi") * 100
    pivot = pivot[[c for c in ZONE_ORDER if c in pivot.columns]]
    pivot.columns = [zone_label_fr(c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="RMSE (pts vol)")
    ax.set_title(f"RMSE SVI par zone et maturité — {snapshot_date}")
    path = out_dir / f"svi_rmse_heatmap_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
