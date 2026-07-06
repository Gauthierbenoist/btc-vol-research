"""Visualisations smiles et surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btc_vol_research.calibration.results import SliceCalibrationResult

def _merton_plotly_scene(*, z_title: str) -> dict:
    """Échelles automatiques (Plotly auto-range sur les données de chaque snapshot)."""
    return {
        "xaxis": {"title": "log(K/F)"},
        "yaxis_title": "T (années)",
        "zaxis": {"title": z_title},
    }


def plot_svi_surface(
    results: "list[SliceCalibrationResult]",
    panel: pd.DataFrame,
    out_dir: Path,
    snapshot_date: str,
    *,
    n_moneyness: int = 50,
    n_maturities: int = 30,
    file_stem: str = "svi_surface",
    title_suffix: str = "",
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
    ax.set_title(f"Surface SVI{title_suffix} — {snapshot_date}")
    path_3d = out_dir / f"{file_stem}_3d_{snapshot_date}.png"
    fig.tight_layout()
    fig.savefig(path_3d, dpi=120)
    plt.close(fig)

    # Contour (souvent plus lisible)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    cf = ax2.contourf(lm_grid, T_grid, iv_matrix * 100, levels=25, cmap="plasma")
    fig2.colorbar(cf, ax=ax2, label="IV (%)")
    ax2.set_xlabel("log(K/F)")
    ax2.set_ylabel("T (années)")
    ax2.set_title(f"Surface SVI{title_suffix} (vue contour) — {snapshot_date}")
    path_contour = out_dir / f"{file_stem}_contour_{snapshot_date}.png"
    fig2.tight_layout()
    fig2.savefig(path_contour, dpi=120)
    plt.close(fig2)

    return path_3d, path_contour


def plot_svi_rho_term_structure(
    results: "list[SliceCalibrationResult]",
    out_dir: Path,
    snapshot_date: str,
    *,
    file_stem: str = "svi_rho_term",
    title_suffix: str = "",
) -> Path:
    """
    Trace ρ(T) — paramètre de corrélation/skew SVI par maturité calibrée.
    """
    from btc_vol_research.analysis.tables import svi_term_structure_table

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
    ax.set_title(f"Structure terme du skew SVI{title_suffix} — rho(T) — {snapshot_date}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out_dir / f"{file_stem}_{snapshot_date}.png"
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


def plot_merton_surface_plotly(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_matrix: np.ndarray,
    out_dir: Path,
    snapshot_date: str,
    *,
    file_stem: str = "merton",
    title_suffix: str = "",
) -> Path:
    """Surface 3D Merton en Plotly (HTML interactif)."""
    import plotly.graph_objects as go

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(
        data=[
            go.Surface(
                x=lm_grid,
                y=t_grid,
                z=iv_matrix * 100.0,
                colorscale="Plasma",
                colorbar={"title": "IV (%)"},
            )
        ]
    )
    fig.update_layout(
        title=f"Surface Merton{title_suffix} — {snapshot_date}",
        scene=_merton_plotly_scene(z_title="IV (%)"),
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    path = out_dir / f"{file_stem}_surface_3d_{snapshot_date}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def plot_merton_iv_and_abs_error_plotly(
    lm_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_matrix: np.ndarray,
    abs_err_matrix: np.ndarray,
    out_dir: Path,
    snapshot_date: str,
    *,
    rmse_pct: float,
    file_stem: str = "merton",
    title_suffix: str = "",
) -> Path:
    """IV modele + |erreur| cote a cote (Plotly) avec RMSE global."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("IV Merton", "|IV marche - IV modele|"),
        horizontal_spacing=0.05,
    )
    fig.add_trace(
        go.Surface(
            x=lm_grid,
            y=t_grid,
            z=iv_matrix * 100.0,
            colorscale="Plasma",
            colorbar={"title": "IV (%)", "len": 0.45, "x": 0.45},
            showscale=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Surface(
            x=lm_grid,
            y=t_grid,
            z=abs_err_matrix,
            colorscale="Reds",
            colorbar={"title": "|erreur| (pts vol)", "len": 0.45, "x": 1.0},
            showscale=True,
        ),
        row=1,
        col=2,
    )
    metrics = f"RMSE global = {rmse_pct:.3f} pts vol | MSE = RMSE^2 = {rmse_pct**2:.3f}"
    fig.update_layout(
        title={
            "text": (
                f"Surface Merton{title_suffix} et erreur absolue — {snapshot_date}"
                f"<br><sup>{metrics}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
        scene=_merton_plotly_scene(z_title="IV (%)"),
        scene2=_merton_plotly_scene(z_title="|erreur| (pts vol)"),
    )
    path = out_dir / f"{file_stem}_iv_and_abs_error_{snapshot_date}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


