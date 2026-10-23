"""
Publication-ready (300 DPI) visualizations for PCA and Factor Analysis of GEO datasets.

Generates:
1. Scree Plot: Bar chart of explained variance per PC + cumulative variance line + 70% threshold.
2. 2D PCA Biplot (PC1 vs PC2): Sample projection colored by target phenotype, 95% confidence
   ellipses for each phenotype group, and top gene loading vectors (arrows).
3. 3D PCA Scatter Plot (PC1 vs PC2 vs PC3): 3D projection colored by phenotype.
4. Factor Loadings Plot: Heatmap across PC1-PC3 and horizontal bar plots of top positive
   and negative driver markers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

logger = logging.getLogger(__name__)

# Publication-ready visual styling
PALETTE_CATEGORICAL = [
    "#1B4F72",  # Deep Navy
    "#C0392B",  # Crimson Red
    "#117864",  # Teal Green
    "#D68910",  # Warm Amber
    "#6C3483",  # Royal Purple
    "#2E86C1",  # Cerulean
    "#A04000",  # Rust
    "#4D5656",  # Slate
]


def _configure_matplotlib_style() -> None:
    """Apply clean publication-grade typography and layout parameters."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _draw_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    confidence: float = 0.95,
    facecolor: str = "none",
    edgecolor: str = "black",
    alpha: float = 0.2,
    lw: float = 1.8,
) -> Optional[Ellipse]:
    """
    Draw a 2D covariance confidence ellipse (default 95% confidence via chi-square df=2).
    Requires at least 3 data points with non-singular covariance.
    """
    if len(x) < 3 or len(y) < 3:
        return None

    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or cov[0, 0] <= 1e-12 or cov[1, 1] <= 1e-12:
        return None

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = np.clip(pearson, -1.0, 1.0)

    ell_radius_x = np.sqrt(1.0 + pearson)
    ell_radius_y = np.sqrt(1.0 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2.0,
        height=ell_radius_y * 2.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=lw,
        zorder=2,
    )

    scale_factor = np.sqrt(chi2.ppf(confidence, df=2))
    scale_x = np.sqrt(cov[0, 0]) * scale_factor
    scale_y = np.sqrt(cov[1, 1]) * scale_factor
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    # Add solid outline for clarity
    outline = Ellipse(
        (0, 0),
        width=ell_radius_x * 2.0,
        height=ell_radius_y * 2.0,
        facecolor="none",
        edgecolor=edgecolor,
        alpha=0.85,
        linewidth=lw,
        linestyle="-",
        zorder=3,
    )
    outline.set_transform(transf + ax.transData)
    ax.add_patch(outline)
    return ellipse


def plot_scree(
    variance_df: pd.DataFrame,
    output_path: Union[str, Path],
    variance_threshold: float = 0.70,
    dpi: int = 300,
) -> Path:
    """
    Generate publication-ready Scree Plot showing individual and cumulative explained variance.
    """
    _configure_matplotlib_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))

    components = variance_df["component"].tolist()
    indiv_pct = variance_df["explained_variance_pct"].values
    cum_pct = variance_df["cumulative_variance_pct"].values
    thresh_pct = variance_threshold * 100.0
    x_pos = np.arange(len(components))

    bars = ax1.bar(
        x_pos,
        indiv_pct,
        width=0.55,
        color="#2874A6",
        edgecolor="#1B4F72",
        alpha=0.88,
        label="Individual Explained Variance (%)",
        zorder=3,
    )

    ax1.plot(
        x_pos,
        cum_pct,
        color="#C0392B",
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Cumulative Explained Variance (%)",
        zorder=4,
    )

    ax1.axhline(
        y=thresh_pct,
        color="#117864",
        linestyle="--",
        linewidth=1.8,
        label=f"Target Variance Threshold ({thresh_pct:.0f}%)",
        zorder=2,
    )

    # Annotate bar values and cumulative curve points
    for idx, (bar, c_val, i_val) in enumerate(zip(bars, cum_pct, indiv_pct)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            i_val + 1.2,
            f"{i_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#1B4F72",
            fontweight="bold",
        )
        if idx > 0:
            ax1.text(
                x_pos[idx],
                c_val + 2.0,
                f"{c_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#922B21",
                fontweight="bold",
            )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components)
    ax1.set_ylim(0, max(108.0, float(np.max(cum_pct)) + 12.0))
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree Plot & Cumulative Explained Variance")
    ax1.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Scree Plot to %s", out_file)
    return out_file


def plot_pca_biplot_2d(
    scores_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    loadings_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    phenotype_col: Optional[str],
    output_path: Union[str, Path],
    n_loading_arrows: int = 8,
    confidence: float = 0.95,
    dpi: int = 300,
) -> Path:
    """
    Generate a 2D PCA Biplot (PC1 vs PC2) with 95% confidence ellipses per phenotype
    group and top gene loading vectors.
    """
    _configure_matplotlib_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if scores_df.shape[1] < 2:
        raise ValueError("At least 2 Principal Components are required for a 2D Biplot.")

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    pc1_pct = float(variance_df.loc[variance_df["component"] == "PC1", "explained_variance_pct"].iloc[0])
    pc2_pct = float(variance_df.loc[variance_df["component"] == "PC2", "explained_variance_pct"].iloc[0])

    x_scores = scores_df["PC1"].values
    y_scores = scores_df["PC2"].values

    if phenotype_col and phenotype_col in metadata_df.columns:
        pheno = metadata_df[phenotype_col].reindex(scores_df.index).fillna("Unknown").astype(str)
        unique_groups = sorted(pheno.unique())
        color_map = {
            grp: PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)]
            for i, grp in enumerate(unique_groups)
        }

        for grp in unique_groups:
            mask = (pheno == grp).values
            gx = x_scores[mask]
            gy = y_scores[mask]
            col = color_map[grp]
            ax.scatter(
                gx,
                gy,
                c=col,
                s=52,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.6,
                label=f"{grp} (n={len(gx)})",
                zorder=4,
            )
            if grp.lower() != "unknown" and len(gx) >= 3:
                _draw_confidence_ellipse(
                    gx,
                    gy,
                    ax,
                    confidence=confidence,
                    facecolor=col,
                    edgecolor=col,
                    alpha=0.14,
                )
        legend_title = f"{phenotype_col} ({int(confidence * 100)}% CI)"
    else:
        ax.scatter(
            x_scores,
            y_scores,
            c="#2874A6",
            s=52,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.6,
            label=f"Samples (n={len(x_scores)})",
            zorder=4,
        )
        legend_title = "Cohort"

    # Overlay top gene loading arrows
    if n_loading_arrows > 0 and loadings_matrix.shape[1] >= 2:
        l_pc1 = loadings_matrix["PC1"]
        l_pc2 = loadings_matrix["PC2"]
        vector_lengths = np.sqrt(l_pc1**2 + l_pc2**2)
        top_genes = vector_lengths.sort_values(ascending=False).head(n_loading_arrows).index.tolist()

        # Scale arrows to 75% of the score radius so they are clearly visible
        score_radius = max(np.max(np.abs(x_scores)), np.max(np.abs(y_scores)))
        max_vec = float(vector_lengths.loc[top_genes].max()) if top_genes else 1.0
        arrow_scale = (0.72 * score_radius / max_vec) if max_vec > 0 else 1.0

        for idx, gene in enumerate(top_genes):
            vx = float(l_pc1.loc[gene]) * arrow_scale
            vy = float(l_pc2.loc[gene]) * arrow_scale
            ax.annotate(
                "",
                xy=(vx, vy),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#2C3E50",
                    lw=1.4,
                    alpha=0.78,
                    mutation_scale=11,
                ),
                zorder=5,
            )
            # Offset label slightly past arrow tip
            offset_factor = 1.08 + (0.04 * (idx % 2))
            ax.text(
                vx * offset_factor,
                vy * offset_factor,
                gene,
                fontsize=8.5,
                fontweight="bold",
                color="#1C2833",
                ha="center" if abs(vx) < 0.1 * score_radius else ("left" if vx >= 0 else "right"),
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=6,
            )

    ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#7F8C8D", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% Explained Variance)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% Explained Variance)")
    title_suffix = f" by {phenotype_col}" if phenotype_col else ""
    ax.set_title(f"2D PCA Biplot (PC1 vs PC2){title_suffix}")
    ax.legend(title=legend_title, loc="best", frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 2D PCA Biplot to %s", out_file)
    return out_file


def plot_pca_scatter_3d(
    scores_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    phenotype_col: Optional[str],
    output_path: Union[str, Path],
    dpi: int = 300,
) -> Optional[Path]:
    """
    Generate a 3D PCA Scatter Plot (PC1 vs PC2 vs PC3) colored by the selected phenotype.
    """
    if scores_df.shape[1] < 3:
        logger.warning("Fewer than 3 principal components available; skipping 3D PCA plot.")
        return None

    _configure_matplotlib_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    pc1_pct = float(variance_df.loc[variance_df["component"] == "PC1", "explained_variance_pct"].iloc[0])
    pc2_pct = float(variance_df.loc[variance_df["component"] == "PC2", "explained_variance_pct"].iloc[0])
    pc3_pct = float(variance_df.loc[variance_df["component"] == "PC3", "explained_variance_pct"].iloc[0])

    x = scores_df["PC1"].values
    y = scores_df["PC2"].values
    z = scores_df["PC3"].values

    if phenotype_col and phenotype_col in metadata_df.columns:
        pheno = metadata_df[phenotype_col].reindex(scores_df.index).fillna("Unknown").astype(str)
        unique_groups = sorted(pheno.unique())
        for i, grp in enumerate(unique_groups):
            mask = (pheno == grp).values
            col = PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)]
            ax.scatter(
                x[mask],
                y[mask],
                z[mask],
                c=col,
                s=46,
                alpha=0.86,
                edgecolors="white",
                linewidth=0.5,
                label=f"{grp} (n={int(mask.sum())})",
            )
        legend_title = phenotype_col
    else:
        ax.scatter(x, y, z, c="#2874A6", s=46, alpha=0.86, label=f"Samples (n={len(x)})")
        legend_title = "Cohort"

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)", labelpad=8)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)", labelpad=8)
    ax.set_zlabel(f"PC3 ({pc3_pct:.1f}%)", labelpad=8)
    title_suffix = f" — Colored by {phenotype_col}" if phenotype_col else ""
    ax.set_title(f"3D PCA Projection (PC1 vs PC2 vs PC3){title_suffix}", pad=14)
    ax.view_init(elev=24, azim=132)
    ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(0.0, 0.95), frameon=True)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 3D PCA Scatter Plot to %s", out_file)
    return out_file


def plot_loadings_heatmap_and_bars(
    loadings_matrix: pd.DataFrame,
    top_drivers_df: pd.DataFrame,
    output_dir: Union[str, Path],
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Generate:
    1. A clustered-style Loadings Heatmap of top driver genes across PC1-PC3.
    2. A multi-panel horizontal Barplot of top positive and negative markers for PC1-PC3.
    """
    _configure_matplotlib_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcs = [c for c in ["PC1", "PC2", "PC3"] if c in loadings_matrix.columns]

    # 1. Heatmap of union of top-5 positive & negative markers per component
    sub_drivers = top_drivers_df[
        (top_drivers_df["component"].isin(pcs)) & (top_drivers_df["rank"] <= 5)
    ]
    union_genes = list(dict.fromkeys(sub_drivers["gene_id"].tolist()))
    heatmap_data = loadings_matrix.loc[union_genes, pcs]

    heatmap_path = out_dir / "pca_loadings_heatmap.png"
    fig_h = max(6.0, 0.28 * len(union_genes) + 1.8)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    max_abs = float(np.max(np.abs(heatmap_data.values))) if not heatmap_data.empty else 1.0

    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0.0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        cbar_kws={"label": "Factor Loading"},
        ax=ax,
    )
    ax.set_title("Factor Loadings Heatmap (Top Driver Markers)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Gene / Marker")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Loadings Heatmap to %s", heatmap_path)

    # 2. Multi-panel Horizontal Barplot for PC1, PC2, PC3 (top 10 pos + top 10 neg)
    barplot_path = out_dir / "pca_loadings_barplot.png"
    n_panels = len(pcs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 7.5), sharex=False)
    if n_panels == 1:
        axes = [axes]

    for ax, pc in zip(axes, pcs):
        pc_sub = top_drivers_df[top_drivers_df["component"] == pc].copy()
        pc_sub = pc_sub.sort_values(by="loading", ascending=True)

        colors = ["#C0392B" if val > 0 else "#2874A6" for val in pc_sub["loading"]]
        y_pos = np.arange(len(pc_sub))
        ax.barh(y_pos, pc_sub["loading"], color=colors, edgecolor="none", height=0.72, alpha=0.88)
        ax.axvline(0, color="#2C3E50", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pc_sub["gene_id"], fontsize=8.5)
        ax.set_xlabel("Factor Loading")
        ax.set_title(f"{pc} Top ±10 Drivers")

    fig.suptitle("Top Positive and Negative Gene Drivers per Principal Component", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(barplot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Loadings Barplot to %s", barplot_path)

    return heatmap_path, barplot_path