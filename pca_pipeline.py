"""
Dimensionality reduction and Factor Analysis / PCA pipeline.

Computes Principal Components, evaluates explained & cumulative variance against
target biostatistical thresholds (e.g. >= 70%), extracts factor loadings, and
identifies top positive and negative driver genes per component.
Uses `sklearn.decomposition.PCA` when installed, with a mathematically identical
SVD fallback via `numpy.linalg.svd`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
except ImportError:

    class PCA:  # type: ignore[no-redef]
        """Mathematically identical SVD implementation of sklearn.decomposition.PCA."""

        def __init__(self, n_components: int = 5, random_state: Optional[int] = 42) -> None:
            self.n_components = n_components
            self.random_state = random_state
            self.components_: np.ndarray = np.empty((0, 0))
            self.explained_variance_: np.ndarray = np.empty(0)
            self.explained_variance_ratio_: np.ndarray = np.empty(0)
            self.mean_: np.ndarray = np.empty(0)

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            arr = np.asarray(X, dtype=float)
            n_samples, _ = arr.shape
            self.mean_ = np.mean(arr, axis=0)
            X_centered = arr - self.mean_

            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            # Deterministic sign flip matching sklearn's svd_flip
            max_abs_cols = np.argmax(np.abs(U), axis=0)
            signs = np.sign(U[max_abs_cols, range(U.shape[1])])
            signs[signs == 0] = 1.0
            U *= signs
            Vt *= signs[:, np.newaxis]

            explained_variance = (S**2) / max(1, n_samples - 1)
            total_var = float(np.sum(explained_variance))
            explained_variance_ratio = (
                explained_variance / total_var if total_var > 0 else np.zeros_like(explained_variance)
            )

            k = min(self.n_components, len(S))
            self.components_ = Vt[:k]
            self.explained_variance_ = explained_variance[:k]
            self.explained_variance_ratio_ = explained_variance_ratio[:k]

            return U[:, :k] * S[:k]


logger = logging.getLogger(__name__)


@dataclass
class PCAPipelineResult:
    """Container for PCA scores, variance summary, and factor loadings."""

    scores_df: pd.DataFrame
    variance_df: pd.DataFrame
    loadings_matrix: pd.DataFrame
    top_drivers_df: pd.DataFrame
    cumulative_variance_total: float
    meets_variance_threshold: bool
    variance_threshold: float


def run_pca_pipeline(
    scaled_expression: pd.DataFrame,
    n_components: int = 5,
    variance_threshold: float = 0.70,
    n_top_drivers: int = 10,
    n_driver_components: int = 3,
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42,
) -> PCAPipelineResult:
    """
    Fit PCA on standardized expression data and extract loadings and driver markers.

    Parameters
    ----------
    scaled_expression : pd.DataFrame
        Z-score standardized expression matrix (N samples x P genes).
    n_components : int, default=5
        Number of principal components to extract.
    variance_threshold : float, default=0.70
        Target cumulative explained variance ratio (e.g., 0.70 for 70%).
    n_top_drivers : int, default=10
        Number of top positive and top negative driver markers to extract per PC.
    n_driver_components : int, default=3
        Number of leading components (PC1..PCk) for driver summary extraction.
    output_dir : str or Path, optional
        If provided, exports `loadings_summary.csv`, `pca_scores.csv`, and
        `pca_variance_summary.csv` into this directory.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    PCAPipelineResult
    """
    n_samples, n_features = scaled_expression.shape
    max_valid_components = min(n_samples, n_features)
    actual_components = min(n_components, max_valid_components)

    if actual_components < 1:
        raise ValueError(
            f"Cannot run PCA on matrix of shape {scaled_expression.shape} with n_components={n_components}."
        )

    if actual_components < n_components:
        logger.warning(
            "Requested n_components=%d exceeds min(n_samples=%d, n_features=%d); clamping to %d.",
            n_components,
            n_samples,
            n_features,
            actual_components,
        )

    pca = PCA(n_components=actual_components, random_state=random_state)
    scores_array = pca.fit_transform(scaled_expression.values)

    pc_names = [f"PC{i}" for i in range(1, actual_components + 1)]
    scores_df = pd.DataFrame(
        scores_array,
        index=scaled_expression.index,
        columns=pc_names,
    )

    explained_ratio = pca.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    variance_df = pd.DataFrame(
        {
            "component": pc_names,
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": explained_ratio,
            "explained_variance_pct": explained_ratio * 100.0,
            "cumulative_variance_ratio": cumulative_ratio,
            "cumulative_variance_pct": cumulative_ratio * 100.0,
        }
    )

    total_cum_var = float(cumulative_ratio[-1])
    meets_threshold = total_cum_var >= variance_threshold

    if meets_threshold:
        logger.info(
            "PCA variance check PASSED: First %d components explain %.2f%% of variance (target >= %.1f%%).",
            actual_components,
            total_cum_var * 100.0,
            variance_threshold * 100.0,
        )
    else:
        logger.warning(
            "PCA variance check WARNING: First %d components explain %.2f%% of variance, "
            "which is below the target threshold of %.1f%%. Consider reducing --top-genes "
            "(e.g. --top-genes 30 to 100) or enabling --use-anova-filter to focus on high-signal markers.",
            actual_components,
            total_cum_var * 100.0,
            variance_threshold * 100.0,
        )

    # Compute Factor Loadings: eigenvectors scaled by sqrt(eigenvalues)
    eigenvectors = pca.components_.T  # shape: (P, K)
    factor_loadings = eigenvectors * np.sqrt(pca.explained_variance_)

    loadings_matrix = pd.DataFrame(
        factor_loadings,
        index=scaled_expression.columns,
        columns=pc_names,
    )
    loadings_matrix.index.name = "gene_id"

    # Extract Top Positive and Top Negative Drivers for the first `n_driver_components` PCs
    driver_records: List[dict] = []
    n_pcs_for_drivers = min(n_driver_components, actual_components)

    for pc_idx in range(n_pcs_for_drivers):
        pc_label = pc_names[pc_idx]
        pc_weights = pd.Series(eigenvectors[:, pc_idx], index=scaled_expression.columns)
        pc_loadings = loadings_matrix[pc_label]

        # Top positive drivers
        pos_genes = pc_loadings.sort_values(ascending=False).head(n_top_drivers)
        for rank, (gene, loading_val) in enumerate(pos_genes.items(), start=1):
            driver_records.append(
                {
                    "component": pc_label,
                    "direction": "positive",
                    "rank": rank,
                    "gene_id": gene,
                    "loading": float(loading_val),
                    "eigenvector_weight": float(pc_weights[gene]),
                    "abs_loading": float(abs(loading_val)),
                }
            )

        # Top negative drivers
        neg_genes = pc_loadings.sort_values(ascending=True).head(n_top_drivers)
        for rank, (gene, loading_val) in enumerate(neg_genes.items(), start=1):
            driver_records.append(
                {
                    "component": pc_label,
                    "direction": "negative",
                    "rank": rank,
                    "gene_id": gene,
                    "loading": float(loading_val),
                    "eigenvector_weight": float(pc_weights[gene]),
                    "abs_loading": float(abs(loading_val)),
                }
            )

    top_drivers_df = pd.DataFrame.from_records(driver_records)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        loadings_csv = out_path / "loadings_summary.csv"
        top_drivers_df.to_csv(loadings_csv, index=False)
        scores_df.to_csv(out_path / "pca_scores.csv")
        variance_df.to_csv(out_path / "pca_variance_summary.csv", index=False)
        loadings_matrix.to_csv(out_path / "full_loadings_matrix.csv")
        logger.info("Exported PCA loadings summary to %s", loadings_csv)

    return PCAPipelineResult(
        scores_df=scores_df,
        variance_df=variance_df,
        loadings_matrix=loadings_matrix,
        top_drivers_df=top_drivers_df,
        cumulative_variance_total=total_cum_var,
        meets_variance_threshold=meets_threshold,
        variance_threshold=variance_threshold,
    )