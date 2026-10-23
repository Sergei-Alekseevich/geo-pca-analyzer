"""
Preprocessing and marker selection pipeline for gene expression matrices.

Handles missing-value filtering (>10% missingness), median imputation, optional
log2 transformation diagnostics, variance-based marker filtering (MAD or SD),
supervised ANOVA F-test filtering across phenotype groups, and Z-score scaling.
Uses scikit-learn when installed, with a seamless NumPy/SciPy fallback so the
pipeline runs out-of-the-box in minimal environments as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.feature_selection import f_classif as _sk_f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

    class SimpleImputer:  # type: ignore[no-redef]
        """Compatible fallback implementation of sklearn.impute.SimpleImputer."""

        def __init__(self, strategy: str = "median") -> None:
            self.strategy = strategy
            self.statistics_: Optional[np.ndarray] = None

        def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "SimpleImputer":
            arr = np.asarray(X, dtype=float)
            if self.strategy == "median":
                self.statistics_ = np.nanmedian(arr, axis=0)
            elif self.strategy == "mean":
                self.statistics_ = np.nanmean(arr, axis=0)
            else:
                raise ValueError(f"Unsupported imputation strategy: {self.strategy}")
            return self

        def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.statistics_ is None:
                raise RuntimeError("SimpleImputer must be fitted before transform.")
            arr = np.asarray(X, dtype=float).copy()
            nan_mask = np.isnan(arr)
            if np.any(nan_mask):
                arr[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
            return arr

        def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            return self.fit(X).transform(X)

    class StandardScaler:  # type: ignore[no-redef]
        """Compatible fallback implementation of sklearn.preprocessing.StandardScaler."""

        def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
            self.with_mean = with_mean
            self.with_std = with_std
            self.mean_: Optional[np.ndarray] = None
            self.scale_: Optional[np.ndarray] = None

        def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "StandardScaler":
            arr = np.asarray(X, dtype=float)
            self.mean_ = np.mean(arr, axis=0) if self.with_mean else np.zeros(arr.shape[1])
            std = np.std(arr, axis=0, ddof=0) if self.with_std else np.ones(arr.shape[1])
            std[std == 0.0] = 1.0
            self.scale_ = std
            return self

        def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler must be fitted before transform.")
            arr = np.asarray(X, dtype=float)
            return (arr - self.mean_) / self.scale_

        def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
            return self.fit(X).transform(X)


def _compute_anova_f_classif(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-way ANOVA F-value and p-value for each gene across phenotype groups."""
    if _HAS_SKLEARN:
        f_vals, p_vals = _sk_f_classif(X, y)
        return np.asarray(f_vals, dtype=float), np.asarray(p_vals, dtype=float)

    classes = y.unique()
    groups = [X.loc[y == cls].values for cls in classes]
    f_vals, p_vals = stats.f_oneway(*groups, axis=0)
    return np.asarray(f_vals, dtype=float), np.asarray(p_vals, dtype=float)


logger = logging.getLogger(__name__)

VarianceMetric = Literal["mad", "std"]


@dataclass
class PreprocessingResult:
    """Container for preprocessed expression matrix and feature selection diagnostics."""

    scaled_expression: pd.DataFrame
    imputed_expression: pd.DataFrame
    feature_metrics: pd.DataFrame
    dropped_missing_genes: List[str] = field(default_factory=list)
    dropped_zero_var_genes: List[str] = field(default_factory=list)
    log2_applied: bool = False
    summary_stats: Dict[str, float] = field(default_factory=dict)


def compute_mad(df: pd.DataFrame) -> pd.Series:
    """Compute Median Absolute Deviation (MAD) across samples for each gene."""
    medians = df.median(axis=0)
    mad = (df.sub(medians, axis=1)).abs().median(axis=0)
    return mad


def preprocess_expression(
    expression_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    phenotype_col: Optional[str] = None,
    max_missing_ratio: float = 0.10,
    imputation_strategy: str = "median",
    auto_log2: bool = False,
    top_genes: Optional[int] = 500,
    variance_metric: VarianceMetric = "mad",
    use_anova_filter: bool = False,
    anova_alpha: float = 0.05,
    anova_top_k: Optional[int] = None,
) -> PreprocessingResult:
    """
    Clean, filter, impute, and standardize the gene expression matrix.

    Strategy to ensure high explained variance in the first 3-5 principal components
    (>= 60-70%):
    1. Filter out genes with > `max_missing_ratio` (default 10%) missing values.
    2. Impute remaining sporadic NaNs with `SimpleImputer(strategy='median')`.
    3. Remove zero-variance / near-zero-variance markers.
    4. Select top-K most variable genes by MAD (Median Absolute Deviation) or SD,
       and optionally refine via one-way ANOVA F-test across `phenotype_col` groups
       to concentrate biological signal.
    5. Apply Z-score standardization (`StandardScaler`) across samples.
    """
    if expression_df.empty:
        raise ValueError("Input expression DataFrame is empty.")

    n_samples, n_initial_genes = expression_df.shape
    logger.info(
        "Starting preprocessing on expression matrix: %d samples x %d genes.",
        n_samples,
        n_initial_genes,
    )

    # 1. Missing Value Filtering (>10% missing)
    missing_ratios = expression_df.isna().mean(axis=0)
    keep_mask = missing_ratios <= max_missing_ratio
    dropped_missing = expression_df.columns[~keep_mask].tolist()

    if dropped_missing:
        logger.info(
            "Dropped %d genes exceeding %.1f%% missing value threshold.",
            len(dropped_missing),
            max_missing_ratio * 100.0,
        )

    filtered_df = expression_df.loc[:, keep_mask].copy()
    if filtered_df.shape[1] == 0:
        raise ValueError(
            f"All {n_initial_genes} genes exceeded the missing ratio threshold of {max_missing_ratio:.2f}."
        )

    # 2. Imputation via SimpleImputer(strategy='median')
    total_missing_remaining = int(filtered_df.isna().sum().sum())
    imputer = SimpleImputer(strategy=imputation_strategy)
    imputed_array = imputer.fit_transform(filtered_df)
    imputed_df = pd.DataFrame(
        imputed_array,
        index=filtered_df.index,
        columns=filtered_df.columns,
    )
    if total_missing_remaining > 0:
        logger.info(
            "Imputed %d remaining missing values using SimpleImputer(strategy='%s').",
            total_missing_remaining,
            imputation_strategy,
        )

    # 3. Optional auto-log2 transform if raw counts/intensities are skewed
    log2_applied = False
    if auto_log2:
        q99 = float(np.nanpercentile(imputed_df.values, 99))
        min_val = float(np.nanmin(imputed_df.values))
        if q99 > 100.0 and min_val >= 0.0:
            logger.info("Applying log2(x + 1) transformation (99th percentile = %.2f).", q99)
            imputed_df = np.log2(imputed_df + 1.0)
            log2_applied = True

    # 4. Drop zero-variance genes
    stds = imputed_df.std(axis=0, ddof=1)
    non_zero_mask = stds > 1e-8
    dropped_zero_var = imputed_df.columns[~non_zero_mask].tolist()
    if dropped_zero_var:
        logger.info("Dropped %d constant/zero-variance genes.", len(dropped_zero_var))
        imputed_df = imputed_df.loc[:, non_zero_mask].copy()
        stds = stds.loc[non_zero_mask]

    # 5. Compute variability metrics (MAD and SD)
    mads = compute_mad(imputed_df)
    metrics_df = pd.DataFrame(
        {
            "gene_id": imputed_df.columns,
            "missing_ratio": missing_ratios.loc[imputed_df.columns].values,
            "mean": imputed_df.mean(axis=0).values,
            "median": imputed_df.median(axis=0).values,
            "std": stds.values,
            "mad": mads.values,
        },
        index=imputed_df.columns,
    )

    # 6. Optional ANOVA F-test calculation across target phenotype groups
    if metadata_df is not None and phenotype_col and phenotype_col in metadata_df.columns:
        pheno_series = metadata_df[phenotype_col].reindex(imputed_df.index)
        valid_mask = pheno_series.notna()
        valid_groups = pheno_series[valid_mask]
        group_counts = valid_groups.value_counts()
        eligible_classes = group_counts[group_counts >= 2].index.tolist()
        valid_mask = pheno_series.isin(eligible_classes)

        if len(eligible_classes) >= 2 and valid_mask.sum() >= len(eligible_classes) * 2:
            f_vals, p_vals = _compute_anova_f_classif(
                imputed_df.loc[valid_mask], pheno_series.loc[valid_mask]
            )
            metrics_df["anova_f_stat"] = f_vals
            metrics_df["anova_p_value"] = p_vals
        else:
            logger.warning(
                "Phenotype '%s' has fewer than 2 valid groups with >=2 samples; skipping ANOVA F-test.",
                phenotype_col,
            )
            metrics_df["anova_f_stat"] = np.nan
            metrics_df["anova_p_value"] = np.nan
    else:
        metrics_df["anova_f_stat"] = np.nan
        metrics_df["anova_p_value"] = np.nan

    # 7. Feature Selection (Top-K by MAD/SD + Optional ANOVA filter)
    sort_col = "mad" if variance_metric == "mad" else "std"
    selected_genes = metrics_df.sort_values(by=sort_col, ascending=False).index.tolist()

    if top_genes is not None and top_genes > 0 and top_genes < len(selected_genes):
        selected_genes = selected_genes[:top_genes]
        logger.info(
            "Filtered top %d most variable markers by %s.",
            len(selected_genes),
            sort_col.upper(),
        )

    if use_anova_filter and metrics_df["anova_p_value"].notna().any():
        sub_metrics = metrics_df.loc[selected_genes].sort_values(
            by="anova_f_stat", ascending=False
        )
        if anova_top_k is not None and anova_top_k > 0:
            selected_genes = sub_metrics.head(anova_top_k).index.tolist()
            logger.info(
                "Refined feature set to top %d genes by ANOVA F-statistic across '%s'.",
                len(selected_genes),
                phenotype_col,
            )
        else:
            sig_genes = sub_metrics[sub_metrics["anova_p_value"] <= anova_alpha].index.tolist()
            if len(sig_genes) >= 10:
                selected_genes = sig_genes
                logger.info(
                    "Refined feature set to %d significant genes (ANOVA p <= %.3f) across '%s'.",
                    len(selected_genes),
                    anova_alpha,
                    phenotype_col,
                )
            else:
                fallback_k = max(10, len(sub_metrics) // 2)
                selected_genes = sub_metrics.head(fallback_k).index.tolist()
                logger.warning(
                    "Fewer than 10 genes passed ANOVA p <= %.3f; keeping top %d genes by F-statistic.",
                    anova_alpha,
                    len(selected_genes),
                )

    metrics_df["selected_for_pca"] = metrics_df.index.isin(selected_genes)
    selected_imputed_df = imputed_df.loc[:, selected_genes].copy()

    # 8. Z-score Standardization (StandardScaler)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(selected_imputed_df)
    scaled_df = pd.DataFrame(
        scaled_array,
        index=selected_imputed_df.index,
        columns=selected_imputed_df.columns,
    )

    logger.info(
        "Preprocessing complete: %d samples x %d standardized markers ready for PCA.",
        scaled_df.shape[0],
        scaled_df.shape[1],
    )

    return PreprocessingResult(
        scaled_expression=scaled_df,
        imputed_expression=selected_imputed_df,
        feature_metrics=metrics_df,
        dropped_missing_genes=dropped_missing,
        dropped_zero_var_genes=dropped_zero_var,
        log2_applied=log2_applied,
        summary_stats={
            "initial_genes": float(n_initial_genes),
            "dropped_missing_genes": float(len(dropped_missing)),
            "dropped_zero_var_genes": float(len(dropped_zero_var)),
            "selected_genes": float(scaled_df.shape[1]),
            "imputed_values_count": float(total_missing_remaining),
        },
    )