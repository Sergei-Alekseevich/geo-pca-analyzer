"""
Statistical validation and hypothesis testing between Principal Components and clinical covariates.

Automatically dispatches categorical covariates to one-way ANOVA and Kruskal-Wallis tests,
and continuous covariates to Pearson correlation, Spearman rank correlation, and OLS linear
regression. Applies Benjamini-Hochberg False Discovery Rate (FDR) correction across tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def benjamini_hochberg_fdr(p_values: Sequence[float]) -> np.ndarray:
    """
    Compute Benjamini-Hochberg FDR-adjusted p-values (q-values), handling NaNs safely.
    """
    p_arr = np.asarray(p_values, dtype=float)
    q_arr = np.full_like(p_arr, np.nan, dtype=float)

    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]
    m = len(valid_p)
    if m == 0:
        return q_arr

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    raw_q = ranked_p * m / ranks
    # Enforce monotonicity from largest to smallest rank
    monotonic_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotonic_q = np.clip(monotonic_q, 0.0, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    q_arr[valid_mask] = monotonic_q[inv_order]
    return q_arr


def _is_categorical_series(series: pd.Series, max_unique_for_numeric_cat: int = 5) -> bool:
    """
    Determine whether a clinical metadata series should be treated as categorical
    or continuous.

    Numeric columns with <= `max_unique_for_numeric_cat` distinct values (such as
    `keratinization: 0, 1, 2, 3` or `study_arm: 1, 2`) are treated as categorical
    unless they represent continuous measurements.
    """
    clean = series.dropna()
    if clean.empty:
        return True
    if pd.api.types.is_string_dtype(clean) or pd.api.types.is_categorical_dtype(clean) or pd.api.types.is_bool_dtype(clean):
        return True
    if pd.api.types.is_numeric_dtype(clean):
        n_unique = clean.nunique()
        if n_unique <= max_unique_for_numeric_cat:
            return True
        return False
    return True


def evaluate_pc_associations(
    scores_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    target_variables: Optional[Sequence[str]] = None,
    max_unique_for_numeric_cat: int = 5,
    min_group_size: int = 2,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Test statistical associations between Principal Components (PC1..PCk) and
    clinical/demographic variables.

    - Categorical variables: One-way ANOVA (parametric) and Kruskal-Wallis H-test (non-parametric).
    - Quantitative variables: Pearson r / OLS slope (parametric) and Spearman rho (non-parametric).
    - Applies Benjamini-Hochberg FDR adjustment across all conducted hypothesis tests.
    """
    aligned_meta = metadata_df.reindex(scores_df.index)

    if target_variables is None or len(target_variables) == 0:
        # Exclude pure sample identifier columns
        excluded = {"sample_title", "sample_type", "sample_source_name_ch1", "sample_organism_ch1", "sample_platform_id"}
        candidate_vars = [c for c in aligned_meta.columns if c not in excluded]
    else:
        candidate_vars = []
        for var in target_variables:
            if var in aligned_meta.columns:
                candidate_vars.append(var)
            else:
                # Case-insensitive fallback lookup
                matches = [c for c in aligned_meta.columns if c.lower() == var.lower()]
                if matches:
                    candidate_vars.append(matches[0])
                else:
                    logger.warning(
                        "Requested covariate '%s' not found in sample metadata (available: %s).",
                        var,
                        ", ".join(aligned_meta.columns),
                    )

    records: List[dict] = []

    for var_name in candidate_vars:
        raw_series = aligned_meta[var_name]
        is_cat = _is_categorical_series(raw_series, max_unique_for_numeric_cat=max_unique_for_numeric_cat)

        for pc_col in scores_df.columns:
            pc_series = scores_df[pc_col]
            valid_df = pd.DataFrame({"pc": pc_series, "cov": raw_series}).dropna()
            n_valid = len(valid_df)

            if n_valid < 4:
                continue

            if is_cat:
                # Group by categorical levels with >= min_group_size observations
                valid_df["cov_str"] = valid_df["cov"].astype(str)
                counts = valid_df["cov_str"].value_counts()
                valid_levels = counts[counts >= min_group_size].index.tolist()

                if len(valid_levels) < 2:
                    continue

                filtered_df = valid_df[valid_df["cov_str"].isin(valid_levels)]
                groups = [
                    grp["pc"].values
                    for _, grp in filtered_df.groupby("cov_str", sort=True)
                ]
                group_summary = ", ".join(
                    f"{lvl} (n={int(counts[lvl])})" for lvl in sorted(valid_levels)
                )

                # 1. One-way ANOVA
                try:
                    f_stat, f_pval = stats.f_oneway(*groups)
                except Exception:
                    f_stat, f_pval = np.nan, np.nan

                # Compute eta-squared effect size: SS_between / SS_total
                grand_mean = filtered_df["pc"].mean()
                ss_total = float(((filtered_df["pc"] - grand_mean) ** 2).sum())
                ss_between = float(
                    sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups)
                )
                eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

                records.append(
                    {
                        "component": pc_col,
                        "variable": var_name,
                        "variable_type": "categorical",
                        "test_name": "One-way ANOVA (F-test)",
                        "n_samples": int(len(filtered_df)),
                        "n_groups_or_df": int(len(valid_levels)),
                        "groups_detail": group_summary,
                        "statistic": float(f_stat) if not pd.isna(f_stat) else np.nan,
                        "effect_size_name": "eta_squared",
                        "effect_size": float(eta_sq) if not pd.isna(eta_sq) else np.nan,
                        "p_value": float(f_pval) if not pd.isna(f_pval) else np.nan,
                    }
                )

                # 2. Kruskal-Wallis non-parametric test
                try:
                    kw_stat, kw_pval = stats.kruskal(*groups)
                except Exception:
                    kw_stat, kw_pval = np.nan, np.nan

                n_tot = len(filtered_df)
                k_grp = len(valid_levels)
                epsilon_sq = (
                    float((kw_stat - k_grp + 1) / (n_tot - k_grp))
                    if (n_tot - k_grp) > 0 and not pd.isna(kw_stat)
                    else np.nan
                )

                records.append(
                    {
                        "component": pc_col,
                        "variable": var_name,
                        "variable_type": "categorical",
                        "test_name": "Kruskal-Wallis H-test",
                        "n_samples": int(n_tot),
                        "n_groups_or_df": int(k_grp),
                        "groups_detail": group_summary,
                        "statistic": float(kw_stat) if not pd.isna(kw_stat) else np.nan,
                        "effect_size_name": "epsilon_squared",
                        "effect_size": float(epsilon_sq) if not pd.isna(epsilon_sq) else np.nan,
                        "p_value": float(kw_pval) if not pd.isna(kw_pval) else np.nan,
                    }
                )

            else:
                # Continuous variable: Pearson + OLS linear regression and Spearman rank correlation
                x = pd.to_numeric(valid_df["cov"], errors="coerce")
                y = valid_df["pc"]
                mask = x.notna() & y.notna()
                x_clean = x[mask].values.astype(float)
                y_clean = y[mask].values.astype(float)

                if len(x_clean) < 4 or np.std(x_clean) < 1e-12:
                    continue

                range_summary = (
                    f"mean={np.mean(x_clean):.2f}, sd={np.std(x_clean, ddof=1):.2f}, "
                    f"median={np.median(x_clean):.2f} [{np.min(x_clean):.1f}-{np.max(x_clean):.1f}]"
                )

                # 1. Pearson correlation / OLS linear regression
                try:
                    reg = stats.linregress(x_clean, y_clean)
                    r_val = float(reg.rvalue)
                    p_pearson = float(reg.pvalue)
                    slope = float(reg.slope)
                except Exception:
                    r_val, p_pearson, slope = np.nan, np.nan, np.nan

                records.append(
                    {
                        "component": pc_col,
                        "variable": var_name,
                        "variable_type": "continuous",
                        "test_name": "Pearson Correlation / OLS Regression",
                        "n_samples": int(len(x_clean)),
                        "n_groups_or_df": int(len(x_clean) - 2),
                        "groups_detail": f"{range_summary}; beta={slope:.4f}",
                        "statistic": r_val,
                        "effect_size_name": "r_squared",
                        "effect_size": float(r_val**2) if not pd.isna(r_val) else np.nan,
                        "p_value": p_pearson,
                    }
                )

                # 2. Spearman rank correlation
                try:
                    rho_val, p_spearman = stats.spearmanr(x_clean, y_clean)
                    rho_val = float(rho_val)
                    p_spearman = float(p_spearman)
                except Exception:
                    rho_val, p_spearman = np.nan, np.nan

                records.append(
                    {
                        "component": pc_col,
                        "variable": var_name,
                        "variable_type": "continuous",
                        "test_name": "Spearman Rank Correlation",
                        "n_samples": int(len(x_clean)),
                        "n_groups_or_df": int(len(x_clean) - 2),
                        "groups_detail": range_summary,
                        "statistic": rho_val,
                        "effect_size_name": "spearman_rho",
                        "effect_size": rho_val,
                        "p_value": p_spearman,
                    }
                )

    results_df = pd.DataFrame.from_records(records)
    if results_df.empty:
        logger.warning("No valid statistical tests could be performed on the provided metadata.")
        return results_df

    # Compute FDR (Benjamini-Hochberg) globally and per test family
    results_df["p_value_fdr_bh"] = benjamini_hochberg_fdr(results_df["p_value"].values)
    results_df["significant_fdr_0_05"] = results_df["p_value_fdr_bh"] < 0.05
    results_df["significant_nominal_0_05"] = results_df["p_value"] < 0.05

    results_df = results_df.sort_values(by=["p_value_fdr_bh", "p_value"], ascending=[True, True]).reset_index(
        drop=True
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stats_csv = out_path / "statistical_tests_pvalues.csv"
        results_df.to_csv(stats_csv, index=False)
        logger.info("Exported statistical hypothesis test summary to %s", stats_csv)

    return results_df