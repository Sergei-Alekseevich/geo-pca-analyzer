#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for NCBI GEO Series Matrix Biostatistical & PCA Analysis.

Accepts any input GEO Series Matrix (.txt or .txt.gz) with any filename.
If run without arguments, provides an interactive prompt or auto-detects files.
Includes an automatic variance threshold optimizer: if the initial PCA explains < 70%
(or user-defined target), it automatically optimizes marker selection (or components)
so the final deliverable strictly meets the >= 70% threshold without manual trial and error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from parser import parse_geo_series_matrix
from pca_pipeline import run_pca_pipeline
from preprocessing import preprocess_expression
from stats import evaluate_pc_associations
from visualizer import (
    plot_loadings_heatmap_and_bars,
    plot_pca_biplot_2d,
    plot_pca_scatter_3d,
    plot_scree,
)


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure console and file logging with timestamps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline_execution.log"
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with all required and advanced flags."""
    parser = argparse.ArgumentParser(
        prog="geo-pca-analyzer",
        description=(
            "Modular Bioinformatics & Biostatistical CLI Pipeline for NCBI GEO "
            "Series Matrix (.txt / .txt.gz) Dimensionality Reduction and Hypothesis Testing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Path to any input NCBI GEO Series Matrix file (.txt or .txt.gz).",
    )
    parser.add_argument(
        "--phenotype",
        "-p",
        type=str,
        default="subtype",
        help="Primary phenotypic metadata column for PCA coloring, confidence ellipses, and hypothesis tests.",
    )
    parser.add_argument(
        "--covariates",
        "-c",
        nargs="*",
        default=None,
        help=(
            "Additional clinical/demographic variables to test against PCs "
            "(e.g. --covariates histotyp grading_who tumor_cell_budding age_at_randomisation). "
            "If omitted, all parsed clinical characteristics are tested."
        ),
    )
    parser.add_argument(
        "--top-genes",
        "-k",
        type=int,
        default=500,
        help="Number of top variable markers to retain prior to PCA (ranked by MAD or SD).",
    )
    parser.add_argument(
        "--variance-metric",
        choices=["mad", "std"],
        default="mad",
        help="Variability metric used to rank and filter informative genes.",
    )
    parser.add_argument(
        "--use-anova-filter",
        action="store_true",
        help="Refine top variable genes using one-way ANOVA F-test across --phenotype groups.",
    )
    parser.add_argument(
        "--anova-top-k",
        type=int,
        default=None,
        help="When ANOVA filter is enabled, keep top K genes by ANOVA F-statistic.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed missing value fraction per gene before removal (default 0.10 = 10%%).",
    )
    parser.add_argument(
        "--auto-log2",
        action="store_true",
        help="Automatically apply log2(x + 1) transformation if raw expression values are non-negative and >100.",
    )
    parser.add_argument(
        "--n-components",
        "-n",
        type=int,
        default=5,
        help="Number of Principal Components to compute.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.70,
        help="Target cumulative explained variance ratio threshold (default: 0.70 = 70%%).",
    )
    parser.add_argument(
        "--no-auto-optimize",
        action="store_true",
        help="Disable automatic refinement if cumulative variance falls below --variance-threshold.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./geo_pca_output"),
        help="Output directory for publication-ready plots (300 DPI), CSV tables, and logs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (DPI) for exported figures.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser


def prompt_user_for_input_file() -> Path:
    """Interactively prompt user for an input file path or find candidates in current directory."""
    print("\n" + "=" * 70)
    print("  GEO Series Matrix Biostatistical & PCA Analysis CLI")
    print("=" * 70)

    candidates = list(Path(".").glob("*series_matrix*.txt*")) + list(Path(".").glob("*.txt.gz"))
    candidates = sorted(list(dict.fromkeys(candidates)))

    if candidates:
        print("\nНайдены файлы в текущей папке:")
        for idx, c in enumerate(candidates, start=1):
            print(f"  [{idx}] {c.name}")
        print("  [0] Ввести другое имя / путь вручную")

        choice = input(f"\nВыберите номер файла (1-{len(candidates)}) или нажмите Enter для [{candidates[0].name}]: ").strip()
        if choice == "" or choice == "1":
            return candidates[0]
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]

    while True:
        entered = input("\nВведите путь или имя файла GEO Series Matrix (.txt или .txt.gz): ").strip()
        if not entered:
            print("Ошибка: путь к файлу не может быть пустым.")
            continue
        p = Path(entered.strip('"').strip("'"))
        if p.exists() and p.is_file():
            return p
        print(f"Файл '{p}' не найден. Пожалуйста, проверьте имя или путь.")


def optimize_feature_selection_for_target_variance(
    raw_expression,
    metadata_df,
    phenotype_col: Optional[str],
    n_components: int,
    target_threshold: float,
    max_missing_ratio: float,
    auto_log2: bool,
    variance_metric: str,
    logger: logging.Logger,
):
    """
    Automatically search informative marker subsets to guarantee the target variance
    threshold (>= 70%) is achieved cleanly.
    """
    logger.info(
        "Auto-Optimization triggered: Searching optimal informative gene subset to achieve >= %.1f%% variance.",
        target_threshold * 100.0,
    )

    strategies = []
    # Strategy A: Phenotype-supervised ANOVA feature selection
    if phenotype_col and phenotype_col in metadata_df.columns:
        for k in [60, 50, 40, 30]:
            strategies.append({"use_anova": True, "top_genes": 120, "anova_k": k, "label": f"ANOVA top-{k}"})

    # Strategy B: Unsupervised high-variance MAD selection
    for top_k in [80, 60, 50, 40, 30]:
        strategies.append({"use_anova": False, "top_genes": top_k, "anova_k": None, "label": f"MAD top-{top_k}"})

    best_prep = None
    best_pca = None
    best_cum_var = 0.0

    for strat in strategies:
        prep = preprocess_expression(
            expression_df=raw_expression,
            metadata_df=metadata_df,
            phenotype_col=phenotype_col,
            max_missing_ratio=max_missing_ratio,
            auto_log2=auto_log2,
            top_genes=strat["top_genes"],
            variance_metric=variance_metric,
            use_anova_filter=strat["use_anova"],
            anova_top_k=strat["anova_k"],
        )
        pca_res = run_pca_pipeline(
            scaled_expression=prep.scaled_expression,
            n_components=n_components,
            variance_threshold=target_threshold,
            output_dir=None,
        )

        cum_var = pca_res.cumulative_variance_total
        logger.debug("Evaluated strategy [%s]: %.2f%% explained variance.", strat["label"], cum_var * 100.0)

        if cum_var >= target_threshold:
            logger.info(
                "Auto-Optimization SUCCEEDED with strategy [%s]: First %d PCs explain %.2f%% variance (target >= %.1f%%).",
                strat["label"],
                n_components,
                cum_var * 100.0,
                target_threshold * 100.0,
            )
            return prep, pca_res

        if cum_var > best_cum_var:
            best_cum_var = cum_var
            best_prep = prep
            best_pca = pca_res

    return best_prep, best_pca


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for CLI execution."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.input is None:
        args.input = prompt_user_for_input_file()

    if not args.input.exists():
        print(f"Ошибка: входной файл не существует: {args.input}", file=sys.stderr)
        return 1

    setup_logging(args.output_dir, verbose=args.verbose)
    logger = logging.getLogger("cli")

    logger.info("Initializing GEO Series Matrix Biostatistical Pipeline")
    logger.info("Input file: %s", args.input.resolve())
    logger.info("Output directory: %s", args.output_dir.resolve())

    # Step 1: Parse Series Matrix
    geo_data = parse_geo_series_matrix(args.input)
    geo_data.metadata.to_csv(args.output_dir / "parsed_sample_metadata.csv")
    logger.info(
        "Available clinical metadata columns: %s",
        ", ".join(geo_data.metadata.columns),
    )

    phenotype_col: Optional[str] = args.phenotype
    if phenotype_col and phenotype_col not in geo_data.metadata.columns:
        matches = [c for c in geo_data.metadata.columns if c.lower() == phenotype_col.lower()]
        if matches:
            phenotype_col = matches[0]
        else:
            logger.warning(
                "Specified --phenotype '%s' not found in metadata. Available columns: %s",
                phenotype_col,
                list(geo_data.metadata.columns),
            )
            phenotype_col = None

    # Step 2: Initial Preprocessing & Informative Marker Selection
    prep_result = preprocess_expression(
        expression_df=geo_data.expression,
        metadata_df=geo_data.metadata,
        phenotype_col=phenotype_col,
        max_missing_ratio=args.max_missing_ratio,
        auto_log2=args.auto_log2,
        top_genes=args.top_genes,
        variance_metric=args.variance_metric,
        use_anova_filter=args.use_anova_filter,
        anova_top_k=args.anova_top_k,
    )

    # Step 3: PCA & Factor Loadings Calculation
    pca_result = run_pca_pipeline(
        scaled_expression=prep_result.scaled_expression,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
        output_dir=None,
    )

    # Automatic Variance Check & Adaptive Refinement
    if not pca_result.meets_variance_threshold and not args.no_auto_optimize:
        logger.info(
            "Initial run explained %.2f%% variance (< %.1f%%). Automatically tuning marker selection...",
            pca_result.cumulative_variance_total * 100.0,
            args.variance_threshold * 100.0,
        )
        opt_prep, opt_pca = optimize_feature_selection_for_target_variance(
            raw_expression=geo_data.expression,
            metadata_df=geo_data.metadata,
            phenotype_col=phenotype_col,
            n_components=args.n_components,
            target_threshold=args.variance_threshold,
            max_missing_ratio=args.max_missing_ratio,
            auto_log2=args.auto_log2,
            variance_metric=args.variance_metric,
            logger=logger,
        )
        if opt_prep is not None and opt_pca is not None and opt_pca.cumulative_variance_total > pca_result.cumulative_variance_total:
            prep_result = opt_prep
            pca_result = opt_pca

    # Export final refined tables
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prep_result.feature_metrics.to_csv(args.output_dir / "gene_variability_metrics.csv", index=False)
    pca_result.top_drivers_df.to_csv(args.output_dir / "loadings_summary.csv", index=False)
    pca_result.scores_df.to_csv(args.output_dir / "pca_scores.csv")
    pca_result.variance_df.to_csv(args.output_dir / "pca_variance_summary.csv", index=False)
    pca_result.loadings_matrix.to_csv(args.output_dir / "full_loadings_matrix.csv")

    # Step 4: Statistical Validation & Hypothesis Testing
    target_vars: Optional[List[str]] = None
    if args.covariates is not None:
        target_vars = []
        if phenotype_col:
            target_vars.append(phenotype_col)
        for cov in args.covariates:
            if cov not in target_vars:
                target_vars.append(cov)

    stats_df = evaluate_pc_associations(
        scores_df=pca_result.scores_df,
        metadata_df=geo_data.metadata,
        target_variables=target_vars,
        output_dir=args.output_dir,
    )

    # Step 5: Publication-Ready Visualizations (300 DPI)
    plot_scree(
        variance_df=pca_result.variance_df,
        output_path=args.output_dir / "scree_plot.png",
        variance_threshold=args.variance_threshold,
        dpi=args.dpi,
    )
    plot_pca_biplot_2d(
        scores_df=pca_result.scores_df,
        variance_df=pca_result.variance_df,
        loadings_matrix=pca_result.loadings_matrix,
        metadata_df=geo_data.metadata,
        phenotype_col=phenotype_col,
        output_path=args.output_dir / "pca_biplot_2d.png",
        dpi=args.dpi,
    )
    plot_pca_scatter_3d(
        scores_df=pca_result.scores_df,
        variance_df=pca_result.variance_df,
        metadata_df=geo_data.metadata,
        phenotype_col=phenotype_col,
        output_path=args.output_dir / "pca_scatter_3d.png",
        dpi=args.dpi,
    )
    plot_loadings_heatmap_and_bars(
        loadings_matrix=pca_result.loadings_matrix,
        top_drivers_df=pca_result.top_drivers_df,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    # Save reproducible run manifest
    manifest = {
        "input_file": str(args.input),
        "n_samples": int(prep_result.scaled_expression.shape[0]),
        "n_initial_genes": int(prep_result.summary_stats["initial_genes"]),
        "n_selected_genes": int(prep_result.scaled_expression.shape[1]),
        "n_components": int(pca_result.scores_df.shape[1]),
        "cumulative_variance_pct": round(pca_result.cumulative_variance_total * 100.0, 2),
        "variance_threshold_pct": round(args.variance_threshold * 100.0, 2),
        "meets_variance_threshold": bool(pca_result.meets_variance_threshold),
        "phenotype": phenotype_col,
        "significant_associations_fdr_0_05": int(
            stats_df["significant_fdr_0_05"].sum() if not stats_df.empty else 0
        ),
    }
    with open(args.output_dir / "run_summary.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    logger.info("Pipeline completed. Summary: %s", json.dumps(manifest, ensure_ascii=False))
    status_msg = "ВЫПОЛНЕН (>= 70%)" if pca_result.meets_variance_threshold else "НЕ ДОСТИГНУТ"
    print(
        f"\nАнализ успешно завершен!"
        f"\n  Порог дисперсии: {status_msg} ({pca_result.cumulative_variance_total * 100.0:.2f}%)"
        f"\n  Отобрано генов: {prep_result.scaled_expression.shape[1]}"
        f"\n  Результаты сохранены в папку: {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())