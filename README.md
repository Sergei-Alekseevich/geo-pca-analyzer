# geo-pca-analyzer
Modular Python CLI pipeline for NCBI GEO Series Matrix biostatistical analysis: streaming parsing, MAD/ANOVA feature selection, PCA dimensionality reduction (≥70% variance), factor loadings, hypothesis testing with FDR, and publication-ready 300 DPI visualizations.

# GEO Series Matrix Biostatistical & PCA Analysis CLI

A modular Python command-line tool for automated streaming parsing of NCBI GEO Series Matrix files (`.txt` and `.txt.gz`), gene expression preprocessing, Principal Component Analysis (PCA / Factor Analysis), and clinical hypothesis testing with Benjamini-Hochberg FDR correction.

Developed and validated on the CeFCiD cohort dataset (**NCBI GEO GSE212070**, 103 R/M-HNSCC patients, 231 NanoString genes, 38 clinical covariates).

---

## Key Features

* **Streaming Parser:** Robust extraction of sparse, shifted `!Sample_characteristics_ch1` metadata and expression tables without high memory overhead.
* **Smart Preprocessing:** Missing-value filtering (>10%), median imputation, MAD/SD marker selection, and Z-score standardization.
* **Target Variance Optimizer:** Automatically tunes gene selection to ensure the first components satisfy cumulative explained variance thresholds (e.g. $\ge 70\%$).
* **Factor Loadings:** Identifies top positive and negative driver genes for PC1–PC3.
* **Hypothesis Testing:** One-way ANOVA and Kruskal-Wallis for categorical factors; Pearson and Spearman rank correlation for continuous traits, with global FDR (BH) adjustment.
* **Publication-Quality Visuals:** 300 DPI Scree plots, 2D Biplots with 95% confidence ellipses, 3D projections, and loadings heatmaps/barplots.

---

## Project Structure

| File | Description |
| :--- | :--- |
| `cli.py` | Command-line entry point with interactive file detection and automated parameter optimization. |
| `parser.py` | Streaming parser for plain `.txt` and gzip-compressed `.txt.gz` Series Matrix files. |
| `preprocessing.py` | Data hygiene, median imputation, variability ranking, and Z-score scaling. |
| `pca_pipeline.py` | PCA computation, explained variance evaluation, and factor loading matrices. |
| `stats.py` | Parametric and non-parametric hypothesis tests with Benjamini-Hochberg FDR correction. |
| `visualizer.py` | High-resolution (300 DPI) plotting routines. |
| `requirements.txt` | Core dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`). |

---

## Installation

```bash
git clone [https://github.com/Sergei-Alekseevich/](https://github.com/Sergei-Alekseevich/)<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
