"""
Streaming parser for NCBI GEO Series Matrix files (.txt and .txt.gz).

Extracts series-level metadata, sample-level clinical/phenotypic characteristics
(handling heterogeneous !Sample_characteristics_ch1 rows), and the gene expression
matrix (transposed to Samples x Genes and aligned with metadata).
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GEOSeriesData:
    """Container for parsed NCBI GEO Series Matrix data."""

    expression: pd.DataFrame
    metadata: pd.DataFrame
    series_metadata: Dict[str, List[str]] = field(default_factory=dict)


def _open_maybe_gzip(filepath: Union[str, Path]) -> TextIO:
    """Open a plain text or gzip-compressed file in text mode with UTF-8 encoding."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "rb") as raw_f:
        magic = raw_f.read(2)

    if magic == b"\x1f\x8b" or path.suffix.lower() == ".gz":
        logger.info("Opening gzip-compressed Series Matrix file: %s", path)
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")

    logger.info("Opening plain-text Series Matrix file: %s", path)
    return open(path, mode="rt", encoding="utf-8", errors="replace")


def _clean_token(token: str) -> str:
    """Strip surrounding whitespace, quotes, and non-breaking spaces."""
    cleaned = token.strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    return cleaned.replace("\xa0", " ").strip()


def _parse_metadata_value(val: str) -> Optional[Union[int, float, str]]:
    """Convert raw string values into NA, int, float, or clean string."""
    if val == "" or val.lower() in {"unknown", "na", "n/a", "null", "none", "nan", "--"}:
        return np.nan
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_geo_series_matrix(
    filepath: Union[str, Path],
    drop_empty_metadata_cols: bool = True,
) -> GEOSeriesData:
    """
    Stream-parse an NCBI GEO Series Matrix file (.txt or .txt.gz).

    Parameters
    ----------
    filepath : str or Path
        Path to the GEO Series Matrix file (.txt or .txt.gz).
    drop_empty_metadata_cols : bool, default=True
        Whether to drop metadata columns that contain only NaN values.

    Returns
    -------
    GEOSeriesData
        Dataclass with aligned `expression` (N samples x P genes DataFrame),
        `metadata` (N samples x K clinical features DataFrame), and `series_metadata`.
    """
    series_meta: Dict[str, List[str]] = {}
    sample_headers: Dict[str, List[str]] = {}
    characteristics_rows: List[List[str]] = []
    table_lines: List[str] = []
    in_expression_table = False

    with _open_maybe_gzip(filepath) as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            if line.startswith("!series_matrix_table_begin"):
                in_expression_table = True
                logger.debug("Found !series_matrix_table_begin at line %d", line_num)
                continue

            if line.startswith("!series_matrix_table_end"):
                in_expression_table = False
                logger.debug("Found !series_matrix_table_end at line %d", line_num)
                break

            if in_expression_table:
                table_lines.append(line)
                continue

            if line.startswith("!Series_"):
                parts = line.split("\t")
                key = parts[0][len("!Series_") :].strip()
                vals = [_clean_token(p) for p in parts[1:] if _clean_token(p)]
                series_meta.setdefault(key, []).extend(vals)

            elif line.startswith("!Sample_characteristics_ch"):
                parts = line.split("\t")
                vals = [_clean_token(p) for p in parts[1:]]
                characteristics_rows.append(vals)

            elif line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0][len("!Sample_") :].strip()
                vals = [_clean_token(p) for p in parts[1:]]
                if key in sample_headers:
                    sample_headers[key] = [
                        f"{prev}; {curr}" if prev and curr and prev != curr else (prev or curr)
                        for prev, curr in zip(sample_headers[key], vals)
                    ]
                else:
                    sample_headers[key] = vals

    # 1. Build Sample Metadata DataFrame
    if "geo_accession" not in sample_headers:
        raise ValueError(
            "Missing '!Sample_geo_accession' line in Series Matrix file; cannot identify samples."
        )

    sample_ids = sample_headers["geo_accession"]
    n_samples = len(sample_ids)
    logger.info("Parsed metadata header for %d samples.", n_samples)

    records: List[Dict[str, Optional[Union[int, float, str]]]] = [
        {"sample_id": sid} for sid in sample_ids
    ]

    kept_header_keys = {"title", "type", "source_name_ch1", "organism_ch1", "platform_id"}
    for key, vals in sample_headers.items():
        if key in kept_header_keys and len(vals) == n_samples:
            col_name = f"sample_{key}"
            for idx, val in enumerate(vals):
                records[idx][col_name] = _parse_metadata_value(val)

    # Parse heterogeneous key: value pairs from !Sample_characteristics_ch1
    for row_idx, row_vals in enumerate(characteristics_rows):
        for sample_idx, cell in enumerate(row_vals):
            if sample_idx >= n_samples or not cell:
                continue
            if ":" in cell:
                raw_key, raw_val = cell.split(":", 1)
                norm_key = (
                    raw_key.strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
                )
                parsed_val = _parse_metadata_value(raw_val.strip())
                existing = records[sample_idx].get(norm_key, np.nan)
                if pd.isna(existing) or existing is None:
                    records[sample_idx][norm_key] = parsed_val
                elif parsed_val is not np.nan and not pd.isna(parsed_val):
                    if str(existing) != str(parsed_val):
                        records[sample_idx][norm_key] = f"{existing}; {parsed_val}"
            else:
                fallback_key = f"characteristic_row_{row_idx + 1}"
                records[sample_idx][fallback_key] = _parse_metadata_value(cell)

    metadata_df = pd.DataFrame.from_records(records).set_index("sample_id")

    if drop_empty_metadata_cols:
        metadata_df = metadata_df.dropna(axis=1, how="all")

    for col in metadata_df.columns:
        try:
            metadata_df[col] = pd.to_numeric(metadata_df[col])
        except (ValueError, TypeError):
            pass

    # 2. Parse Expression Matrix Table
    if not table_lines:
        raise ValueError(
            "No expression matrix found between !series_matrix_table_begin and !series_matrix_table_end."
        )

    table_buffer = io.StringIO("\n".join(table_lines))
    raw_expr_df = pd.read_csv(
        table_buffer,
        sep="\t",
        quotechar='"',
        na_values=["", "NA", "null", "NULL", "NaN", "nan"],
        low_memory=False,
    )

    if raw_expr_df.empty:
        raise ValueError("Expression matrix table between begin/end tags is empty.")

    id_col = raw_expr_df.columns[0]
    raw_expr_df[id_col] = raw_expr_df[id_col].astype(str).map(_clean_token)
    raw_expr_df = raw_expr_df.set_index(id_col)
    raw_expr_df.index.name = "gene_id"

    raw_expr_df.columns = [_clean_token(str(c)) for c in raw_expr_df.columns]

    if raw_expr_df.index.duplicated().any():
        n_dups = int(raw_expr_df.index.duplicated().sum())
        logger.warning(
            "Found %d duplicate gene/probe IDs in expression table; aggregating by mean.",
            n_dups,
        )
        raw_expr_df = raw_expr_df.groupby(level=0).mean(numeric_only=True)

    # Transpose so rows = Samples (N) and columns = Genes/Markers (P)
    expr_df = raw_expr_df.T
    expr_df.index.name = "sample_id"
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

    # 3. Align Sample IDs between Metadata and Expression Matrix
    metadata_df, expr_df = align_samples(metadata_df, expr_df)

    logger.info(
        "Successfully parsed GEO Series Matrix: %d aligned samples, %d genes/markers, %d metadata variables.",
        expr_df.shape[0],
        expr_df.shape[1],
        metadata_df.shape[1],
    )

    return GEOSeriesData(
        expression=expr_df,
        metadata=metadata_df,
        series_metadata=series_meta,
    )


def align_samples(
    metadata_df: pd.DataFrame,
    expression_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align sample IDs between metadata and expression matrices, preserving order.

    Raises ValueError if there is no overlap between sample IDs.
    """
    common_samples = [sid for sid in metadata_df.index if sid in expression_df.index]

    if not common_samples:
        raise ValueError(
            "No overlapping sample IDs found between !Sample_geo_accession metadata "
            f"(e.g. {list(metadata_df.index[:3])}) and expression matrix columns "
            f"(e.g. {list(expression_df.index[:3])})."
        )

    if len(common_samples) < len(metadata_df.index) or len(common_samples) < len(expression_df.index):
        logger.warning(
            "Sample count mismatch during alignment: metadata=%d, expression=%d, intersection=%d.",
            len(metadata_df.index),
            len(expression_df.index),
            len(common_samples),
        )

    aligned_meta = metadata_df.loc[common_samples].copy()
    aligned_expr = expression_df.loc[common_samples].copy()
    return aligned_meta, aligned_expr