"""
Data Preprocessing Module for Supply Chain Predictive Analytics.

This module provides reusable functions for cleaning and preparing raw supply
chain delivery history data from:
data/raw/SCMS_Delivery_History_Dataset.csv

Target features and data hygiene:
- Parses dates safely, replacing text sentinels with NaT.
- Coerces dirty numeric columns (e.g. Weight, Freight Cost) to float, setting non-numeric strings to NaN.
- Derives whole-day delay targets:
    delay_days = Delivered to Client Date - Scheduled Delivery Date
    is_delayed = 1 if delay_days > 0 else 0
- Preserves missing target values without inventing or imputing dates.
- Preserves all rows in the dataset and logs counts of records with invalid target dates.
- Enforces data leakage safeguards: actual delivery dates and target-derived fields
  must never be passed to predictive feature matrices.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple
import pandas as pd
import numpy as np

# Configure module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Explicit list of columns that must NEVER be used as prediction features to prevent data leakage
LEAKAGE_COLUMNS = [
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "delay_days",
    "is_delayed",
]

# Date columns present in the dataset
DATE_COLUMNS = [
    "PQ First Sent to Client Date",
    "PO Sent to Vendor Date",
    "Scheduled Delivery Date",
    "Delivered to Client Date",
    "Delivery Recorded Date",
]

# Numeric columns that need safe conversion
NUMERIC_COLUMNS = [
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Line Item Quantity",
    "Line Item Insurance (USD)",
]


def parse_dates(df: pd.DataFrame, date_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Safely parse date columns into pandas datetime objects.

    Placeholder / sentinel values (e.g., 'Pre-PQ Process', 'Date Not Captured',
    'N/A - From RDC') are converted to pd.NaT without imputing or inventing dates.

    Args:
        df: Input DataFrame.
        date_cols: List of column names to parse as dates. Defaults to DATE_COLUMNS.

    Returns:
        DataFrame with parsed date columns.
    """
    df = df.copy()
    if date_cols is None:
        date_cols = [col for col in DATE_COLUMNS if col in df.columns]

    for col in date_cols:
        # errors='coerce' turns non-date strings and sentinels into NaT
        original_non_null = df[col].notna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        parsed_non_null = df[col].notna().sum()
        coerced_count = original_non_null - parsed_non_null
        if coerced_count > 0:
            logger.info(
                f"Column '{col}': coerced {coerced_count} placeholder/invalid values to NaT."
            )

    return df


def clean_numeric_fields(df: pd.DataFrame, num_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Safely convert numeric columns to float, coercing descriptive text to NaN.

    Handles mixed text/numeric columns such as 'Weight (Kilograms)' and
    'Freight Cost (USD)' which may contain strings like 'Weight Captured Separately',
    'Freight Included in Commodity Cost', or reference IDs like 'See DN-304'.

    Args:
        df: Input DataFrame.
        num_cols: List of column names to convert. Defaults to NUMERIC_COLUMNS.

    Returns:
        DataFrame with cleaned numeric columns.
    """
    df = df.copy()
    if num_cols is None:
        num_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]

    for col in num_cols:
        # Check original non-null before coercion
        original_non_null = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_non_null = df[col].notna().sum()
        coerced_count = original_non_null - numeric_non_null
        if coerced_count > 0:
            logger.info(
                f"Column '{col}': coerced {coerced_count} non-numeric/text values to NaN."
            )

    return df


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive delivery delay targets from scheduled and actual delivery dates.

    Calculates:
    - delay_days: Whole-day difference (Delivered to Client Date - Scheduled Delivery Date).
                  Negative values indicate early deliveries, 0 indicates on-time,
                  positive values indicate late deliveries.
    - is_delayed: Binary indicator (1 if delay_days > 0, 0 if delay_days <= 0).

    Safeguards:
    - If either Scheduled Delivery Date or Delivered to Client Date is NaT,
      both delay_days and is_delayed are left as NaN. No dates or targets are imputed.
    - Day-level granularity only: does not imply hour-level precision.
    - Logs counts of rows with missing/unparseable target dates.

    Args:
        df: Input DataFrame containing 'Scheduled Delivery Date' and 'Delivered to Client Date'.

    Returns:
        DataFrame with added target columns 'delay_days' and 'is_delayed'.
    """
    df = df.copy()

    required_dates = ["Scheduled Delivery Date", "Delivered to Client Date"]
    for col in required_dates:
        if col not in df.columns:
            raise KeyError(f"Required date column '{col}' not found in DataFrame.")

    # Calculate difference in whole days
    # (Delivered - Scheduled): positive means delayed, negative means early
    date_diff = (df["Delivered to Client Date"] - df["Scheduled Delivery Date"]).dt.days

    # Identify rows where either date was missing or invalid
    missing_targets_mask = df["Delivered to Client Date"].isna() | df["Scheduled Delivery Date"].isna()
    missing_target_count = int(missing_targets_mask.sum())

    if missing_target_count > 0:
        logger.warning(
            f"Encountered {missing_target_count} records with missing/invalid target dates. "
            "Preserving these rows in cleaned data, setting delay_days and is_delayed to NaN."
        )
    else:
        logger.info("All records contain valid Scheduled and Delivered delivery dates.")

    # Assign delay_days
    df["delay_days"] = date_diff

    # Assign is_delayed: 1 for delayed (>0 days), 0 for on-time or early (<=0 days), NaN if missing
    df["is_delayed"] = np.where(
        missing_targets_mask,
        np.nan,
        np.where(df["delay_days"] > 0, 1.0, 0.0)
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the end-to-end data cleaning and target derivation pipeline.

    1. Parses all dates safely with placeholder coercion.
    2. Cleans numeric columns with safe float coercion.
    3. Derives targets 'delay_days' and 'is_delayed'.
    4. Preserves all rows without dropping unlabelled rows or inventing values.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with standardized types and derived targets.
    """
    logger.info(f"Starting data cleaning on {len(df)} rows and {len(df.columns)} columns.")

    df_cleaned = parse_dates(df)
    df_cleaned = clean_numeric_fields(df_cleaned)
    df_cleaned = derive_targets(df_cleaned)

    logger.info(
        f"Data cleaning complete. Output contains {len(df_cleaned)} rows and {len(df_cleaned.columns)} columns."
    )
    return df_cleaned


def main(
    input_path: str = "data/raw/SCMS_Delivery_History_Dataset.csv",
    output_path: str = "data/processed/scms_cleaned.csv"
) -> None:
    """
    CLI entry point for running data preprocessing manually.

    Reads the raw dataset from input_path, runs clean_data(), and writes
    the result to output_path. Does not execute automatically upon import.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path where processed CSV will be saved.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    logger.info(f"Loading raw dataset from: {input_path}")
    raw_df = pd.read_csv(input_path)

    cleaned_df = clean_data(raw_df)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Writing cleaned dataset to: {output_path}")
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(cleaned_df)} rows to {output_path}.")


if __name__ == "__main__":
    main()
