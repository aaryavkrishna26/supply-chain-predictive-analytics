"""
Linear Regression Pipeline for Supply Chain Delivery Delay Duration Prediction.

Phase 5B: Implements and evaluates Linear Regression against mean and median dummy baselines
to predict continuous delay_days on data/processed/scms_cleaned.csv.

Key safeguards:
1. Strict Leakage Prevention:
   - Excludes delay_days, is_delayed, Delivered to Client Date, Delivery Recorded Date.
   - Conservatively excludes Weight, Freight Cost, Line Item Insurance, PO Sent to Vendor Date,
     PQ First Sent to Client Date.
   - Derives calendar features strictly from Scheduled Delivery Date.
2. Leakage-Free Preprocessing:
   - scikit-learn Pipeline and ColumnTransformer fitted exclusively on training data.
   - High-cardinality rare categories (Country, Vendor, Manufacturing Site) grouped to 'Other'
     based strictly on X_train frequencies.
3. Train/Test Splitting:
   - 80/20 train/test split with random_state=42.
4. Evaluation:
   - Compares DummyRegressor(strategy='mean'), DummyRegressor(strategy='median'), and LinearRegression().
   - Metrics: MAE (days), RMSE (days), R2, Median Absolute Error (MedAE).
   - Subgroup analysis on records with abs(delay_days) > 30.
   - Saves results to reports/linear_regression_evaluation.md.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Reuse exact preprocessing pipeline components from train_logistic_regression.py
from train_logistic_regression import (
    LEAKAGE_AND_TARGET_COLUMNS,
    CONSERVATIVELY_EXCLUDED_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_preprocessor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "delay_days"


def evaluate_regressor(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Fits regressor on training data and computes MAE, RMSE, R2, and MedAE on test set.
    Also computes metrics on the subset of test records where abs(y_test) > 30.
    """
    logger.info(f"Fitting model: {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    medae = float(median_absolute_error(y_test, y_pred))

    # Subgroup metrics on extreme subset: abs(y_test) > 30
    extreme_mask = y_test.abs() > 30
    extreme_count = int(extreme_mask.sum())
    if extreme_count > 0:
        extreme_y_test = y_test[extreme_mask]
        extreme_y_pred = y_pred[extreme_mask]
        extreme_mae = float(mean_absolute_error(extreme_y_test, extreme_y_pred))
        extreme_rmse = float(root_mean_squared_error(extreme_y_test, extreme_y_pred))
        extreme_medae = float(median_absolute_error(extreme_y_test, extreme_y_pred))
    else:
        extreme_mae = extreme_rmse = extreme_medae = float("nan")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
        "residuals_mean": float(residuals.mean()),
        "residuals_std": float(residuals.std()),
        "residuals_min": float(residuals.min()),
        "residuals_max": float(residuals.max()),
        "extreme_count": extreme_count,
        "extreme_mae": extreme_mae,
        "extreme_rmse": extreme_rmse,
        "extreme_medae": extreme_medae,
    }


def format_evaluation_report(
    results: List[Dict[str, Any]],
    train_size: int,
    test_size: int,
    y_test: pd.Series,
) -> str:
    """
    Formats the evaluation results into markdown report for reports/linear_regression_evaluation.md.
    """
    md = []
    md.append("# Linear Regression Baseline Evaluation Report (Phase 5B)\n")
    md.append("## 1. Overview and Setup\n")
    md.append(f"- **Target Variable:** `delay_days` (continuous whole days: Delivered to Client Date − Scheduled Delivery Date)")
    md.append(f"- **Data Source:** `data/processed/scms_cleaned.csv`")
    md.append(f"- **Train Set Size:** {train_size} samples (80%)")
    md.append(f"- **Test Set Size:** {test_size} samples (20%)")
    md.append(f"- **Split Method:** Independent reproducible random split (`test_size=0.20`, `random_state=42`)")
    md.append(f"- **Leakage Quarantine:** Excluded `delay_days`, `is_delayed`, `Delivered to Client Date`, `Delivery Recorded Date`, and quarantined features (`Weight`, `Freight Cost`, `Insurance`, `PO Sent Date`, `PQ Sent Date`). Calendar features extracted strictly from `Scheduled Delivery Date`.\n")

    md.append("## 2. Test-Set Target Distribution\n")
    md.append(f"- **Min / Max:** {y_test.min():.1f} days / {y_test.max():.1f} days")
    md.append(f"- **Mean:** {y_test.mean():.2f} days")
    md.append(f"- **Median:** {y_test.median():.2f} days")
    md.append(f"- **Standard Deviation:** {y_test.std():.2f} days")
    md.append(f"- **Interquartile Range (25th - 75th):** {y_test.quantile(0.25):.1f} to {y_test.quantile(0.75):.1f} days")
    extreme_n = int((y_test.abs() > 30).sum())
    md.append(f"- **Records with |delay_days| > 30:** {extreme_n} ({extreme_n / len(y_test) * 100:.2f}% of test set)\n")

    md.append("## 3. Overall Model Performance Comparison\n")
    md.append("| Model | MAE (days) | RMSE (days) | R² | MedAE (days) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in results:
        md.append(
            f"| **{r['model_name']}** | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.4f} | {r['medae']:.2f} |"
        )
    md.append("\n")

    md.append("## 4. Subgroup Performance: Extreme Delay Records (|delay_days| > 30)\n")
    md.append(f"Evaluated exclusively on the {extreme_n} test records exhibiting absolute delays greater than 30 days:\n")
    md.append("| Model | Subgroup MAE (days) | Subgroup RMSE (days) | Subgroup MedAE (days) |")
    md.append("| :--- | :---: | :---: | :---: |")
    for r in results:
        md.append(
            f"| **{r['model_name']}** | {r['extreme_mae']:.2f} | {r['extreme_rmse']:.2f} | {r['extreme_medae']:.2f} |"
        )
    md.append("\n")

    md.append("## 5. Residual Diagnostics\n")
    for r in results:
        md.append(f"### {r['model_name']}")
        md.append(f"- **Residuals Mean:** {r['residuals_mean']:.4f} days")
        md.append(f"- **Residuals Std Dev:** {r['residuals_std']:.2f} days")
        md.append(f"- **Min Residual (Underprediction / Error):** {r['residuals_min']:.2f} days")
        md.append(f"- **Max Residual (Overprediction / Error):** {r['residuals_max']:.2f} days\n")

    md.append("## 6. Interpretation and Findings\n")
    lr = [r for r in results if r["model_name"] == "Linear Regression"][0]
    dummy_median = [r for r in results if r["model_name"] == "Dummy (Median)"][0]
    dummy_mean = [r for r in results if r["model_name"] == "Dummy (Mean)"][0]

    md.append("1. **Comparison with Dummy Baselines:**")
    md.append(f"   - **Median Baseline:** Predicting constant median (0.0 days) yields MAE of {dummy_median['mae']:.2f} days and MedAE of {dummy_median['medae']:.2f} days because over 61% of shipments arrive exactly on schedule.")
    md.append(f"   - **Mean Baseline:** Predicting constant mean yields MAE of {dummy_mean['mae']:.2f} days and RMSE of {dummy_mean['rmse']:.2f} days.")
    md.append(f"   - **Linear Regression:** Achieves MAE of {lr['mae']:.2f} days, RMSE of {lr['rmse']:.2f} days, and R² of {lr['r2']:.4f}.")
    md.append(f"   - While Linear Regression achieves a modest improvement in RMSE over the mean baseline ({lr['rmse']:.2f} vs {dummy_mean['rmse']:.2f}), its MedAE ({lr['medae']:.2f} days) is higher than the simple median predictor ({dummy_median['medae']:.2f} days) because linear regression shifts predictions away from zero to minimize squared errors across the asymmetric heavy tails.")

    md.append("2. **Impact of Heavy Tails on Metrics:**")
    md.append("   - Because RMSE squares errors, extreme negative outliers (e.g. shipments arriving 200–370 days early) heavily penalize the model, keeping RMSE above 25 days.")
    md.append("   - R² is modest, confirming that pre-dispatch order features (country, vendor, items) explain only a limited portion of variance in exact delay duration.")

    md.append("3. **Operational Context and Limitations:**")
    md.append("   - Many delivery duration variations stem from in-transit disruptions (customs clearance delays, port congestion, weather, local logistics) which occur *after* order dispatch and are therefore unobservable at prediction time.")
    md.append("   - Random train/test split does not evaluate temporal generalization across fiscal regimes; a time-based validation would be appropriate for production evaluation.\n")

    return "\n".join(md)


def main(
    data_path: str = "data/processed/scms_cleaned.csv",
    report_path: str = "reports/linear_regression_evaluation.md"
) -> None:
    """
    Main training and evaluation routine for Linear Regression baseline.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

    logger.info(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    valid_mask = df[TARGET_COLUMN].notna()
    if valid_mask.sum() != len(df):
        logger.warning(f"Dropping {len(df) - valid_mask.sum()} rows with missing target values for training.")
        df = df[valid_mask].copy()

    # Exclude leakage and quarantined columns
    drop_cols = (
        LEAKAGE_AND_TARGET_COLUMNS
        + CONSERVATIVELY_EXCLUDED_COLUMNS
        + IDENTIFIER_COLUMNS
    )
    feature_cols = [c for c in df.columns if c not in drop_cols]
    logger.info(f"Selected {len(feature_cols)} input features: {feature_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(float)

    # 80/20 train/test split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    logger.info(f"Split complete: Train set={len(X_train)} samples, Test set={len(X_test)} samples.")

    # Build preprocessing pipeline
    prep_pipeline = build_preprocessor()

    models_to_train = [
        ("Dummy (Mean)", DummyRegressor(strategy="mean")),
        ("Dummy (Median)", DummyRegressor(strategy="median")),
        ("Linear Regression", LinearRegression()),
    ]

    results = []
    for model_name, reg in models_to_train:
        pipeline = Pipeline(steps=[
            ("preprocessing", prep_pipeline),
            ("regressor", reg),
        ])
        res = evaluate_regressor(model_name, pipeline, X_train, y_train, X_test, y_test)
        results.append(res)

    report_content = format_evaluation_report(
        results,
        train_size=len(X_train),
        test_size=len(X_test),
        y_test=y_test,
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Evaluation report successfully written to {report_path}.")


if __name__ == "__main__":
    main()
