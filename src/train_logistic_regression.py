"""
Logistic Regression Baseline Pipeline for Supply Chain Delivery Delay Prediction.

This module implements a self-contained training and evaluation pipeline for
classifying shipment delays (is_delayed: 0 = on-time/early, 1 = late) using:
- data/processed/scms_cleaned.csv

Key design safeguards:
1. Strict Leakage Prevention:
   - Excludes actual delivery dates, recording dates, and delay targets.
   - Conservatively quarantines financial/post-hoc fields: Weight, Freight Cost,
     Insurance, PO Sent Date, PQ Sent Date.
2. Leakage-Free Preprocessing:
   - Uses scikit-learn Pipeline and ColumnTransformer.
   - Encoders, imputers, and scalers are fitted strictly on X_train.
   - Handles rare high-cardinality categories using training-set frequencies only.
3. Class Imbalance Handling & Models:
   - DummyClassifier (most_frequent) as a naive baseline.
   - LogisticRegression (default class_weight=None).
   - LogisticRegression (class_weight='balanced').
4. Comprehensive Metrics:
   - Accuracy, Precision, Recall, F1-Score, PR-AUC (Average Precision), ROC-AUC,
     and Confusion Matrices on held-out test data.
5. Report Generation:
   - Writes complete metrics and comparison to reports/logistic_regression_evaluation.md
     only when executed manually.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Target definition
TARGET_COLUMN = "is_delayed"

# Explicitly excluded columns to prevent leakage or post-hoc confounding
LEAKAGE_AND_TARGET_COLUMNS = [
    "is_delayed",
    "delay_days",
    "Delivered to Client Date",
    "Delivery Recorded Date",
]

# Conservatively excluded columns due to missingness (>25%) or post-event reconciliation risk
CONSERVATIVELY_EXCLUDED_COLUMNS = [
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)",
    "PO Sent to Vendor Date",
    "PQ First Sent to Client Date",
]

# Non-predictive identifier columns
IDENTIFIER_COLUMNS = [
    "ID",
    "Project Code",
    "PQ #",
    "PO / SO #",
    "ASN/DN #",
    "Item Description",  # Free-form high-cardinality text
]

# Selected candidate feature groups available at scheduling/order placement
NUMERIC_FEATURES = [
    "Unit of Measure (Per Pack)",
    "Line Item Quantity",
    "Line Item Value",
    "Pack Price",
    "Unit Price",
]

LOW_CARDINALITY_CATEGORICAL_FEATURES = [
    "Shipment Mode",
    "Fulfill Via",
    "Vendor INCO Term",
    "Product Group",
    "Sub Classification",
    "First Line Designation",
    "Dosage Form",
]

HIGH_CARDINALITY_CATEGORICAL_FEATURES = [
    "Country",
    "Vendor",
    "Manufacturing Site",
]

DATE_FEATURE = "Scheduled Delivery Date"


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts calendar components from Scheduled Delivery Date strictly without leakage.
    Produces scheduled_month, scheduled_quarter, and scheduled_day_of_week.
    """
    def __init__(self, date_col: str = "Scheduled Delivery Date"):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        extracted = pd.DataFrame(index=X.index)
        extracted["scheduled_month"] = dates.dt.month.fillna(0).astype(int)
        extracted["scheduled_quarter"] = dates.dt.quarter.fillna(0).astype(int)
        extracted["scheduled_day_of_week"] = dates.dt.dayofweek.fillna(0).astype(int)
        return extracted


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces infrequent categories with 'Other' based strictly on training frequencies.
    Fitted exclusively on X_train to prevent test data leakage.
    """
    def __init__(self, columns: List[str], top_n: int = 15):
        self.columns = columns
        self.top_n = top_n
        self.frequent_categories_: Dict[str, List[Any]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.frequent_categories_ = {}
        for col in self.columns:
            if col in X.columns:
                counts = X[col].astype(str).value_counts()
                top_cats = counts.head(self.top_n).index.tolist()
                self.frequent_categories_[col] = top_cats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns and col in self.frequent_categories_:
                top_cats = set(self.frequent_categories_[col])
                X_out[col] = X_out[col].astype(str).apply(
                    lambda val: val if val in top_cats else "Other"
                )
        return X_out


def build_preprocessor() -> Pipeline:
    """
    Constructs a composite scikit-learn Pipeline and ColumnTransformer.
    Applies:
    - RareCategoryEncoder to high-cardinality features.
    - ColumnTransformer with:
        * Numeric pipeline: SimpleImputer(median) + StandardScaler()
        * Categorical pipeline: SimpleImputer('Unknown') + OneHotEncoder(handle_unknown='ignore')
        * Date features: DateFeatureExtractor + StandardScaler()
    """
    all_cat_features = LOW_CARDINALITY_CATEGORICAL_FEATURES + HIGH_CARDINALITY_CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    date_transformer = Pipeline(steps=[
        ("extractor", DateFeatureExtractor(date_col=DATE_FEATURE)),
        ("scaler", StandardScaler()),
    ])

    preprocessor_ct = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, all_cat_features),
            ("date", date_transformer, [DATE_FEATURE]),
        ],
        remainder="drop"
    )

    full_prep_pipeline = Pipeline(steps=[
        ("rare_encoder", RareCategoryEncoder(columns=HIGH_CARDINALITY_CATEGORICAL_FEATURES, top_n=15)),
        ("column_transform", preprocessor_ct),
    ])

    return full_prep_pipeline


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Fits the pipeline strictly on training data and evaluates on test data.
    """
    logger.info(f"Training model: {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    # Predict probabilities for ROC-AUC and PR-AUC if available
    has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
    if has_proba:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def format_evaluation_report(
    results: List[Dict[str, Any]],
    train_size: int,
    test_size: int,
    class_balance_info: Dict[str, Any],
) -> str:
    """
    Generates a structured markdown evaluation report for reports/logistic_regression_evaluation.md.
    """
    md = []
    md.append("# Logistic Regression Baseline Evaluation Report\n")
    md.append("## 1. Evaluation Overview\n")
    md.append(f"- **Dataset Source:** `data/processed/scms_cleaned.csv`")
    md.append(f"- **Train Set Size:** {train_size} samples (80%)")
    md.append(f"- **Test Set Size:** {test_size} samples (20%, Stratified)")
    md.append(f"- **Random Seed:** 42")
    md.append(f"- **Positive Class (Late / is_delayed = 1):** {class_balance_info['pos_count']} ({class_balance_info['pos_rate']:.2f}%)")
    md.append(f"- **Negative Class (On-Time / is_delayed = 0):** {class_balance_info['neg_count']} ({class_balance_info['neg_rate']:.2f}%)\n")

    md.append("## 2. Model Performance Summary\n")
    md.append("| Model | Accuracy | Precision (Late) | Recall (Late) | F1-Score (Late) | PR-AUC | ROC-AUC |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        pr_str = f"{r['pr_auc']:.4f}" if not np.isnan(r['pr_auc']) else "N/A"
        roc_str = f"{r['roc_auc']:.4f}" if not np.isnan(r['roc_auc']) else "N/A"
        md.append(
            f"| **{r['model_name']}** | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {pr_str} | {roc_str} |"
        )
    md.append("")

    md.append("## 3. Confusion Matrix Breakdown\n")
    for r in results:
        cm = r["confusion_matrix"]
        md.append(f"### {r['model_name']}")
        md.append(f"- **True Negatives (TN):** {cm['TN']}")
        md.append(f"- **False Positives (FP):** {cm['FP']}")
        md.append(f"- **False Negatives (FN):** {cm['FN']}")
        md.append(f"- **True Positives (TP):** {cm['TP']}")
        md.append("```text")
        md.append(r["classification_report"].strip())
        md.append("```\n")

    md.append("## 4. Class Imbalance and Trade-off Analysis\n")
    md.append("- **Majority Class Baseline:** The dummy baseline highlights that predicting all non-delayed yields ~88.5% accuracy but zero delay recall.")
    md.append("- **Default vs. Balanced Logistic Regression:**")
    md.append("  - Unweighted Logistic Regression typically optimizes for accuracy by skewing towards class 0.")
    md.append("  - Balanced Logistic Regression weights errors inversely proportional to class frequencies (~7.7:1), prioritizing recall at the cost of precision.")
    md.append("- **Leakage Safeguards:** All models were trained strictly without actual delivery dates, delivery recording dates, delay days, or quarantined post-hoc financial columns.\n")

    return "\n".join(md)


def main(
    data_path: str = "data/processed/scms_cleaned.csv",
    report_path: str = "reports/logistic_regression_evaluation.md"
) -> None:
    """
    Main training and evaluation routine.
    Loads data, validates target, creates train/test split, trains models,
    and writes the evaluation report.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}. Run preprocessing first.")

    logger.info(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate target presence and integrity
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    valid_mask = df[TARGET_COLUMN].notna()
    if valid_mask.sum() != len(df):
        logger.warning(f"Dropping {len(df) - valid_mask.sum()} rows with missing target values for training.")
        df = df[valid_mask].copy()

    # Separate features and target
    drop_cols = (
        LEAKAGE_AND_TARGET_COLUMNS
        + CONSERVATIVELY_EXCLUDED_COLUMNS
        + IDENTIFIER_COLUMNS
    )
    feature_cols = [c for c in df.columns if c not in drop_cols]
    logger.info(f"Selected {len(feature_cols)} input features: {feature_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Class balance statistics
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    total_count = len(y)
    class_balance_info = {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_rate": (pos_count / total_count) * 100,
        "neg_rate": (neg_count / total_count) * 100,
    }
    logger.info(
        f"Target distribution: 0={neg_count} ({class_balance_info['neg_rate']:.2f}%), "
        f"1={pos_count} ({class_balance_info['pos_rate']:.2f}%)"
    )

    # Stratified train/test split (80/20) with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Split complete: Train set={len(X_train)} samples, Test set={len(X_test)} samples.")

    # Base preprocessing pipeline
    prep_pipeline = build_preprocessor()

    # Define model configurations
    models_to_train = [
        ("Dummy (Most Frequent)", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression (Default)", LogisticRegression(max_iter=1000, random_state=42)),
        ("Logistic Regression (Balanced)", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]

    results = []
    for model_name, clf in models_to_train:
        # Build composite pipeline with prep + classifier
        model_pipe = Pipeline(steps=[
            ("preprocessing", prep_pipeline),
            ("classifier", clf),
        ])
        res = evaluate_model(model_name, model_pipe, X_train, y_train, X_test, y_test)
        results.append(res)

    # Generate and write evaluation report
    report_content = format_evaluation_report(
        results,
        train_size=len(X_train),
        test_size=len(X_test),
        class_balance_info=class_balance_info
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Evaluation report successfully written to {report_path}.")


if __name__ == "__main__":
    main()
