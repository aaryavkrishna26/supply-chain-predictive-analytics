"""
Phase 6A: Improved Classification Experiment - RandomForest & ExtraTrees vs Logistic Regression.

Compares:
  - DummyClassifier (most_frequent) - naive baseline
  - LogisticRegression (default)
  - LogisticRegression (class_weight=balanced)
  - RandomForestClassifier (tuned via 3-fold CV on training data only)
  - ExtraTreesClassifier (tuned via 3-fold CV on training data only)

Key safeguards:
  1. Identical leakage exclusions and feature definitions as all prior phases.
  2. Same stratified 80/20 train/test split (random_state=42) for direct baseline comparison.
  3. All preprocessing fitted exclusively on X_train.
  4. Hyperparameter search via RandomizedSearchCV on X_train only.
  5. Decision-threshold selection via 5-fold CV probabilities on X_train only.
  6. Final evaluation uses the held-out test set once, after all tuning is complete.
  7. Convergence warnings captured and reported.
  8. Results saved to reports/improved_classification_comparison.md.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from train_logistic_regression import (
    CONSERVATIVELY_EXCLUDED_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAKAGE_AND_TARGET_COLUMNS,
    build_preprocessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_delayed"

RF_PARAM_DIST: Dict[str, Any] = {
    "classifier__n_estimators": [200, 400, 600],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced", "balanced_subsample"],
}

ET_PARAM_DIST: Dict[str, Any] = {
    "classifier__n_estimators": [200, 400, 600],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": ["balanced", "balanced_subsample"],
}

N_ITER_SEARCH = 20
CV_FOLDS_TUNE = 3
CV_FOLDS_THRESHOLD = 5
RANDOM_STATE = 42
THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def evaluate_on_test(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
    convergence_note: str = "Converged normally.",
) -> Dict[str, Any]:
    has_proba = hasattr(pipeline[-1], "predict_proba")

    if has_proba:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
    else:
        y_proba = None
        roc_auc = float("nan")
        pr_auc = float("nan")

    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "convergence_note": convergence_note,
    }


def select_threshold_by_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimize_metric: str = "f1",
    n_folds: int = CV_FOLDS_THRESHOLD,
) -> Tuple[float, Dict[float, float]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_proba_cv = cross_val_predict(
        pipeline, X_train, y_train, cv=skf, method="predict_proba"
    )[:, 1]

    scores: Dict[float, float] = {}
    for thr in THRESHOLD_GRID:
        y_pred_thr = (y_proba_cv >= thr).astype(int)
        if optimize_metric == "f1":
            scores[thr] = float(f1_score(y_train, y_pred_thr, zero_division=0))
        elif optimize_metric == "recall":
            scores[thr] = float(recall_score(y_train, y_pred_thr, zero_division=0))

    best_threshold = max(scores, key=scores.get)
    logger.info(
        f"CV threshold scan ({optimize_metric}): best={best_threshold:.2f} "
        f"({optimize_metric}={scores[best_threshold]:.4f})"
    )
    return best_threshold, scores


def tune_ensemble(
    base_pipeline: Pipeline,
    param_dist: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Tuple[Pipeline, Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS_TUNE, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=skf,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    logger.info(f"Starting RandomizedSearchCV for {model_name} (n_iter={N_ITER_SEARCH}, cv={CV_FOLDS_TUNE})...")
    search.fit(X_train, y_train)
    logger.info(f"{model_name} best CV ROC-AUC={search.best_score_:.4f}")
    logger.info(f"{model_name} best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def main(
    data_path: str = "data/processed/scms_cleaned.csv",
    report_path: str = "reports/improved_classification_comparison.md",
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found.")

    valid_mask = df[TARGET_COLUMN].notna()
    if valid_mask.sum() != len(df):
        logger.warning(f"Dropping {len(df) - valid_mask.sum()} rows with missing target.")
        df = df[valid_mask].copy()

    drop_cols = (
        LEAKAGE_AND_TARGET_COLUMNS
        + CONSERVATIVELY_EXCLUDED_COLUMNS
        + IDENTIFIER_COLUMNS
    )
    feature_cols = [c for c in df.columns if c not in drop_cols]
    logger.info(f"Using {len(feature_cols)} input features: {feature_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)

    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    total = len(y)
    logger.info(
        f"Dataset: {total} rows | Late={pos_count} ({pos_count/total*100:.2f}%) | "
        f"OnTime={neg_count} ({neg_count/total*100:.2f}%)"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    test_pos = int((y_test == 1).sum())
    test_neg = int((y_test == 0).sum())
    logger.info(f"Test set: Late={test_pos} ({test_pos/len(y_test)*100:.2f}%) | OnTime={test_neg}")

    results: List[Dict[str, Any]] = []
    tuning_notes: List[str] = []

    # 1. DummyClassifier
    logger.info("--- Dummy Classifier ---")
    dummy_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ])
    dummy_pipe.fit(X_train, y_train)
    res = evaluate_on_test("Dummy (Most Frequent)", dummy_pipe, X_test, y_test, threshold=0.50)
    results.append(res)

    # 2. Logistic Regression (default, threshold 0.50)
    logger.info("--- Logistic Regression (Default, thr=0.50) ---")
    lr_default_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    conv_note_lr = "Converged normally."
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        lr_default_pipe.fit(X_train, y_train)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                conv_note_lr = str(warning.message)
    res = evaluate_on_test("Logistic Regression (Default)", lr_default_pipe, X_test, y_test,
                           threshold=0.50, convergence_note=conv_note_lr)
    results.append(res)

    # 3. Logistic Regression (balanced, threshold=0.50)
    logger.info("--- Logistic Regression (Balanced, thr=0.50) ---")
    lr_balanced_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=RANDOM_STATE)),
    ])
    conv_note_lrb = "Converged normally."
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        lr_balanced_pipe.fit(X_train, y_train)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                conv_note_lrb = str(warning.message)
    res = evaluate_on_test("Logistic Regression (Balanced)", lr_balanced_pipe, X_test, y_test,
                           threshold=0.50, convergence_note=conv_note_lrb)
    results.append(res)

    # 4. Logistic Regression (default) with CV-tuned threshold
    logger.info("--- Logistic Regression (Default, CV threshold tuning) ---")
    lr_thresh_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    conv_note_lrt = "Converged normally."
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        lr_thresh_pipe.fit(X_train, y_train)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                conv_note_lrt = str(warning.message)

    lr_best_thr, lr_thr_scores = select_threshold_by_cv(lr_thresh_pipe, X_train, y_train, "f1")
    tuning_notes.append(
        "LR (Default) CV threshold scan (F1): " +
        ", ".join([f"thr={t:.2f}->F1={s:.4f}" for t, s in lr_thr_scores.items()])
    )
    res = evaluate_on_test(
        f"Logistic Regression (Default, thr={lr_best_thr:.2f})",
        lr_thresh_pipe, X_test, y_test,
        threshold=lr_best_thr, convergence_note=conv_note_lrt,
    )
    results.append(res)

    # 5. RandomForestClassifier
    logger.info("--- RandomForestClassifier (tuned) ---")
    rf_base_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_best_pipe, rf_best_params = tune_ensemble(rf_base_pipe, RF_PARAM_DIST, X_train, y_train, "RandomForest")
    tuning_notes.append(f"RandomForest best params: {rf_best_params}")

    rf_best_thr, rf_thr_scores = select_threshold_by_cv(rf_best_pipe, X_train, y_train, "f1")
    tuning_notes.append(
        "RandomForest CV threshold scan (F1): " +
        ", ".join([f"thr={t:.2f}->F1={s:.4f}" for t, s in rf_thr_scores.items()])
    )

    res_50 = evaluate_on_test("RandomForest (thr=0.50)", rf_best_pipe, X_test, y_test, threshold=0.50)
    results.append(res_50)

    res_cv = evaluate_on_test(
        f"RandomForest (thr={rf_best_thr:.2f}, CV-tuned)",
        rf_best_pipe, X_test, y_test, threshold=rf_best_thr,
    )
    results.append(res_cv)

    # 6. ExtraTreesClassifier
    logger.info("--- ExtraTreesClassifier (tuned) ---")
    et_base_pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    et_best_pipe, et_best_params = tune_ensemble(et_base_pipe, ET_PARAM_DIST, X_train, y_train, "ExtraTrees")
    tuning_notes.append(f"ExtraTrees best params: {et_best_params}")

    et_best_thr, et_thr_scores = select_threshold_by_cv(et_best_pipe, X_train, y_train, "f1")
    tuning_notes.append(
        "ExtraTrees CV threshold scan (F1): " +
        ", ".join([f"thr={t:.2f}->F1={s:.4f}" for t, s in et_thr_scores.items()])
    )

    res_50_et = evaluate_on_test("ExtraTrees (thr=0.50)", et_best_pipe, X_test, y_test, threshold=0.50)
    results.append(res_50_et)

    res_cv_et = evaluate_on_test(
        f"ExtraTrees (thr={et_best_thr:.2f}, CV-tuned)",
        et_best_pipe, X_test, y_test, threshold=et_best_thr,
    )
    results.append(res_cv_et)

    # Generate report
    report_content = _format_report(
        results=results,
        tuning_notes=tuning_notes,
        train_size=len(X_train),
        test_size=len(X_test),
        test_pos=test_pos,
        test_neg=test_neg,
        total_rows=total,
        pos_count=pos_count,
        neg_count=neg_count,
        rf_best_params=rf_best_params,
        et_best_params=et_best_params,
        lr_thr_scores=lr_thr_scores,
        rf_thr_scores=rf_thr_scores,
        et_thr_scores=et_thr_scores,
        lr_best_thr=lr_best_thr,
        rf_best_thr=rf_best_thr,
        et_best_thr=et_best_thr,
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Classification report written to {report_path}")

    print("\n" + "=" * 90)
    print("CLASSIFICATION EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Model':<55} {'Recall':>7} {'Prec':>7} {'F1':>7} {'ROC-AUC':>9} {'FPR':>7} {'TP':>5} {'FP':>6}"
    print(header)
    print("-" * 90)
    for r in results:
        roc = r['roc_auc'] if not np.isnan(r['roc_auc']) else 0.0
        print(
            f"{r['model_name']:<55} {r['recall']:>7.4f} {r['precision']:>7.4f} "
            f"{r['f1']:>7.4f} {roc:>9.4f} {r['fpr']:>7.4f} "
            f"{r['TP']:>5d} {r['FP']:>6d}"
        )
    print("=" * 90)


def _format_report(
    results: List[Dict[str, Any]],
    tuning_notes: List[str],
    train_size: int,
    test_size: int,
    test_pos: int,
    test_neg: int,
    total_rows: int,
    pos_count: int,
    neg_count: int,
    rf_best_params: Dict[str, Any],
    et_best_params: Dict[str, Any],
    lr_thr_scores: Dict[float, float],
    rf_thr_scores: Dict[float, float],
    et_thr_scores: Dict[float, float],
    lr_best_thr: float,
    rf_best_thr: float,
    et_best_thr: float,
) -> str:
    md: List[str] = []
    md.append("# Improved Classification Model Comparison Report\n")
    md.append("## 1. Experimental Setup\n")
    md.append(f"- **Script:** `src/improve_classification_models.py`")
    md.append(f"- **Data source:** `data/processed/scms_cleaned.csv`")
    md.append(f"- **Total rows:** {total_rows:,} ({pos_count:,} Late [{pos_count/total_rows*100:.2f}%], "
              f"{neg_count:,} On-Time [{neg_count/total_rows*100:.2f}%])")
    md.append(f"- **Train set:** {train_size:,} samples (80%, stratified)")
    md.append(f"- **Test set:** {test_size:,} samples (20%, stratified, {test_pos} Late [{test_pos/test_size*100:.2f}%])")
    md.append(f"- **Split:** `train_test_split(test_size=0.20, random_state=42, stratify=y)` - identical to prior phases")
    md.append(f"- **Leakage exclusions:** Same as Phases 4-5 (delivery dates, delay targets, post-hoc financial fields)")
    md.append(f"- **Tuning method:** RandomizedSearchCV on X_train only (n_iter={N_ITER_SEARCH}, {CV_FOLDS_TUNE}-fold CV, scoring=ROC-AUC)")
    md.append(f"- **Threshold selection:** {CV_FOLDS_THRESHOLD}-fold CV on X_train only - test set not inspected until tuning complete\n")

    md.append("## 2. Model Performance Summary\n")
    md.append("| Model | Thr | Accuracy | Precision | Recall | F1 | Avg Prec | ROC-AUC | FPR | TP | FP | FN | TN |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        pr_auc_str = f"{r['pr_auc']:.4f}" if not (isinstance(r['pr_auc'], float) and np.isnan(r['pr_auc'])) else "N/A"
        roc_str = f"{r['roc_auc']:.4f}" if not (isinstance(r['roc_auc'], float) and np.isnan(r['roc_auc'])) else "N/A"
        md.append(
            f"| **{r['model_name']}** | {r['threshold']:.2f} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {pr_auc_str} | "
            f"{roc_str} | {r['fpr']:.4f} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} |"
        )
    md.append("")

    md.append("## 3. Confusion Matrix Breakdown\n")
    for r in results:
        md.append(f"### {r['model_name']} (threshold={r['threshold']:.2f})")
        md.append(f"- TN={r['TN']}, FP={r['FP']}, FN={r['FN']}, TP={r['TP']}")
        md.append(f"- **Delays correctly caught:** {r['TP']} / {test_pos} ({r['recall']*100:.1f}%)")
        md.append(f"- **False alarms on on-time shipments:** {r['FP']} / {test_neg} ({r['fpr']*100:.1f}%)")
        md.append(f"- **Convergence:** {r['convergence_note']}")
        md.append("```text")
        md.append(r["classification_report"].strip())
        md.append("```\n")

    md.append("## 4. Hyperparameter Tuning Summary\n")
    md.append("### RandomForestClassifier")
    md.append(f"Best parameters found via RandomizedSearchCV (n_iter={N_ITER_SEARCH}, CV={CV_FOLDS_TUNE}-fold, scoring=ROC-AUC):")
    for k, v in rf_best_params.items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")
    md.append("### ExtraTreesClassifier")
    md.append(f"Best parameters found via RandomizedSearchCV (n_iter={N_ITER_SEARCH}, CV={CV_FOLDS_TUNE}-fold, scoring=ROC-AUC):")
    for k, v in et_best_params.items():
        md.append(f"- `{k}`: `{v}`")
    md.append("")

    md.append("## 5. Training-Only CV Threshold Calibration\n")
    md.append("Thresholds selected using 5-fold cross-validated predict_proba on X_train. "
              "The test set was never used during threshold selection.\n")

    md.append("### Logistic Regression (Default)")
    md.append(f"Best threshold (F1-optimized on training CV): **{lr_best_thr:.2f}**\n")
    md.append("| Threshold | CV F1 |")
    md.append("| :---: | :---: |")
    for thr, score in lr_thr_scores.items():
        marker = " <- selected" if thr == lr_best_thr else ""
        md.append(f"| {thr:.2f} | {score:.4f}{marker} |")
    md.append("")

    md.append("### RandomForestClassifier")
    md.append(f"Best threshold: **{rf_best_thr:.2f}**\n")
    md.append("| Threshold | CV F1 |")
    md.append("| :---: | :---: |")
    for thr, score in rf_thr_scores.items():
        marker = " <- selected" if thr == rf_best_thr else ""
        md.append(f"| {thr:.2f} | {score:.4f}{marker} |")
    md.append("")

    md.append("### ExtraTreesClassifier")
    md.append(f"Best threshold: **{et_best_thr:.2f}**\n")
    md.append("| Threshold | CV F1 |")
    md.append("| :---: | :---: |")
    for thr, score in et_thr_scores.items():
        marker = " <- selected" if thr == et_best_thr else ""
        md.append(f"| {thr:.2f} | {score:.4f}{marker} |")
    md.append("")

    md.append("## 6. Model-Selection Analysis\n")
    md.append("### Objective\n")
    md.append("Primary objective: catch genuine delay events (high recall) while managing false alarm volume. "
              "Accuracy alone is misleading given 11.49% positive prevalence.\n")
    md.append("### Candidate Assessment\n")

    valid_results = [r for r in results if not np.isnan(r["roc_auc"])]
    high_recall_model = max(valid_results, key=lambda r: (r["recall"], r["f1"]))
    best_f1_model = max(valid_results, key=lambda r: r["f1"])
    best_roc_model = max(valid_results, key=lambda r: r["roc_auc"])

    md.append(f"- **Highest recall:** `{high_recall_model['model_name']}` - "
              f"Recall={high_recall_model['recall']:.4f}, TP={high_recall_model['TP']}, FP={high_recall_model['FP']}, FPR={high_recall_model['fpr']:.4f}")
    md.append(f"- **Best F1 (practical balance):** `{best_f1_model['model_name']}` - "
              f"F1={best_f1_model['f1']:.4f}, Recall={best_f1_model['recall']:.4f}, Precision={best_f1_model['precision']:.4f}")
    md.append(f"- **Best ROC-AUC (ranking quality):** `{best_roc_model['model_name']}` - "
              f"ROC-AUC={best_roc_model['roc_auc']:.4f}")
    md.append("")

    md.append("### False-Positive Rate Context\n")
    md.append("FPR is the fraction of on-time shipments incorrectly flagged as delayed. "
              "A high FPR generates alert fatigue. Optimal model depends on the "
              "stockout cost vs inspection labor cost ratio of the organization.\n")

    md.append("## 7. Limitations and Evaluation Integrity\n")
    md.append("- Hyperparameter tuning performed exclusively on X_train using cross-validation.")
    md.append("- Decision thresholds calibrated using training-only CV probabilities.")
    md.append("- Held-out test set (20%, 2,065 samples) used only for final evaluation.")
    md.append("- Same 80/20 split as prior phases ensures direct comparability.")
    md.append("- Random split does not evaluate temporal/chronological generalization.")
    md.append("- Multiple models on same test set introduces mild post-hoc selection risk.\n")

    md.append("## 8. Tuning Log Notes\n")
    for note in tuning_notes:
        md.append(f"- {note}")
    md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    main()
