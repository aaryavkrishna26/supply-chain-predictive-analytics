# Logistic Regression Baseline Evaluation Report

## 1. Evaluation Overview

- **Dataset Source:** `data/processed/scms_cleaned.csv`
- **Train Set Size:** 8259 samples (80%)
- **Test Set Size:** 2065 samples (20%, Stratified)
- **Random Seed:** 42
- **Positive Class (Late / is_delayed = 1):** 1186 (11.49%)
- **Negative Class (On-Time / is_delayed = 0):** 9138 (88.51%)

## 2. Model Performance Summary

| Model | Accuracy | Precision (Late) | Recall (Late) | F1-Score (Late) | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.5000 |
| **Logistic Regression (Default)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.3099 | 0.8062 |
| **Logistic Regression (Balanced)** | 0.6305 | 0.2208 | 0.8776 | 0.3528 | 0.3060 | 0.8043 |

## 3. Confusion Matrix Breakdown

### Dummy (Most Frequent)
- **True Negatives (TN):** 1828
- **False Positives (FP):** 0
- **False Negatives (FN):** 237
- **True Positives (TP):** 0
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Default)
- **True Negatives (TN):** 1828
- **False Positives (FP):** 0
- **False Negatives (FN):** 237
- **True Positives (TP):** 0
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Balanced)
- **True Negatives (TN):** 1094
- **False Positives (FP):** 734
- **False Negatives (FN):** 29
- **True Positives (TP):** 208
```text
precision    recall  f1-score   support

           0       0.97      0.60      0.74      1828
           1       0.22      0.88      0.35       237

    accuracy                           0.63      2065
   macro avg       0.60      0.74      0.55      2065
weighted avg       0.89      0.63      0.70      2065
```

## 4. Class Imbalance and Trade-off Analysis

- **Majority Class Baseline:** The dummy baseline highlights that predicting all non-delayed yields ~88.5% accuracy but zero delay recall.
- **Default vs. Balanced Logistic Regression:**
  - Unweighted Logistic Regression typically optimizes for accuracy by skewing towards class 0.
  - Balanced Logistic Regression weights errors inversely proportional to class frequencies (~7.7:1), prioritizing recall at the cost of precision.
- **Leakage Safeguards:** All models were trained strictly without actual delivery dates, delivery recording dates, delay days, or quarantined post-hoc financial columns.
