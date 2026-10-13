# Phase 5E: SGD Classification Comparison Report

## 1. Overview and Problem Setup

- **Target Variable:** `is_delayed` (Binary: 1 = Late, 0 = On-Time / Early)
- **Dataset:** `data/processed/scms_cleaned.csv`
- **Train Set Size:** 8259 samples (80%)
- **Test Set Size:** 2065 samples (20%, Stratified)
- **Class Prevalence:** 1186 Late (11.49%) vs 9138 On-Time (88.51%)
- **Leakage Quarantine:** Excluded `delay_days`, `is_delayed`, `Delivered to Client Date`, `Delivery Recorded Date`, and quarantined features (`Weight`, `Freight Cost`, `Insurance`, `PO Sent Date`, `PQ Sent Date`). Scaler and OneHotEncoder fitted only on training data.

## 2. Model Performance Summary Table (Held-Out Test Set)

| Model | Accuracy | Precision (Late) | Recall (Late) | F1-Score | PR-AUC | ROC-AUC | Convergence Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Most Frequent)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.5000 | Converged |
| **Logistic Regression (Default)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.3099 | 0.8062 | Converged |
| **Logistic Regression (Balanced)** | 0.6305 | 0.2208 | 0.8776 | 0.3528 | 0.3060 | 0.8043 | Converged |
| **SGDClassifier (Default, log_loss)** | 0.8673 | 0.3131 | 0.1308 | 0.1845 | 0.2910 | 0.7977 | Converged |
| **SGDClassifier (Balanced, log_loss)** | 0.6203 | 0.2154 | 0.8734 | 0.3456 | 0.2562 | 0.7822 | Converged |


## 3. Confusion Matrix Breakdown on Test Set

### Dummy (Most Frequent)
- **TN:** 1828 | **FP:** 0 | **FN:** 237 | **TP:** 0
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Default)
- **TN:** 1828 | **FP:** 0 | **FN:** 237 | **TP:** 0
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Balanced)
- **TN:** 1094 | **FP:** 734 | **FN:** 29 | **TP:** 208
```text
precision    recall  f1-score   support

           0       0.97      0.60      0.74      1828
           1       0.22      0.88      0.35       237

    accuracy                           0.63      2065
   macro avg       0.60      0.74      0.55      2065
weighted avg       0.89      0.63      0.70      2065
```

### SGDClassifier (Default, log_loss)
- **TN:** 1760 | **FP:** 68 | **FN:** 206 | **TP:** 31
```text
precision    recall  f1-score   support

           0       0.90      0.96      0.93      1828
           1       0.31      0.13      0.18       237

    accuracy                           0.87      2065
   macro avg       0.60      0.55      0.56      2065
weighted avg       0.83      0.87      0.84      2065
```

### SGDClassifier (Balanced, log_loss)
- **TN:** 1074 | **FP:** 754 | **FN:** 30 | **TP:** 207
```text
precision    recall  f1-score   support

           0       0.97      0.59      0.73      1828
           1       0.22      0.87      0.35       237

    accuracy                           0.62      2065
   macro avg       0.59      0.73      0.54      2065
weighted avg       0.89      0.62      0.69      2065
```

## 4. Internal Cross-Validation Threshold Analysis (Training Set Only)

To prevent test-set leakage and repeated tuning bias, thresholds were evaluated via 5-fold cross-validation strictly on `X_train` for default `SGDClassifier(loss='log_loss')`:

| Threshold | CV TP | CV FP | CV TN | CV FN | CV Accuracy | CV Precision | CV Recall | CV F1 | CV FPR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 748 | 3073 | 4237 | 201 | 0.6036 | 0.1958 | 0.7882 | 0.3136 | 0.4204 |
| **0.15** | 601 | 2233 | 5077 | 348 | 0.6875 | 0.2121 | 0.6333 | 0.3177 | 0.3055 |
| **0.20** | 443 | 1504 | 5806 | 506 | 0.7566 | 0.2275 | 0.4668 | 0.3059 | 0.2057 |
| **0.25** | 309 | 928 | 6382 | 640 | 0.8101 | 0.2498 | 0.3256 | 0.2827 | 0.1269 |
| **0.30** | 195 | 557 | 6753 | 754 | 0.8413 | 0.2593 | 0.2055 | 0.2293 | 0.0762 |
| **0.40** | 76 | 188 | 7122 | 873 | 0.8715 | 0.2879 | 0.0801 | 0.1253 | 0.0257 |
| **0.50** | 18 | 50 | 7260 | 931 | 0.8812 | 0.2647 | 0.0190 | 0.0354 | 0.0068 |


## 5. Comparative Analysis & Key Findings

1. **Why Accuracy Alone Is Misleading:**
   - Because only 11.49% of shipments are delayed, a naive Dummy classifier achieves 88.52% accuracy while identifying exactly **zero** delayed shipments (0.0% recall).
   - Unweighted models (default Logistic Regression and default SGDClassifier) similarly prioritize accuracy by predicting class 0 for virtually all samples at the standard 0.5 threshold.
2. **SGDClassifier vs. Standard Logistic Regression:**
   - When class-weighted (`class_weight='balanced'`), `SGDClassifier` successfully catches late shipments: Recall = **0.8734** (207/237 delays caught) with F1 = **0.3456** and ROC-AUC = **0.7822**.
   - Standard balanced Logistic Regression achieves Recall = **0.8776** (208/237) with ROC-AUC = **0.8043**.
   - SGD provides an efficient stochastic gradient alternative with comparable discriminatory capability (ROC-AUC ~0.80) while requiring substantially fewer memory resources for large datasets.
3. **Threshold Calibration Safeguard:**
   - Based on internal cross-validation, the peak F1 on training data occurs at threshold **0.15** (CV F1 = 0.3177, CV Recall = 0.6333).
   - Calibrating decision thresholds on training cross-validation guarantees that the held-out test set remains an unbiased benchmark of generalization.
