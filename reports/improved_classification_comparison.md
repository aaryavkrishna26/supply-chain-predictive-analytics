# Improved Classification Model Comparison Report

## 1. Experimental Setup

- **Script:** `src/improve_classification_models.py`
- **Data source:** `data/processed/scms_cleaned.csv`
- **Total rows:** 10,324 (1,186 Late [11.49%], 9,138 On-Time [88.51%])
- **Train set:** 8,259 samples (80%, stratified)
- **Test set:** 2,065 samples (20%, stratified, 237 Late [11.48%])
- **Split:** `train_test_split(test_size=0.20, random_state=42, stratify=y)` - identical to prior phases
- **Leakage exclusions:** Same as Phases 4-5 (delivery dates, delay targets, post-hoc financial fields)
- **Tuning method:** RandomizedSearchCV on X_train only (n_iter=20, 3-fold CV, scoring=ROC-AUC)
- **Threshold selection:** 5-fold CV on X_train only - test set not inspected until tuning complete

## 2. Model Performance Summary

| Model | Thr | Accuracy | Precision | Recall | F1 | Avg Prec | ROC-AUC | FPR | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.50 | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.5000 | 0.0000 | 0 | 0 | 237 | 1828 |
| **Logistic Regression (Default)** | 0.50 | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.3099 | 0.8062 | 0.0000 | 0 | 0 | 237 | 1828 |
| **Logistic Regression (Balanced)** | 0.50 | 0.6305 | 0.2208 | 0.8776 | 0.3528 | 0.3060 | 0.8043 | 0.4015 | 208 | 734 | 29 | 1094 |
| **Logistic Regression (Default, thr=0.20)** | 0.20 | 0.7942 | 0.2939 | 0.5654 | 0.3867 | 0.3099 | 0.8062 | 0.1761 | 134 | 322 | 103 | 1506 |
| **RandomForest (thr=0.50)** | 0.50 | 0.8751 | 0.4516 | 0.4135 | 0.4317 | 0.4006 | 0.8576 | 0.0651 | 98 | 119 | 139 | 1709 |
| **RandomForest (thr=0.40, CV-tuned)** | 0.40 | 0.8300 | 0.3575 | 0.6034 | 0.4490 | 0.4006 | 0.8576 | 0.1406 | 143 | 257 | 94 | 1571 |
| **ExtraTrees (thr=0.50)** | 0.50 | 0.7738 | 0.2932 | 0.6878 | 0.4111 | 0.3769 | 0.8442 | 0.2150 | 163 | 393 | 74 | 1435 |
| **ExtraTrees (thr=0.50, CV-tuned)** | 0.50 | 0.7738 | 0.2932 | 0.6878 | 0.4111 | 0.3769 | 0.8442 | 0.2150 | 163 | 393 | 74 | 1435 |

## 3. Confusion Matrix Breakdown

### Dummy (Most Frequent) (threshold=0.50)
- TN=1828, FP=0, FN=237, TP=0
- **Delays correctly caught:** 0 / 237 (0.0%)
- **False alarms on on-time shipments:** 0 / 1828 (0.0%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Default) (threshold=0.50)
- TN=1828, FP=0, FN=237, TP=0
- **Delays correctly caught:** 0 / 237 (0.0%)
- **False alarms on on-time shipments:** 0 / 1828 (0.0%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.89      1.00      0.94      1828
           1       0.00      0.00      0.00       237

    accuracy                           0.89      2065
   macro avg       0.44      0.50      0.47      2065
weighted avg       0.78      0.89      0.83      2065
```

### Logistic Regression (Balanced) (threshold=0.50)
- TN=1094, FP=734, FN=29, TP=208
- **Delays correctly caught:** 208 / 237 (87.8%)
- **False alarms on on-time shipments:** 734 / 1828 (40.2%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.97      0.60      0.74      1828
           1       0.22      0.88      0.35       237

    accuracy                           0.63      2065
   macro avg       0.60      0.74      0.55      2065
weighted avg       0.89      0.63      0.70      2065
```

### Logistic Regression (Default, thr=0.20) (threshold=0.20)
- TN=1506, FP=322, FN=103, TP=134
- **Delays correctly caught:** 134 / 237 (56.5%)
- **False alarms on on-time shipments:** 322 / 1828 (17.6%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.94      0.82      0.88      1828
           1       0.29      0.57      0.39       237

    accuracy                           0.79      2065
   macro avg       0.61      0.69      0.63      2065
weighted avg       0.86      0.79      0.82      2065
```

### RandomForest (thr=0.50) (threshold=0.50)
- TN=1709, FP=119, FN=139, TP=98
- **Delays correctly caught:** 98 / 237 (41.4%)
- **False alarms on on-time shipments:** 119 / 1828 (6.5%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.92      0.93      0.93      1828
           1       0.45      0.41      0.43       237

    accuracy                           0.88      2065
   macro avg       0.69      0.67      0.68      2065
weighted avg       0.87      0.88      0.87      2065
```

### RandomForest (thr=0.40, CV-tuned) (threshold=0.40)
- TN=1571, FP=257, FN=94, TP=143
- **Delays correctly caught:** 143 / 237 (60.3%)
- **False alarms on on-time shipments:** 257 / 1828 (14.1%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.94      0.86      0.90      1828
           1       0.36      0.60      0.45       237

    accuracy                           0.83      2065
   macro avg       0.65      0.73      0.67      2065
weighted avg       0.88      0.83      0.85      2065
```

### ExtraTrees (thr=0.50) (threshold=0.50)
- TN=1435, FP=393, FN=74, TP=163
- **Delays correctly caught:** 163 / 237 (68.8%)
- **False alarms on on-time shipments:** 393 / 1828 (21.5%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.95      0.79      0.86      1828
           1       0.29      0.69      0.41       237

    accuracy                           0.77      2065
   macro avg       0.62      0.74      0.64      2065
weighted avg       0.88      0.77      0.81      2065
```

### ExtraTrees (thr=0.50, CV-tuned) (threshold=0.50)
- TN=1435, FP=393, FN=74, TP=163
- **Delays correctly caught:** 163 / 237 (68.8%)
- **False alarms on on-time shipments:** 393 / 1828 (21.5%)
- **Convergence:** Converged normally.
```text
precision    recall  f1-score   support

           0       0.95      0.79      0.86      1828
           1       0.29      0.69      0.41       237

    accuracy                           0.77      2065
   macro avg       0.62      0.74      0.64      2065
weighted avg       0.88      0.77      0.81      2065
```

## 4. Hyperparameter Tuning Summary

### RandomForestClassifier
Best parameters found via RandomizedSearchCV (n_iter=20, CV=3-fold, scoring=ROC-AUC):
- `classifier__n_estimators`: `600`
- `classifier__min_samples_split`: `5`
- `classifier__min_samples_leaf`: `2`
- `classifier__max_features`: `sqrt`
- `classifier__max_depth`: `30`
- `classifier__class_weight`: `balanced_subsample`

### ExtraTreesClassifier
Best parameters found via RandomizedSearchCV (n_iter=20, CV=3-fold, scoring=ROC-AUC):
- `classifier__n_estimators`: `400`
- `classifier__min_samples_split`: `2`
- `classifier__min_samples_leaf`: `2`
- `classifier__max_features`: `log2`
- `classifier__max_depth`: `30`
- `classifier__class_weight`: `balanced`

## 5. Training-Only CV Threshold Calibration

Thresholds selected using 5-fold cross-validated predict_proba on X_train. The test set was never used during threshold selection.

### Logistic Regression (Default)
Best threshold (F1-optimized on training CV): **0.20**

| Threshold | CV F1 |
| :---: | :---: |
| 0.10 | 0.3200 |
| 0.15 | 0.3410 |
| 0.20 | 0.3442 <- selected |
| 0.25 | 0.2947 |
| 0.30 | 0.1752 |
| 0.35 | 0.0702 |
| 0.40 | 0.0305 |
| 0.45 | 0.0042 |
| 0.50 | 0.0000 |

### RandomForestClassifier
Best threshold: **0.40**

| Threshold | CV F1 |
| :---: | :---: |
| 0.10 | 0.3132 |
| 0.15 | 0.3354 |
| 0.20 | 0.3646 |
| 0.25 | 0.3943 |
| 0.30 | 0.4217 |
| 0.35 | 0.4291 |
| 0.40 | 0.4387 <- selected |
| 0.45 | 0.4187 |
| 0.50 | 0.3820 |

### ExtraTreesClassifier
Best threshold: **0.50**

| Threshold | CV F1 |
| :---: | :---: |
| 0.10 | 0.2857 |
| 0.15 | 0.3023 |
| 0.20 | 0.3146 |
| 0.25 | 0.3278 |
| 0.30 | 0.3439 |
| 0.35 | 0.3602 |
| 0.40 | 0.3785 |
| 0.45 | 0.3924 |
| 0.50 | 0.4139 <- selected |

## 6. Model-Selection Analysis

### Objective

Primary objective: catch genuine delay events (high recall) while managing false alarm volume. Accuracy alone is misleading given 11.49% positive prevalence.

### Candidate Assessment

- **Highest recall:** `Logistic Regression (Balanced)` - Recall=0.8776, TP=208, FP=734, FPR=0.4015
- **Best F1 (practical balance):** `RandomForest (thr=0.40, CV-tuned)` - F1=0.4490, Recall=0.6034, Precision=0.3575
- **Best ROC-AUC (ranking quality):** `RandomForest (thr=0.50)` - ROC-AUC=0.8576

### False-Positive Rate Context

FPR is the fraction of on-time shipments incorrectly flagged as delayed. A high FPR generates alert fatigue. Optimal model depends on the stockout cost vs inspection labor cost ratio of the organization.

## 7. Limitations and Evaluation Integrity

- Hyperparameter tuning performed exclusively on X_train using cross-validation.
- Decision thresholds calibrated using training-only CV probabilities.
- Held-out test set (20%, 2,065 samples) used only for final evaluation.
- Same 80/20 split as prior phases ensures direct comparability.
- Random split does not evaluate temporal/chronological generalization.
- Multiple models on same test set introduces mild post-hoc selection risk.

## 8. Tuning Log Notes

- LR (Default) CV threshold scan (F1): thr=0.10->F1=0.3200, thr=0.15->F1=0.3410, thr=0.20->F1=0.3442, thr=0.25->F1=0.2947, thr=0.30->F1=0.1752, thr=0.35->F1=0.0702, thr=0.40->F1=0.0305, thr=0.45->F1=0.0042, thr=0.50->F1=0.0000
- RandomForest best params: {'classifier__n_estimators': 600, 'classifier__min_samples_split': 5, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'sqrt', 'classifier__max_depth': 30, 'classifier__class_weight': 'balanced_subsample'}
- RandomForest CV threshold scan (F1): thr=0.10->F1=0.3132, thr=0.15->F1=0.3354, thr=0.20->F1=0.3646, thr=0.25->F1=0.3943, thr=0.30->F1=0.4217, thr=0.35->F1=0.4291, thr=0.40->F1=0.4387, thr=0.45->F1=0.4187, thr=0.50->F1=0.3820
- ExtraTrees best params: {'classifier__n_estimators': 400, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'log2', 'classifier__max_depth': 30, 'classifier__class_weight': 'balanced'}
- ExtraTrees CV threshold scan (F1): thr=0.10->F1=0.2857, thr=0.15->F1=0.3023, thr=0.20->F1=0.3146, thr=0.25->F1=0.3278, thr=0.30->F1=0.3439, thr=0.35->F1=0.3602, thr=0.40->F1=0.3785, thr=0.45->F1=0.3924, thr=0.50->F1=0.4139
