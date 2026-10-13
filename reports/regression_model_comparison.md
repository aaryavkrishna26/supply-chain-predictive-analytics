# Regression Model Improvement and Comparison Report (Phase 5C)

## 1. Overview and Setup

- **Target Variable:** `delay_days` (continuous whole days: Delivered to Client Date − Scheduled Delivery Date)
- **Dataset:** `data/processed/scms_cleaned.csv`
- **Train Set Size:** 8259 samples (80%)
- **Test Set Size:** 2065 samples (20%)
- **Split:** Reproducible random split (`test_size=0.20`, `random_state=42`) identical to Phase 5B
- **Leakage Safeguards:** Identical feature exclusions and column transformations fitted exclusively on training data.

## 2. Test-Set Performance Summary Table

| Model | MAE (days) | RMSE (days) | R² | MedAE (days) | Convergence Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Mean)** | 14.48 | 28.60 | -0.0000 | 6.01 | Converged |
| **Dummy (Median)** | 11.47 | 29.24 | -0.0450 | 0.00 | Converged |
| **Linear Regression** | 13.98 | 26.90 | 0.1153 | 6.46 | Converged |
| **Ridge (alpha=1.0)** | 13.96 | 26.90 | 0.1156 | 6.46 | Converged |
| **Huber Regressor** | 11.13 | 28.68 | -0.0053 | 0.00 | Warning: lbfgs failed to converge after 1000... |


## 3. Subgroup Performance: Extreme Delays (|delay_days| > 30)

Evaluated on 229 test records (11.09% of test set) with absolute delays > 30 days:

| Model | Subgroup MAE (days) | Subgroup RMSE (days) | Subgroup MedAE (days) |
| :--- | :---: | :---: | :---: |
| **Dummy (Mean)** | 69.98 | 81.99 | 57.99 |
| **Dummy (Median)** | 73.10 | 84.63 | 62.00 |
| **Linear Regression** | 61.51 | 74.19 | 50.56 |
| **Ridge (alpha=1.0)** | 61.54 | 74.21 | 50.60 |
| **Huber Regressor** | 71.18 | 83.31 | 59.00 |


## 4. Best Model Identification by Metric

- **Lowest MAE (Overall):** **Huber Regressor** (11.13 days)
- **Lowest RMSE (Overall):** **Ridge (alpha=1.0)** (26.90 days)
- **Highest R² (Overall):** **Ridge (alpha=1.0)** (0.1156)
- **Lowest MedAE (Overall):** **Dummy (Median)** (0.00 days)
- **Lowest Subgroup MAE (|delay| > 30):** **Linear Regression** (61.51 days)

## 5. Comparative Analysis and Findings

1. **Ordinary Least Squares vs. Ridge Regularization:**
   - Ridge regression (`alpha=1.0`) behaves very similarly to unregularized Linear Regression (RMSE: 26.90 vs 26.90; MAE: 13.96 vs 13.98; R²: 0.1156 vs 0.1153).
   - Because the feature matrix has 20 candidate features expanding to ~70 one-hot columns on >8,000 training rows, multicollinearity is mild and L2 regularization with default alpha has minimal impact.
2. **Huber Regressor Behavior:**
   - Huber Regressor uses a hybrid loss (squared loss for small errors, linear loss for large errors), making it robust to heavy outlier tails.
   - Result: Huber achieves MAE of 11.13 days, RMSE of 28.68 days, and MedAE of 0.00 days.
   - By not squaring large errors, Huber avoids skewing its predictions toward extreme outliers, yielding lower median error than OLS.
3. **Comparison Against Dummy Baselines:**
   - The simple median predictor (`0.0 days`) maintains the lowest overall MAE (11.47 days) and MedAE (0.00 days) because >61% of shipments arrive exactly on schedule.
   - However, on the extreme subgroup (|delay| > 30), trained models significantly outperform the median baseline (e.g. Linear/Ridge MAE ~61.5 days vs Median Dummy MAE 73.10 days).
4. **Methodological Note on Test-Set Model Selection:**
   - Repeatedly evaluating candidate models on the single held-out test set risks selection bias. These comparisons demonstrate sensitivity to loss functions, but final model validation in a production pipeline should be cross-validated across training folds before final evaluation on a held-out test split.
