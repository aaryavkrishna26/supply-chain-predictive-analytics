# Phase 5D: SGD Regression Comparison Report

## 1. Overview and Setup

- **Target Variable:** `delay_days` (continuous days: Delivered to Client Date − Scheduled Delivery Date)
- **Data Source:** `data/processed/scms_cleaned.csv`
- **Train Set Size:** 8259 samples (80%)
- **Test Set Size:** 2065 samples (20%)
- **Split:** Reproducible random split (`test_size=0.20`, `random_state=42`)
- **Leakage Prevention:** Strictly excluded target labels and actual delivery dates. Features processed with StandardScaler and OneHotEncoder fitted only on training data.

## 2. SGDRegressor Configuration Justification

- **Loss:** `loss='squared_error'` (standard OLS objective via stochastic gradient descent).
- **Penalty:** `penalty='l2'` with `alpha=0.0001` (conservative L2 shrinkage matching standard Ridge).
- **Optimization:** `learning_rate='invscaling'` with `eta0=0.01` and `power_t=0.25` (standard decay schedule for smooth convergence on scaled tabular data).
- **Iterations & Tolerance:** `max_iter=2000`, `tol=1e-3` (allows adequate epoch passes without early truncation).
- **Random State:** `random_state=42` for reproducibility.

## 3. Test-Set Performance Comparison Table

| Model | MAE (days) | RMSE (days) | R² | MedAE (days) | Convergence Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Mean)** | 14.48 | 28.60 | -0.0000 | 6.01 | Converged |
| **Dummy (Median)** | 11.47 | 29.24 | -0.0450 | 0.00 | Converged |
| **Linear Regression** | 13.98 | 26.90 | 0.1153 | 6.46 | Converged |
| **Ridge (alpha=1.0)** | 13.96 | 26.90 | 0.1156 | 6.46 | Converged |
| **SGDRegressor (squared_error, l2)** | 14.16 | 26.95 | 0.1122 | 6.74 | Converged |
| *Huber Regressor (Phase 5C Context)* | *11.13* | *28.68* | *-0.0053* | *0.00* | *Warning: L-BFGS max_iter=1000 limit* |


## 4. Subgroup Performance: Extreme Delays (|delay_days| > 30)

Evaluated on 229 test records (11.09% of test set) with absolute delays > 30 days:

| Model | Subgroup MAE (days) | Subgroup RMSE (days) | Subgroup MedAE (days) |
| :--- | :---: | :---: | :---: |
| **Dummy (Mean)** | 69.98 | 81.99 | 57.99 |
| **Dummy (Median)** | 73.10 | 84.63 | 62.00 |
| **Linear Regression** | 61.51 | 74.19 | 50.56 |
| **Ridge (alpha=1.0)** | 61.54 | 74.21 | 50.60 |
| **SGDRegressor (squared_error, l2)** | 61.42 | 74.08 | 49.90 |
| *Huber Regressor (Phase 5C Context)* | *71.18* | *83.31* | *59.00* |


## 5. Key Findings & Discussion

1. **Convergence and Numerical Stability:**
   - `SGDRegressor` converged cleanly within the 2,000 maximum iteration threshold without numerical overflow or divergence warnings.
2. **SGD vs. Analytical Solvers (LinearRegression & Ridge):**
   - `SGDRegressor` achieves MAE of 14.16 days and RMSE of 26.95 days (R² = 0.1122), which closely mirrors analytical OLS (26.90 days) and Ridge (26.90 days).
   - This confirms that stochastic gradient approximation reliably approaches the closed-form analytical optimum on this scaled dataset.
3. **Comparison with Baselines:**
   - The median baseline retains the lowest overall MAE (11.47 days) due to the point-mass concentration at 0 days (>61% of records).
   - On extreme delays, `SGDRegressor` substantially outperforms the median baseline (Subgroup MAE: 61.42 vs 73.10 days).
