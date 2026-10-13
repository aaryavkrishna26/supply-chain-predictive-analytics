# Improved Regression Model Comparison Report

## 1. Experimental Setup

- **Script:** `src/improve_regression_models.py`
- **Data source:** `data/processed/scms_cleaned.csv`
- **Target variable:** `delay_days` (continuous integer days, unchanged from prior phases)
- **Train set:** 8,259 samples (80%)
- **Test set:** 2,065 samples (20%)
- **Split:** `train_test_split(test_size=0.20, random_state=42)` - identical to Phases 5B-5D
- **Test set target stats:** min=-372.0d, max=165.0d, mean=-6.06d, median=0.0d, std=28.61d
- **Extreme subgroup** (|delay_days|>30): 229 test records (11.09%)
- **Leakage exclusions:** Same as Phases 5B-5D (no delivery dates, delay targets, post-hoc financial fields)
- **Tuning:** RandomizedSearchCV on X_train only (n_iter=20, 3-fold KFold, scoring=neg_MAE)

## 2. Overall Model Performance

| Model | MAE (days) | RMSE (days) | R2 | MedAE (days) | Convergence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Mean)** | 14.48 | 28.60 | -0.0000 | 6.01 | Converged normally. |
| **Dummy (Median)** | 11.47 | 29.24 | -0.0450 | 0.00 | Converged normally. |
| **Linear Regression** | 13.98 | 26.90 | 0.1153 | 6.46 | Converged normally. |
| **Ridge (alpha=1.0)** | 13.96 | 26.90 | 0.1156 | 6.46 | Converged normally. |
| **RandomForest (tuned)** | 9.79 | 21.78 | 0.4199 | 2.31 | Converged normally. |
| **ExtraTrees (tuned)** | 9.97 | 22.78 | 0.3657 | 1.82 | Converged normally. |

## 3. Extreme-Delay Subgroup Performance (|delay_days| > 30)

Evaluated on 229 test records with absolute delay > 30 days (11.09% of test set).

| Model | Subgroup MAE | Subgroup RMSE | Subgroup MedAE | N Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | 69.98 | 81.99 | 57.99 | 229 |
| **Dummy (Median)** | 73.10 | 84.63 | 62.00 | 229 |
| **Linear Regression** | 61.51 | 74.19 | 50.56 | 229 |
| **Ridge (alpha=1.0)** | 61.54 | 74.21 | 50.60 | 229 |
| **RandomForest (tuned)** | 44.07 | 58.20 | 36.84 | 229 |
| **ExtraTrees (tuned)** | 42.35 | 58.73 | 30.95 | 229 |

## 4. Hyperparameter Tuning Summary

### RandomForestRegressor
Best parameters (n_iter=20, CV=3-fold KFold, scoring=neg_MAE):
- `regressor__n_estimators`: `400`
- `regressor__min_samples_split`: `2`
- `regressor__min_samples_leaf`: `2`
- `regressor__max_features`: `1.0`
- `regressor__max_depth`: `20`

### ExtraTreesRegressor
Best parameters (n_iter=20, CV=3-fold KFold, scoring=neg_MAE):
- `regressor__n_estimators`: `400`
- `regressor__min_samples_split`: `2`
- `regressor__min_samples_leaf`: `2`
- `regressor__max_features`: `1.0`
- `regressor__max_depth`: `20`

## 5. Model-Selection Analysis

### MAE vs RMSE Trade-offs

MAE and RMSE measure different aspects of prediction error:
- **MAE** measures the average absolute deviation in days. It treats all errors equally regardless of magnitude. A model optimizing MAE will shift predictions toward the median.
- **RMSE** penalizes large errors quadratically. It is dominated by the extreme tail of delivery outliers (e.g., year-entry typos at -372 days) and favors models that avoid large single-prediction blunders even at the cost of typical-case accuracy.
- **A model with lower MAE but higher RMSE than another** (e.g. Median Dummy vs Linear Regression) is performing better on typical, central predictions but worse on rare large delays. Neither metric is universally superior - the choice depends on whether the operational risk is concentrated in typical delays or in catastrophic tail events.

### Performance Rankings

- **Lowest MAE:** `RandomForest (tuned)` - MAE=9.79 days
- **Lowest RMSE:** `RandomForest (tuned)` - RMSE=21.78 days
- **Highest R2:** `RandomForest (tuned)` - R2=0.4199

### Context: Why Regression Remains Difficult

Pre-dispatch order features (country, vendor, product, scheduling month) explain a limited fraction of delay-duration variance. The dominant contributors to delivery delays are in-transit disruptions (customs clearance, port congestion, carrier routing) that are unobservable at order creation. Tree-based ensemble methods can potentially model non-linear interactions but may over-fit to the large cluster of 0-delay records (61% of data) without regularization constraints.

## 6. Residual Statistics

| Model | Residuals Mean | Residuals Std |
| :--- | :---: | :---: |
| **Dummy (Mean)** | -0.0511 | 28.61 |
| **Dummy (Median)** | -6.0644 | 28.61 |
| **Linear Regression** | -0.1086 | 26.91 |
| **Ridge (alpha=1.0)** | -0.0992 | 26.90 |
| **RandomForest (tuned)** | -0.2460 | 21.79 |
| **ExtraTrees (tuned)** | -0.1873 | 22.78 |

## 7. Limitations and Evaluation Integrity

- Hyperparameter tuning performed exclusively on X_train using cross-validation.
- Extreme target values (range: -372 to +192 days) kept unchanged per project policy.
- Same 80/20 random split as Phases 5B-5D ensures direct baseline comparability.
- Random split does not evaluate temporal generalization.
- Tree-based regressors may memorize 0-delay majority class without enforcing smoothness.
- Repeated test-set evaluation across phases introduces mild selection risk; treat as directional.

## 8. Tuning Log Notes

- RandomForestRegressor best params: {'regressor__n_estimators': 400, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 1.0, 'regressor__max_depth': 20}
- ExtraTreesRegressor best params: {'regressor__n_estimators': 400, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 1.0, 'regressor__max_depth': 20}
