# Linear Regression Baseline Evaluation Report (Phase 5B)

## 1. Overview and Setup

- **Target Variable:** `delay_days` (continuous whole days: Delivered to Client Date − Scheduled Delivery Date)
- **Data Source:** `data/processed/scms_cleaned.csv`
- **Train Set Size:** 8259 samples (80%)
- **Test Set Size:** 2065 samples (20%)
- **Split Method:** Independent reproducible random split (`test_size=0.20`, `random_state=42`)
- **Leakage Quarantine:** Excluded `delay_days`, `is_delayed`, `Delivered to Client Date`, `Delivery Recorded Date`, and quarantined features (`Weight`, `Freight Cost`, `Insurance`, `PO Sent Date`, `PQ Sent Date`). Calendar features extracted strictly from `Scheduled Delivery Date`.

## 2. Test-Set Target Distribution

- **Min / Max:** -372.0 days / 165.0 days
- **Mean:** -6.06 days
- **Median:** 0.00 days
- **Standard Deviation:** 28.61 days
- **Interquartile Range (25th - 75th):** -3.0 to 0.0 days
- **Records with |delay_days| > 30:** 229 (11.09% of test set)

## 3. Overall Model Performance Comparison

| Model | MAE (days) | RMSE (days) | R² | MedAE (days) |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | 14.48 | 28.60 | -0.0000 | 6.01 |
| **Dummy (Median)** | 11.47 | 29.24 | -0.0450 | 0.00 |
| **Linear Regression** | 13.98 | 26.90 | 0.1153 | 6.46 |


## 4. Subgroup Performance: Extreme Delay Records (|delay_days| > 30)

Evaluated exclusively on the 229 test records exhibiting absolute delays greater than 30 days:

| Model | Subgroup MAE (days) | Subgroup RMSE (days) | Subgroup MedAE (days) |
| :--- | :---: | :---: | :---: |
| **Dummy (Mean)** | 69.98 | 81.99 | 57.99 |
| **Dummy (Median)** | 73.10 | 84.63 | 62.00 |
| **Linear Regression** | 61.51 | 74.19 | 50.56 |


## 5. Residual Diagnostics

### Dummy (Mean)
- **Residuals Mean:** -0.0511 days
- **Residuals Std Dev:** 28.61 days
- **Min Residual (Underprediction / Error):** -365.99 days
- **Max Residual (Overprediction / Error):** 171.01 days

### Dummy (Median)
- **Residuals Mean:** -6.0644 days
- **Residuals Std Dev:** 28.61 days
- **Min Residual (Underprediction / Error):** -372.00 days
- **Max Residual (Overprediction / Error):** 165.00 days

### Linear Regression
- **Residuals Mean:** -0.1086 days
- **Residuals Std Dev:** 26.91 days
- **Min Residual (Underprediction / Error):** -371.74 days
- **Max Residual (Overprediction / Error):** 160.40 days

## 6. Interpretation and Findings

1. **Comparison with Dummy Baselines:**
   - **Median Baseline:** Predicting constant median (0.0 days) yields MAE of 11.47 days and MedAE of 0.00 days because over 61% of shipments arrive exactly on schedule.
   - **Mean Baseline:** Predicting constant mean yields MAE of 14.48 days and RMSE of 28.60 days.
   - **Linear Regression:** Achieves MAE of 13.98 days, RMSE of 26.90 days, and R² of 0.1153.
   - While Linear Regression achieves a modest improvement in RMSE over the mean baseline (26.90 vs 28.60), its MedAE (6.46 days) is higher than the simple median predictor (0.00 days) because linear regression shifts predictions away from zero to minimize squared errors across the asymmetric heavy tails.
2. **Impact of Heavy Tails on Metrics:**
   - Because RMSE squares errors, extreme negative outliers (e.g. shipments arriving 200–370 days early) heavily penalize the model, keeping RMSE above 25 days.
   - R² is modest, confirming that pre-dispatch order features (country, vendor, items) explain only a limited portion of variance in exact delay duration.
3. **Operational Context and Limitations:**
   - Many delivery duration variations stem from in-transit disruptions (customs clearance delays, port congestion, weather, local logistics) which occur *after* order dispatch and are therefore unobservable at prediction time.
   - Random train/test split does not evaluate temporal generalization across fiscal regimes; a time-based validation would be appropriate for production evaluation.
