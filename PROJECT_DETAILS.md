# Supply Chain Predictive Analytics — Project Documentation

This document serves as the authoritative, continuously maintained record of the Supply Chain Predictive Analytics project. It is updated alongside codebase changes.

---

## A. Project Overview

- **Project Name:** Supply Chain Predictive Analytics
- **Objective:** Build a standalone Python-based predictive analytics system to model, forecast, and optimize supply chain delivery performance using historical shipment records.
- **Problem Being Addressed:** Supply chain disruptions and late shipments lead to stockouts, operational friction, and increased logistics costs for global health and commodity supplies. This project evaluates whether historical shipment characteristics can reliably predict delivery delays, quantify delay severity, and reveal temporal volume trends.
- **Intended Users and Use Cases:**
  - *Supply Chain Operations Analysts:* Identify high-risk procurement routes, vendors, and shipment modes to proactively adjust lead times or prioritize buffer stock.
  - *Logistics Planners:* Predict delay durations in days to adjust scheduled handoffs and plan resource allocation.
  - *Strategic Planners:* Track seasonal shipment volume and delivery trends over time.
- **Current Project Stage:** **Phase 6A/6B Completed — Ensemble Model Improvement** (RandomForest and ExtraTrees classifiers and regressors trained, tuned (RandomizedSearchCV on training data only), and evaluated. RandomForest achieves F1=0.4490 (classification) and MAE=9.79d / RMSE=21.78d / R²=0.4199 (regression), substantially outperforming all prior baselines. Ready for Phase 6 Time-Series Analysis).

---

## B. Dataset

- **Dataset Filename:** `data/raw/SCMS_Delivery_History_Dataset.csv`
- **Source URL / Provenance:** Public Supply Chain Management System (SCMS) Delivery History dataset (USAID / PEPFAR supply chain data distributed via Kaggle / data.gov).
- **Single Dataset Confirmation:** Exactly one dataset is used in this standalone project. No additional external datasets are incorporated.
- **Known Dimensions and Coverage (Inspection Findings):**
  - *Rows:* 10,324
  - *Columns:* 33
  - *Time Span (Scheduled Delivery):* May 2, 2006 to December 31, 2015
  - *Time Span (Delivered to Client):* May 2, 2006 to September 14, 2015
  - *Duplicate Rows:* 0
- **Important Columns and Definitions:**
  - `ID`: Unique record identifier (`int64`).
  - `Project Code`, `PQ #`, `PO / SO #`, `ASN/DN #`: Procurement and tracking reference numbers.
  - `Country`: Destination country.
  - `Managed By`: Managing office/entity (e.g., PMO - US).
  - `Fulfill Via`: Fulfillment channel (e.g., Direct Drop, From RDC).
  - `Vendor INCO Term`: International Commercial Terms (e.g., CIP, DDU, EXW, FCA).
  - `Shipment Mode`: Transportation mode (`Air`, `Truck`, `Air Charter`, `Ocean`, or `NaN`).
  - `PQ First Sent to Client Date`: Date price quote sent to client (often placeholder text).
  - `PO Sent to Vendor Date`: Date purchase order sent to vendor (often placeholder text).
  - `Scheduled Delivery Date`: Contracted/scheduled arrival date at client.
  - `Delivered to Client Date`: Actual arrival date at client.
  - `Delivery Recorded Date`: System recording timestamp.
  - `Product Group`, `Sub Classification`: High-level commodity classifications (e.g., ARV, ACT, HRDT).
  - `Vendor`, `Item Description`, `Molecule/Test Type`, `Brand`, `Dosage`, `Dosage Form`: Product details.
  - `Unit of Measure (Per Pack)`, `Line Item Quantity`, `Line Item Value`, `Pack Price`, `Unit Price`: Financial and volume quantities.
  - `Weight (Kilograms)`: Shipment weight (mixed numeric and text strings).
  - `Freight Cost (USD)`: Invoiced freight cost (mixed numeric and text strings).
  - `Line Item Insurance (USD)`: Invoiced insurance amount.
- **Data-Quality Issues, Missing Values, and Limitations (Inspection Findings):**
  - *Standard `NaN` counts:* `Dosage` has 1,736 missing; `Shipment Mode` has 360 missing; `Line Item Insurance (USD)` has 287 missing.
  - *Placeholder/Sentinel text in numeric columns:*
    - `Weight (Kilograms)` contains 3,952 non-numeric strings (`'Weight Captured Separately'`: 1,507; cross-reference notes like `'See DN-304'`: 2,445).
    - `Freight Cost (USD)` contains 4,126 non-numeric strings (`'Freight Included in Commodity Cost'`: 1,442; `'Invoiced Separately'`: 239; cross-reference notes: 2,445).
  - *Placeholder text in date columns:*
    - `PO Sent to Vendor Date` contains 5,732 sentinels (`'N/A - From RDC'`: 5,404; `'Date Not Captured'`: 328).
    - `PQ First Sent to Client Date` contains 2,681 sentinels (`'Pre-PQ Process'`: 2,476; `'Date Not Captured'`: 205).
  - *Granularity Limitation:* All timestamps are date-only without hour/minute data. Models operate strictly on whole-day differences; hour-level predictions cannot be made.
- **Licensing and Redistribution:** SCMS public data is provided as open government public health data under Open Data guidelines; raw CSV is preserved unmodified.

---

## C. Technology Stack & Virtual Environment

- **Environment Isolation:** Dedicated project virtual environment located at `.venv/` created using Python `3.14.7` (64-bit).
- **Dependency Specification:** `requirements.txt` containing required core libraries.
- **Installed Verified Package Versions (`.venv`):**
  - **Python:** `3.14.7`
  - **Pandas:** `3.0.6` (Tabular data manipulation and datetime operations).
  - **NumPy:** `2.5.4` (Vectorized numerical computations and array structures).
  - **Scikit-learn:** `1.9.1` (Classification, regression, preprocessing pipelines, transformers, and evaluation metrics).
  - **Statsmodels:** `0.15.0` (Statistical time-series decomposition, ACF/PACF, and ARIMA modeling).
  - **Matplotlib:** `3.11.2` (Diagnostic and performance plotting).
  - **Joblib:** `1.6.0` (Model serialization and pipeline persistence).
- **Scope Restriction:** No web frameworks, database engines, or UI libraries are in scope for this standalone analytics project.

---

## D. Planned Machine Learning Tasks

| Task | Model / Method | Prediction Target | Candidate Features (Subject to Leakage Review) | Required Preprocessing | Evaluation Metrics | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Delay Classification** | Logistic Regression (Default + Balanced class weighting) vs DummyClassifier baseline | `is_delayed` (Binary: `1` if `delay_days > 0`, `0` otherwise) | `Country`, `Shipment Mode`, `Fulfill Via`, `Vendor INCO Term`, `Product Group`, `Sub Classification`, `First Line Designation`, `Dosage Form`, `Vendor`, `Manufacturing Site`, `Unit of Measure (Per Pack)`, `Line Item Quantity`, `Line Item Value`, `Pack Price`, `Unit Price`, `Scheduled Delivery Date` (calendar features) | Stratified 80/20 train/test split; rare category grouping (Top 15 + Other fitted on train); median imputation & standard scaling for numeric; constant imputation & one-hot encoding for categorical | Accuracy, Precision, Recall, F1-Score, PR-AUC, ROC-AUC, Confusion Matrix | **Evaluated** (Default: Acc=0.8852, ROC-AUC=0.8062, Rec=0.0000; Balanced: Acc=0.6305, ROC-AUC=0.8043, Rec=0.8776, F1=0.3528; Dummy: Acc=0.8852, Rec=0.0000) |
| **2. Delay Duration Prediction** | Linear Regression vs Mean and Median Dummy baselines | `delay_days` (Continuous: integer whole days) | Pre-dispatch order features (same candidate set as classification) | 80/20 random split; median imputation & scaling for numeric; rare-category grouping & one-hot encoding for categorical | RMSE, MAE, R², Median Absolute Error | **Evaluated** (Linear Regression: MAE=13.98d, RMSE=26.90d, R²=0.1153, MedAE=6.46d; Median Dummy: MAE=11.47d, RMSE=29.24d, R²=-0.0450, MedAE=0.00d; Mean Dummy: MAE=14.48d, RMSE=28.60d, R²=0.0000, MedAE=6.01d) |
| **3. Time Series Analysis** | Time-series decomposition, ACF/PACF analysis, ARIMA / SARIMA baseline (if series exhibits regularity) | Monthly/Quarterly aggregated shipment volume or late delivery rate | Chronological timestamp index (`Scheduled Delivery Date`), historical lags | Resampling to regular monthly frequency, stationarity testing (ADF test), differencing | RMSE, MAE, MAPE (Mean Absolute Percentage Error) | **Planned** |

*Note: Complete reports written to `reports/logistic_regression_evaluation.md`, `reports/threshold_analysis.md`, and `reports/linear_regression_evaluation.md`.*

---

## E. Data Cleaning and Target Definitions

- **Date Parsing Rules:**
  - Date fields parsed using `pd.to_datetime(..., errors='coerce', format='mixed')`.
  - Descriptive text and unparseable sentinels (`'Pre-PQ Process'`, `'Date Not Captured'`, `'N/A - From RDC'`) are converted strictly to `pd.NaT`.
  - No dates are synthesized, interpolated, or imputed.
- **Numeric Conversion Rules:**
  - Fields such as `Weight (Kilograms)` and `Freight Cost (USD)` are parsed via `pd.to_numeric(..., errors='coerce')`.
  - Text placeholders (`'Weight Captured Separately'`, `'Freight Included in Commodity Cost'`, cross-references) become standard `NaN`.
- **Target Definitions:**
  - `delay_days`: Integer whole-day difference computed as:
    $$\text{delay\_days} = \text{Delivered to Client Date} - \text{Scheduled Delivery Date}$$
    - A negative value indicates an early delivery.
    - Zero (`0`) indicates an on-time delivery.
    - A positive value indicates a late delivery.
  - `is_delayed`: Binary indicator:
    - `1.0` if $\text{delay\_days} > 0$
    - `0.0` if $\text{delay\_days} \le 0$
    - `NaN` if either scheduled or actual delivery date is missing.
- **Handling Rows with Missing Target Dates:**
  - Any row missing either `Scheduled Delivery Date` or `Delivered to Client Date` is kept in the processed dataset with `delay_days = NaN` and `is_delayed = NaN`.
  - A warning and count log is emitted during cleaning.
  - Rows with `NaN` targets will be excluded strictly during model training and evaluation splits, but never discarded from the master cleaned dataset.
- **Processed-Data Output Location:**
  - `data/processed/scms_cleaned.csv` (generated and validated, 10,324 rows × 35 columns).

---

## F. Data Leakage Prevention

A primary objective of this project is strict enforcement of temporal and target boundaries:

1. **Strict Distinction Between Historical Ground Truth and Prediction-Time Features:**
   - In historical records, actual delivery timestamps and post-delivery invoice figures are present.
   - At prediction time (when an order is placed or scheduled), the actual delivery outcome is unknown.
2. **Explicitly Prohibited Features in Model Matrices:**
   - `Delivered to Client Date` (the delivery arrival event itself).
   - `Delivery Recorded Date` (system timestamp recorded upon or after arrival).
   - `delay_days` (continuous target label).
   - `is_delayed` (binary target label).
3. **Quarantine of Suspicious Post-Hoc Fields:**
   - `Freight Cost (USD)` and `Line Item Insurance (USD)`: In cases where freight is settled after delivery or invoiced separately, these figures may reflect post-delivery knowledge or final shipment weight adjustments. They are treated as **unapproved candidate features** until their availability at order scheduling is verified.
4. **Temporal Split Rule:**
   - Model validation will employ chronological train-test splits (e.g., training on earlier years and testing on subsequent years) to prevent temporal leakage inherent in random k-fold cross-validation.

---

## G. Repository Structure

### Existing Files and Folders
```text
supply-chain-predictive-analytics/
├── .venv/                                           # Dedicated Python 3.14 virtual environment
├── data/
│   ├── raw/
│   │   └── SCMS_Delivery_History_Dataset.csv        # Original raw dataset (read-only, 10,324 rows)
│   └── processed/
│       └── scms_cleaned.csv                         # Validated cleaned dataset (10,324 rows × 35 columns)
├── reports/                                         # Output directory for evaluation reports
│   ├── logistic_regression_evaluation.md           # Phase 4B baseline evaluation report
│   ├── threshold_analysis.md                       # Phase 4C decision threshold analysis report
│   ├── linear_regression_evaluation.md              # Phase 5B Linear Regression evaluation report
│   ├── regression_model_comparison.md               # Phase 5C regression model comparison report
│   ├── sgd_regression_comparison.md                 # Phase 5D SGD regression comparison report
│   ├── sgd_classification_comparison.md             # Phase 5E SGD classification comparison report
│   └── final_model_selection.md                     # Phase 5F final model selection and synthesis report
├── src/
│   ├── preprocess.py                                # Reusable preprocessing module and functions
│   ├── train_logistic_regression.py                 # Self-contained Logistic Regression training & eval script
│   ├── analyze_thresholds.py                        # Phase 4C decision threshold analysis script
│   ├── train_linear_regression.py                   # Self-contained Linear Regression training & eval script
│   ├── compare_regression_models.py                 # Phase 5C regression model comparison script
│   ├── compare_sgd_regression.py                    # Phase 5D SGD regression comparison script
│   └── compare_sgd_classification.py                # Phase 5E SGD classification comparison script
├── PROJECT_DETAILS.md                               # Living project documentation (this document)
└── requirements.txt                                 # Pinned project dependencies
```

### Planned Files and Folders (Not Yet Created)
```text
supply-chain-predictive-analytics/
├── notebooks/                                       # Planned for exploratory data analysis
│   └── eda.ipynb
├── src/
│   ├── models/
│   │   ├── predict_duration.py                      # Planned Linear Regression pipeline
│   │   └── time_series.py                           # Planned Time-Series forecasting module
│   └── utils.py                                     # Planned evaluation and plotting utilities
└── models/                                          # Planned directory for serialized model artifacts (.joblib)
```

---

## H. Project Roadmap

1. **Step 1: Dataset Inspection** — *Completed*
   - Verified Python & Pandas availability.
   - Validated dataset dimensions (10,324 × 33), null distributions, duplicate counts (0), and sentinel strings.
2. **Step 2: Preprocessing Pipeline & Documentation** — *Completed*
   - Implemented `src/preprocess.py` with safe conversions, target derivation, and leakage quarantine.
   - Initialized `PROJECT_DETAILS.md`.
3. **Step 3: Dataset Validation (Phase 3)** — *Completed*
   - Validated dimensions: 10,324 rows × 35 columns (zero rows lost, 2 target columns added: `delay_days`, `is_delayed`).
   - Verified target calculations: 0 mismatches in `delay_days` and `is_delayed`.
   - Verified class balance: 1,186 delayed (11.49%) vs 9,138 not delayed (88.51%).
   - Verified early/on-time split: 2,814 early (27.26%), 6,324 on-time (61.26%), 1,186 late (11.49%).
   - Verified duplicate rows: 0 duplicates.
   - Verified missing values: target columns have 0 missing values; 28 of 35 columns are 100% complete.
4. **Step 4: Logistic Regression Baseline (Phase 4B)** — *Completed*
   - Trained `DummyClassifier`, `LogisticRegression(default)`, and `LogisticRegression(balanced)` on stratified 80/20 train/test split.
   - Evaluated on held-out test set (2,065 samples, 237 delayed).
   - Results:
     - `Dummy (Most Frequent)`: Accuracy=0.8852, Precision=0.0000, Recall=0.0000, F1=0.0000, ROC-AUC=0.5000.
     - `Logistic Regression (Default)`: Accuracy=0.8852, Precision=0.0000, Recall=0.0000, F1=0.0000, PR-AUC=0.3099, ROC-AUC=0.8062. (Predicts all 0 due to class imbalance).
     - `Logistic Regression (Balanced)`: Accuracy=0.6305, Precision=0.2208, Recall=0.8776, F1=0.3528, PR-AUC=0.3060, ROC-AUC=0.8043. (Successfully detects 208/237 delayed shipments, Recall=87.8%).
   - Complete report written to `reports/logistic_regression_evaluation.md`.
   - **Phase 4C Threshold Analysis Completed:**
     - Evaluated default Logistic Regression at thresholds: 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50 on test set.
     - Confirmed threshold 0.50 exactly reproduces baseline (TP=0, FP=0, TN=1828, FN=237, Accuracy=0.8852).
     - Peak F1 achieved at `threshold=0.20` (F1=0.3867, Recall=56.54%, Precision=29.39%, FPR=17.61%).
     - High sensitivity alert threshold: `threshold=0.15` (Recall=74.26%, Precision=24.96%, FPR=28.94%).
     - Dedicated analysis script: `src/analyze_thresholds.py`.
     - Dedicated report: `reports/threshold_analysis.md`.
5. **Step 5: Linear Regression (Delay Duration — Phase 5B)** — *Completed*
   - Trained `DummyRegressor(strategy='mean')`, `DummyRegressor(strategy='median')`, and `LinearRegression()` on 80/20 train/test split (`random_state=42`).
   - Evaluated on held-out test set (2,065 samples, target: continuous `delay_days`).
   - Results:
     - `Dummy (Mean)`: MAE = 14.48 days, RMSE = 28.60 days, R² = 0.0000, MedAE = 6.01 days.
     - `Dummy (Median)`: MAE = 11.47 days, RMSE = 29.24 days, R² = -0.0450, MedAE = 0.00 days.
     - `Linear Regression`: MAE = 13.98 days, RMSE = 26.90 days, R² = 0.1153, MedAE = 6.46 days.
     - Subgroup (`|delay_days| > 30`, 229 samples): Linear Regression MAE = 61.51 days, RMSE = 74.19 days, MedAE = 50.56 days vs Mean Dummy MAE = 69.98 days.
   - Complete report written to `reports/linear_regression_evaluation.md`.
   - **Phase 5C Model Comparison Completed:**
     - Compared `DummyRegressor(mean)`, `DummyRegressor(median)`, `LinearRegression()`, `Ridge(alpha=1.0)`, and `HuberRegressor()`.
     - Results on held-out test set:
       - `Ridge (alpha=1.0)`: MAE = 13.96d, RMSE = 26.90d, R² = 0.1156, MedAE = 6.46d (Converged).
       - `Huber Regressor`: MAE = 11.13d, RMSE = 28.68d, R² = -0.0053, MedAE = 0.00d (L-BFGS reached max_iter=1000 limit).
       - `Dummy (Median)`: MAE = 11.47d, MedAE = 0.00d.
       - Extreme Subgroup (`|delay_days| > 30`): Linear Regression achieved lowest error (MAE = 61.51d) vs Huber (71.18d) and Median Dummy (73.10d).
     - Dedicated analysis script: `src/compare_regression_models.py`.
     - Complete report written to `reports/regression_model_comparison.md`.
   - **Phase 5D SGD Regression Comparison Completed:**
     - Compared `SGDRegressor(loss='squared_error', penalty='l2', max_iter=2000, alpha=0.0001)` against OLS and Ridge.
     - Converged without warnings; achieved MAE = 14.16d, RMSE = 26.95d, R² = 0.1122, MedAE = 6.74d (closely matching analytical OLS RMSE of 26.90d).
     - Extreme Subgroup (`|delay_days| > 30`): Subgroup MAE = 61.42d vs Median Dummy 73.10d.
     - Dedicated script: `src/compare_sgd_regression.py`; report: `reports/sgd_regression_comparison.md`.
   - **Phase 5E SGD Classification Comparison Completed:**
     - Compared `SGDClassifier(loss='log_loss')` (default and balanced) against Logistic Regression.
     - Converged without warnings. Balanced SGDClassifier achieved Recall = 87.34% (207/237 delays caught), F1 = 0.3456, ROC-AUC = 0.7822 vs Balanced Logistic Regression (Recall = 87.76%, F1 = 0.3528, ROC-AUC = 0.8043).
     - Performed 5-fold cross-validation threshold analysis strictly on training set `X_train` to prevent test-set tuning leakage (peak CV F1 = 0.3177 at threshold 0.15).
     - Dedicated script: `src/compare_sgd_classification.py`; report: `reports/sgd_classification_comparison.md`.
   - **Phase 5F Final Model Selection Completed:**
     - Synthesized all empirical classification and regression results.
     - Early warning classification: Balanced Logistic Regression recommended for high recall (87.8% sensitivity) or calibrated threshold ~0.15–0.20 depending on stockout-to-alarm cost ratios.
     - Delay duration estimation: Ridge / OLS recommended as directional estimators (RMSE = 26.90d vs 28.60d baseline; R² = 0.1156), anchored by median baseline (0.0d) for nominal schedules.
     - Documented limitations (random split vs chronological deployment, outlier leverage, pre-dispatch visibility horizon).
     - Final synthesis report: `reports/final_model_selection.md`.
6. **Step 6: Time-Series Analysis** — *Planned*
   - Resample shipment counts and delay rates to monthly intervals.
   - Test stationarity and decompose trend/seasonality.
   - Evaluate ARIMA baseline if chronological structure permits.
7. **Step 7: Model Evaluation and Persistence** — *Planned*
   - Consolidate metrics, compare against naive baselines, serialize models to `models/`.
8. **Step 8: Future Work (Explicitly Out of Scope for Current Phase)** — *Deferred*
   - Web application, dashboard UI, REST API, database integration.

---

## I. Decisions, Limitations, and Change Log

### Key Decisions
- **2026-10-14:** Confirmed strict single-dataset constraint (`SCMS_Delivery_History_Dataset.csv`).
- **2026-10-14:** Coerce descriptive strings in numeric columns (`Weight (Kilograms)`, `Freight Cost (USD)`) to `NaN` rather than dropping rows, retaining full dataset size of 10,324 rows.
- **2026-10-14:** Define positive delay strictly as `delay_days > 0`. Shipments arriving on the scheduled date (`delay_days == 0`) or earlier (`delay_days < 0`) are categorized as on-time (`is_delayed = 0`).
- **2026-10-14:** Prohibit automated script execution; `data/processed/scms_cleaned.csv` was generated only after manual execution by the user.
- **2026-10-14:** Dataset validation confirmed 100% check pass rate on `data/processed/scms_cleaned.csv`. Data is approved and ready for Phase 4 (Logistic Regression).
- **2026-10-14:** Excluded all post-delivery and post-hoc fields (`Freight Cost (USD)`, `Line Item Insurance (USD)`, `Weight (Kilograms)`, `PO Sent to Vendor Date`, `PQ First Sent to Client Date`) from Logistic Regression baseline to ensure zero data leakage.
- **2026-10-14:** Verified that `LogisticRegression(balanced)` achieves ROC-AUC of 0.8043 and Recall of 87.76% (identifying 208 of 237 delays) compared to 0.0% recall on the unweighted model, proving that class-weighted penalization is essential for delay risk classification.
- **2026-10-14:** Evaluated continuous delay prediction in Phase 5B without clipping outliers; observed that Linear Regression achieves a modest RMSE improvement (26.90 vs 28.60 days) and R² of 0.1153 over mean baseline, but higher MedAE (6.46 days) than the median baseline (0.00 days) due to heavy asymmetric distribution tails.
- **2026-10-14:** Evaluated Ridge and Huber regressors in Phase 5C. Found Ridge (`alpha=1.0`) virtually identical to OLS (mild regularization effect on 8k rows). Huber Regressor achieved lowest overall MAE (11.13 days) and MedAE (0.00 days) by downweighting tail errors, though triggering an iteration limit warning at `max_iter=1000`.
- **2026-10-14:** Evaluated SGDRegressor and SGDClassifier in Phase 5D and 5E. Confirmed both stochastic gradient estimators converge cleanly on scaled data, achieving parity with analytical OLS and batch Logistic Regression while providing scalability. Validated classification threshold trade-offs using 5-fold cross-validation on training data to protect test-set neutrality.
- **2026-10-14:** Synthesized all models in Phase 5F. Established Balanced Logistic Regression as the primary risk-classification candidate and Ridge/OLS as directional duration estimators, identifying pre-dispatch data limitations and defining next steps for Phase 6 time-series analysis.
- **2026-10-14:** Phase 6A ensemble classification completed. RandomForestClassifier (tuned via RandomizedSearchCV on training data only) achieves ROC-AUC=0.8576 and F1=0.4490 at CV-calibrated threshold=0.40, surpassing all prior classification baselines. Hyperparameters and thresholds tuned on training data only; test set evaluated once after tuning.
- **2026-10-14:** Phase 6B ensemble regression completed. RandomForestRegressor (tuned via RandomizedSearchCV on training data only) achieves MAE=9.79d, RMSE=21.78d, R²=0.4199 — a 29.9% MAE reduction and 263% relative R² improvement over Ridge/OLS baseline. Extreme-delay subgroup MAE reduced from 61.51d to 44.07d. ExtraTreesRegressor runner-up with lowest MedAE (1.82d). Improvement consolidated in `reports/model_improvement_summary.md`.

### Known Limitations & Unresolved Questions
- **Unresolved Question:** Determine whether `Freight Cost (USD)` is known strictly prior to dispatch or is finalized post-delivery. Until confirmed, it will not be used in early prediction models.
- **Limitation:** Temporal granularity is restricted to calendar days; delivery delays cannot be predicted at the hour level.
- **Limitation:** High class imbalance in delay labels (11.49% late vs 88.51% on-time/early) requires balanced class weighting or threshold adjustment in Logistic Regression.
- **Limitation:** RandomForest regression achieves R²=0.42; 58% of delay-duration variance remains unexplained by pre-dispatch features. In-transit disruptions are inherently unobservable at order creation.
- **Limitation:** Tree ensemble probability estimates are known to be poorly calibrated; Platt scaling or isotonic regression is recommended before production classification deployment.
- **Limitation:** Comparing all phases (4, 5, 6) on the same test partition introduces mild post-hoc selection risk. Chronological validation (e.g. train 2006-2012, test 2013-2015) is required before production deployment.

### Change Log
- **2026-10-14:** Initialized project structure (`src/`, `data/processed/`). Created reusable preprocessing script `src/preprocess.py`. Created living documentation `PROJECT_DETAILS.md`.
- **2026-10-14:** Phase 3 Validation completed on `data/processed/scms_cleaned.csv`. Verified all 9 validation checks (dimensions, breakdown, duplicate count, target logic, zero row loss). Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Phase 4B implementation completed. Created `src/train_logistic_regression.py` and initialized `reports/` folder. Script is staged and awaiting manual user execution.
- **2026-10-14:** Dedicated `.venv` virtual environment created using Python 3.14.7. Installed and verified core dependencies from `requirements.txt` (`scikit-learn 1.9.1`, `pandas 3.0.6`, `numpy 2.5.4`, `statsmodels 0.15.0`, `matplotlib 3.11.2`, `joblib 1.6.0`).
- **2026-10-14:** Executed `src/train_logistic_regression.py`. Successfully trained and evaluated Dummy baseline, Default Logistic Regression, and Balanced Logistic Regression models on held-out test data. Generated `reports/logistic_regression_evaluation.md`. Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Completed Phase 4C Decision Threshold Analysis via `src/analyze_thresholds.py`. Generated `reports/threshold_analysis.md`. Confirmed threshold 0.50 exactly reproduces default baseline; identified `threshold=0.20` as peak F1 (0.3867) and `threshold=0.15` as high-sensitivity operational alert threshold (74.26% recall). Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Implemented and executed Phase 5B Linear Regression pipeline via `src/train_linear_regression.py`. Evaluated Linear Regression, Mean Dummy, and Median Dummy on held-out test set. Generated `reports/linear_regression_evaluation.md`. Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Implemented and executed Phase 5C Regression Model Comparison via `src/compare_regression_models.py`. Evaluated Ridge and Huber Regressors against baselines. Documented Huber L-BFGS iteration limit warning and comparative metrics in `reports/regression_model_comparison.md`. Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Implemented and executed Phase 5D (`src/compare_sgd_regression.py`) and Phase 5E (`src/compare_sgd_classification.py`). Generated `reports/sgd_regression_comparison.md` and `reports/sgd_classification_comparison.md`. Evaluated SGDRegressor and SGDClassifier with convergence tracking and internal CV threshold analysis. Updated `PROJECT_DETAILS.md`.
- **2026-10-14:** Completed Phase 5F Model Selection and Final Evaluation Report (`reports/final_model_selection.md`). Synthesized all classification and regression benchmarks, operational trade-offs, methodological safeguards, and future roadmap. Updated `PROJECT_DETAILS.md`.
