# Model Selection and Final Evaluation Report (Phase 5F)

## A. Project Overview

- **Project Name:** Supply Chain Predictive Analytics
- **Dataset:** `data/raw/SCMS_Delivery_History_Dataset.csv` (preprocessed and validated in `data/processed/scms_cleaned.csv`, 10,324 rows × 35 columns).
- **Core Problem:** Anticipating supply chain delivery disruptions for public health commodities (antiretrovirals, malaria treatments, diagnostic kits) using pre-dispatch shipment and procurement features.

### Prediction Targets and Problem Formulations
1. **Binary Delay Classification (`is_delayed`):**
   - *Target Definition:* `1` if `delay_days > 0` (late delivery), `0` if `delay_days <= 0` (on-time or early arrival).
   - *Operational Purpose:* Provide an early-warning risk triage mechanism at order scheduling to flag high-risk procurement routes, vendors, and freight modes before shipment departure.
   - *Class Distribution:* **1,186 Late shipments (11.49%)** vs **9,138 On-Time / Early shipments (88.51%)**.
2. **Continuous Delay Duration Regression (`delay_days`):**
   - *Target Definition:* Integer calendar days calculated as `Delivered to Client Date − Scheduled Delivery Date`.
   - *Operational Purpose:* Estimate the expected magnitude of delay in days to allow warehouse planners and buffer stock managers to adjust replenishment lead times.
   - *Target Distribution:*
     - Range: **−372.0 days** (early) to **+192.0 days** (late).
     - Mean: **−6.02 days** | Median: **0.00 days** | Standard Deviation: **27.23 days**.
     - Early shipments (<0 days): 2,814 (27.26%).
     - Exactly on-time shipments (==0 days): 6,324 (61.26%).
     - Late shipments (>0 days): 1,186 (11.49%).
     - Extreme absolute delays ($|\text{delay\_days}| > 30$): 1,133 shipments (10.97%).

### Difference Between Classification and Regression Formulations
- **Categorical vs. Continuous Modeling:** Classification solves a discrete boundary problem (*is this shipment late or not?*), mapping features to a binary probability score. Regression solves a continuous quantity estimation problem (*by how many days will arrival deviate from schedule?*), mapping features to a scalar value on the real number line.
- **Loss Landscapes and Asymmetry:** Classification optimizes log-likelihood / cross-entropy loss, which penalizes probabilistic misclassifications regardless of the severity of the delay days behind them. Regression optimizes squared-error (OLS/Ridge) or Huber loss, which penalizes the numeric distance between predicted and actual delivery days.
- **Operational Roles:** In supply chain practice, classification serves as an automated triage alarm (flagging shipments that require pre-dispatch expedited handling or vendor check-ins), whereas regression serves as a sizing tool (informing inventory safety buffers, warehouse dock scheduling, and re-order lead times). Neither replaces the other; they represent complementary views of delivery uncertainty.

---

## B. Classification Comparison (`is_delayed`)

All classification models were trained on 8,259 samples (80%) and evaluated on the identical held-out test split of 2,065 samples (20% stratified split, with 237 Late [11.48%] and 1,828 On-Time [88.52%]). Preprocessing (OneHotEncoder and StandardScaler) was fitted strictly on training data without test-set leakage.

### Model Performance Summary Table
| Model | Accuracy | Precision (Late) | Recall (Late) | F1-Score | PR-AUC | ROC-AUC | Convergence Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Most Frequent)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.5000 | Converged |
| **Logistic Regression (Default)** | 0.8852 | 0.0000 | 0.0000 | 0.0000 | **0.3099** | **0.8062** | Converged |
| **Logistic Regression (Balanced)** | 0.6305 | **0.2208** | **0.8776** | **0.3528** | 0.3060 | 0.8043 | Converged |
| **SGDClassifier (Default, `log_loss`)** | 0.8673 | 0.3131 | 0.1308 | 0.1845 | 0.2910 | 0.7977 | Converged |
| **SGDClassifier (Balanced, `log_loss`)** | 0.6203 | 0.2154 | 0.8734 | 0.3456 | 0.2562 | 0.7822 | Converged |

### Confusion Matrix Breakdown (Test Set, 2,065 samples)
- **Dummy (Most Frequent):** TN = 1,828 | FP = 0 | FN = 237 | TP = 0
- **Logistic Regression (Default):** TN = 1,828 | FP = 0 | FN = 237 | TP = 0
- **Logistic Regression (Balanced):** TN = 1,094 | FP = 734 | FN = **29** | TP = **208**
- **SGDClassifier (Default):** TN = 1,760 | FP = 68 | FN = 206 | TP = 31
- **SGDClassifier (Balanced):** TN = 1,074 | FP = 754 | FN = **30** | TP = **207**

### Why Accuracy Alone Is Misleading
With a positive delay prevalence of only 11.48%, standard unweighted empirical loss minimization incentivizes predicting class 0 (on-time) across all instances. A naive baseline that predicts "never delayed" achieves an ostensibly high accuracy of **88.52%**, but yields a **Recall of 0.0%** (missing all 237 delayed shipments). For logistics risk management where failing to catch a stockout is disastrous, accuracy is completely uninformative.

### Candidate Selection: Balanced Logistic Regression
- **Strengths:** By weighting errors inversely proportional to class frequencies (~7.7:1), Balanced Logistic Regression achieves **87.76% Recall** (detecting 208 out of 237 late deliveries) with an **ROC-AUC of 0.8043**.
- **Trade-off:** This high sensitivity generates 734 false alarms out of 1,828 on-time shipments (FPR = 40.15%), resulting in a Precision of 22.08%.
- **Caveat:** It is not universally optimal; for organizations where the labor cost of inspecting flagged orders is high relative to delay penalties, a higher probability threshold (e.g. 0.20 via threshold calibration) offers a more balanced trade-off.

---

## C. Regression Comparison (`delay_days`)

All regression models were trained on 8,259 samples (80%) and evaluated on the identical held-out test split of 2,065 samples (20% random split, `random_state=42`). Preprocessing was fitted strictly on training data without test-set leakage.

### Model Performance Summary Table
| Model | MAE (days) | RMSE (days) | R² | MedAE (days) | Convergence Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy (Mean)** | 14.48 | 28.60 | 0.0000 | 6.01 | Converged |
| **Dummy (Median)** | **11.47** | 29.24 | −0.0450 | **0.00** | Converged |
| **Linear Regression** | 13.98 | **26.90** | 0.1153 | 6.46 | Converged |
| **Ridge (`alpha=1.0`)** | 13.96 | **26.90** | **0.1156** | 6.46 | Converged |
| **SGDRegressor (`squared_error`, L2)** | 14.16 | 26.95 | 0.1122 | 6.74 | Converged |
| **Huber Regressor** | **11.13** | 28.68 | −0.0053 | **0.00** | *Warning: L-BFGS reached max_iter=1000 limit* |

### Subgroup Performance on Extreme Delays ($|\text{delay\_days}| > 30$, 229 samples)
| Model | Subgroup MAE (days) | Subgroup RMSE (days) | Subgroup MedAE (days) |
| :--- | :---: | :---: | :---: |
| **Dummy (Mean)** | 69.98 | 81.99 | 57.99 |
| **Dummy (Median)** | 73.10 | 84.63 | 62.00 |
| **Linear Regression** | **61.51** | **74.19** | **50.56** |
| **Ridge (`alpha=1.0`)** | 61.54 | 74.21 | 50.60 |
| **SGDRegressor** | 61.42 | 74.08 | 49.90 |
| **Huber Regressor** | 71.18 | 83.31 | 59.00 |

### Critical Analytical Distinctions
1. **Huber Regressor's Hybrid Loss:** Huber achieved the lowest overall MAE (11.13 days) and a MedAE of 0.00 days because its linear penalty on large errors prevents tail leverage from shifting the central estimate away from 0. However, on squared-error metrics (RMSE = 28.68 days, $R^2 = -0.0053$), Huber performed worse than OLS. Furthermore, Huber triggered an optimization warning (`lbfgs failed to converge after 1000 iterations`), requiring caution before production use.
2. **Linear Regression & Ridge Equivalence:** Ridge with default regularization performed nearly identically to unregularized Linear Regression (RMSE: 26.90 vs 26.90 days; $R^2$: 0.1156 vs 0.1153), indicating that feature multicollinearity is mild across the ~70 encoded columns on 8k rows.
3. **Why Predicting Delay Duration Remains Challenging:**
   - Pre-dispatch order features (country, vendor, items, scheduling month) explain only ~11.5% of the variance in delivery duration ($R^2 \approx 0.115$).
   - Over 61% of records arrive exactly on schedule (0 days), while tail events stretch from −372 days (programmatic early framework deliveries or date-entry year offsets) to +192 days (in-transit customs clearance bottlenecks). In-transit factors cannot be observed prior to shipment dispatch.

---

## D. Model-Selection Recommendations

### 1. Early-Warning Delay Classification
- **Recommended Model:** **Logistic Regression with Balanced Class Weighting (`class_weight='balanced'`)** or calibrated thresholding (~0.15–0.20) on default Logistic Regression.
- **Operational Justification:** Achieves **87.8% Recall** and **ROC-AUC = 0.8043**, successfully detecting 208 out of 237 delayed shipments on the held-out test set.
- **Cost Trade-off Rule:** Model selection in classification fundamentally depends on the asymmetric cost ratio between a missed delay ($C_{\text{FN}}$: stockout, patient treatment interruption, emergency air-freight costs) and an unnecessary alert ($C_{\text{FP}}$: warehouse inspection labor, expedited administrative follow-up). If $C_{\text{FN}} / C_{\text{FP}} > 4$, the balanced model or threshold 0.15 is justified despite the higher false positive rate (FPR = 40.15% or 28.94%). If inspection capacity is constrained, calibrate the decision threshold toward 0.20 (F1 = 0.3867, Recall = 56.54%, FPR = 17.61%).

### 2. Delay-Duration Estimation
- **Recommended Model:** **Linear Regression / Ridge (`alpha=1.0`)** for directional duration estimation, coupled with a **Median Baseline (0.0 days)** for nominal scheduling.
- **Operational Justification:** Ridge and OLS reduce test RMSE from 28.60 to 26.90 days and lower extreme subgroup error ($|\text{delay\_days}| > 30$) from 73.10 to 61.51 days. However, because pre-dispatch features explain only ~11.5% of duration variance ($R^2 = 0.1156$), point estimates must not be treated as deterministic guarantees.
- **Role of the Median Baseline:** With over 61% of shipments arriving on the scheduled date, predicting 0 days error provides an optimal median absolute error (MedAE = 0.00 days). Ridge/OLS should be used selectively when procurement risk indicators signal non-zero duration deviations.

### 3. Role of Simple Baselines
- **Classification Baseline:** `DummyClassifier(strategy='most_frequent')` exposes the trap of naive accuracy metrics (88.52% accuracy, but 0.0% recall), serving as a benchmark to ensure any deployed model delivers genuine operational sensitivity.
- **Regression Baselines:** `DummyRegressor(strategy='mean')` (−6.01 days) provides the squared-error benchmark ($R^2 = 0$), while `DummyRegressor(strategy='median')` (0.0 days) serves as the nominal operational anchor for shipments with no risk flags.

### 4. Further Investigation Before Operational Deployment
- **Business Cost Matrix Alignment:** Formalize exact dollar and operational costs with logistics stakeholders to mathematically solve for the optimal classification threshold rather than relying on default 0.50 cutoffs.
- **Chronological Backtesting:** Re-fit and test candidate models across rolling multi-year time windows (e.g. 2006–2012 train vs 2013–2015 test) to verify temporal robustness.
- **Data Quality Audit:** Confirm and flag the 4 anomalous year-offset records in source ERP databases prior to production model retraining.

---

## E. Inherent Project Limitations

1. **Random Train/Test Split vs. Chronological Reality:**
   - The evaluation utilized a random 80/20 train/test split. In actual operations, models must predict future shipments across evolving logistics networks, seasonal cycles, changing carrier contracts, and global macroeconomic conditions. Random splitting allows temporal interpolation across the entire 2006–2015 timeline, which can overestimate real-world forward-looking performance.
2. **Repeated Test-Set Comparisons and Potential Selection Bias:**
   - Comparing multiple model architectures (Dummy, OLS, Ridge, SGD, Huber, Logistic, Balanced SGD) on the exact same held-out test split introduces the risk of subtle post-hoc selection bias. Final production hyperparameters and configurations should be validated using internal cross-validation on training data alone.
3. **Extreme and Potentially Anomalous Delay Values:**
   - The observed target range spans −372 to +192 days, with 10.97% of records experiencing absolute delays exceeding 30 days. Four records (−368 to −372 days) show matching month and day between delivered and recorded dates but a mismatched year ($Y-1$), strongly indicating clerical ERP date-entry errors. These extreme outliers exert substantial leverage on squared-error loss functions, inflating RMSE and penalizing linear regressors.
4. **Class Imbalance:**
   - Late shipments constitute only 11.49% of the historical dataset (11.48% in the test set). Under standard unweighted objective functions, models default to predicting the majority class ("on-time"), rendering raw accuracy completely misleading. Addressing this via class reweighting or threshold reduction successfully recovers recall but inevitably inflates false positives.
5. **Threshold Selection and Validation Requirements:**
   - Decision threshold selection fundamentally alters the operating sensitivity and precision of classification models. In operational settings, threshold tuning must be performed strictly using cross-validation or a dedicated validation split—never on test data—and requires ongoing recalibration as operational cost ratios change.
6. **Dataset Size and Generalizability Limitations:**
   - The dataset comprises 10,324 historical shipments specific to the USAID / PEPFAR Supply Chain Management System (SCMS), primarily focused on HIV/AIDS and malaria medical commodities delivered to developing nations between 2006 and 2015. Findings and learned weights may not generalize to commercial supply chains, different geographical trade lanes, or alternative product categories.
7. **Pre-Dispatch Visibility Horizon and Lack of Business Guarantees:**
   - To eliminate target leakage, models are restricted strictly to features known prior to dispatch (country, product group, vendor, scheduled delivery date). Crucial in-transit variables—such as carrier routing changes, customs clearance hold-ups, extreme weather events, and port congestion—are fundamentally unobservable at order creation. Consequently, offline historical metrics provide statistical benchmarks under past conditions, but do not guarantee future business performance during real-time supply chain shocks.

---

## F. Recommended Future Roadmap

1. **Phase 6: Time-Series Analysis & Volume Forecasting (Planned Next Phase):**
   - Aggregate shipment volume and late delivery rates into regular monthly intervals.
   - Perform seasonal decomposition, stationarity testing (Augmented Dickey-Fuller test), and evaluate an ARIMA / SARIMA baseline.
   - *Note: In accordance with project instructions, time-series forecasting is reserved for Phase 6 and is not implemented in this phase.*
2. **Advanced Tabular Modeling (Deferred Future Phase):**
   - Evaluate non-linear gradient-boosted decision trees (LightGBM, XGBoost, CatBoost) to test whether high-order feature interactions among vendors, transit corridors, and commodity types improve delay duration estimation.
   - *Note: Gradient Boosting and XGBoost are deliberately deferred to a later phase and are not implemented in this phase.*
3. **Chronological Validation Strategy:**
   - Implement temporal rolling-origin cross-validation (e.g., training on 2006–2012, testing on 2013–2015) to evaluate out-of-time generalizability.
4. **Probability Calibration & Cost-Sensitive Optimization:**
   - Calibrate predicted probabilities using Platt scaling or isotonic regression to ensure that output probabilities reflect true empirical empirical risk frequencies.
   - Construct a formal cost-loss matrix with logistics stakeholders to automate dynamic threshold selection.
5. **Data Quality Sensitivity Experiment:**
   - Run a comparative regression experiment with the 4 year-offset records sanitized to quantify the exact distortion attributable to ERP clerical errors.
