# Model Improvement Summary Report (Phase 6A/6B)

## 1. Objective

This report summarises the results of the Phase 6 model improvement experiments, comparing
tree-based ensemble classifiers and regressors (RandomForestClassifier, ExtraTreesClassifier,
RandomForestRegressor, ExtraTreesRegressor) against all prior baselines established in Phases 4-5.

**Key principle:** All hyperparameter tuning and threshold calibration was performed exclusively
on training data. The held-out test set (20%, 2,065 samples) was evaluated once per experiment,
after tuning was complete. Results are real computed values — not estimated.

---

## 2. Classification Results Summary (`is_delayed`)

**Test set:** 2,065 samples | 237 Late (11.48%) | 1,828 On-Time | Stratified 80/20 split, random_state=42

### Full Comparison Table

| Model | Phase | Thr | Recall | Precision | F1 | Avg Prec | ROC-AUC | FPR | TP | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dummy (Most Frequent) | 4 | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.5000 | 0.0000 | 0 | 0 |
| LR (Default) | 4 | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.3099 | 0.8062 | 0.0000 | 0 | 0 |
| LR (Balanced) | 4 | 0.50 | 0.8776 | 0.2208 | 0.3528 | 0.3060 | 0.8043 | 0.4015 | 208 | 734 |
| LR (Default, thr=0.20) | 4C | 0.20 | 0.5654 | 0.2939 | 0.3867 | 0.3099 | 0.8062 | 0.1761 | 134 | 322 |
| SGDClassifier (Balanced) | 5E | 0.50 | 0.8734 | 0.2154 | 0.3456 | 0.2562 | 0.7822 | — | 207 | 754 |
| **RandomForest (thr=0.50)** | **6A** | **0.50** | **0.4135** | **0.4516** | **0.4317** | **0.4006** | **0.8576** | **0.0651** | **98** | **119** |
| **RandomForest (thr=0.40, CV-tuned)** | **6A** | **0.40** | **0.6034** | **0.3575** | **0.4490** | **0.4006** | **0.8576** | **0.1406** | **143** | **257** |
| ExtraTrees (thr=0.50, CV-tuned) | 6A | 0.50 | 0.6878 | 0.2932 | 0.4111 | 0.3769 | 0.8442 | 0.2150 | 163 | 393 |

### Key Classification Findings

#### What improved
- **ROC-AUC:** RandomForest raised ROC-AUC from 0.8062 (prior best) to **0.8576** — a
  meaningful improvement in probability ranking quality without any threshold tuning.
- **F1-Score:** RandomForest (thr=0.40, CV-tuned) achieves F1=**0.4490** vs 0.3528 (Balanced LR)
  — a **+27% improvement** in the balanced recall/precision trade-off.
- **Average Precision (PR-AUC):** RandomForest improves from 0.3099 to **0.4006**, indicating
  better ordering of positive cases across all thresholds.
- **False-Positive Rate at comparable recall:** At 60.3% recall, RandomForest generates 257 false
  alarms (FPR=14.1%) vs 322 false alarms for LR at 56.5% recall (FPR=17.6%) — better precision
  at similar or higher recall.

#### Operating point trade-offs
| Scenario | Recommended model | Recall | FPR | F1 |
| :--- | :--- | :---: | :---: | :---: |
| Maximum recall (catch every delay) | LR Balanced | 87.8% | 40.2% | 0.353 |
| Best operational balance (F1) | **RandomForest thr=0.40** | **60.3%** | **14.1%** | **0.449** |
| Precision-sensitive (low false alarms) | **RandomForest thr=0.50** | **41.4%** | **6.5%** | **0.432** |
| Moderate recall with low alarm budget | LR Default thr=0.20 | 56.5% | 17.6% | 0.387 |
| High recall with moderate alarm budget | ExtraTrees thr=0.50 | 68.8% | 21.5% | 0.411 |

#### What did not change
- Logistic Regression at threshold 0.50 still predicts all on-time (0 recall).
- Balanced Logistic Regression still has the highest absolute recall (87.8%) but at the
  cost of 734 false alarms (40.2% FPR) — only justified when stockout cost far exceeds
  inspection labor cost.

---

## 3. Regression Results Summary (`delay_days`)

**Test set:** 2,065 samples | min=-372d, max=165d, mean=-6.06d, median=0.0d, std=28.61d
**Extreme subgroup** (|delay_days|>30): 229 samples (11.09%)

### Full Comparison Table

| Model | Phase | MAE (d) | RMSE (d) | R2 | MedAE (d) | Ext. MAE (d) | Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Dummy (Mean) | 5B | 14.48 | 28.60 | 0.0000 | 6.01 | 69.98 | OK |
| Dummy (Median) | 5B | 11.47 | 29.24 | -0.0450 | 0.00 | 73.10 | OK |
| Linear Regression | 5B | 13.98 | 26.90 | 0.1153 | 6.46 | 61.51 | OK |
| Ridge (alpha=1.0) | 5C | 13.96 | 26.90 | 0.1156 | 6.46 | 61.54 | OK |
| Huber Regressor | 5C | 11.13 | 28.68 | -0.0053 | 0.00 | 71.18 | Warning: L-BFGS iter limit |
| SGDRegressor | 5D | 14.16 | 26.95 | 0.1122 | 6.74 | 61.42 | OK |
| **RandomForest (tuned)** | **6B** | **9.79** | **21.78** | **0.4199** | **2.31** | **44.07** | **OK** |
| ExtraTrees (tuned) | 6B | 9.97 | 22.78 | 0.3657 | 1.82 | 42.35 | OK |

### Key Regression Findings

#### What improved — substantially
- **MAE:** RandomForest achieves MAE=**9.79 days** vs 13.96 days (Ridge) — a **29.9% reduction**.
  ExtraTrees achieves 9.97 days, also a meaningful improvement.
- **RMSE:** RandomForest achieves RMSE=**21.78 days** vs 26.90 days (Ridge/OLS) — a **19.0%
  reduction**. This is not driven by cherry-picking; both RF and ET improve RMSE substantially.
- **R²:** RandomForest achieves R²=**0.4199** vs 0.1156 (Ridge) — explaining **3.6× more variance**
  with the same feature set. This demonstrates that non-linear feature interactions (e.g., vendor ×
  country × product type) are genuinely informative for delay duration prediction.
- **Extreme-delay subgroup (|delay_days|>30):** RandomForest achieves MAE=44.07d and ExtraTrees
  42.35d, vs 61.51d for Linear Regression — a **31-28% improvement** on the hardest cases.
- **MedAE:** Both ensembles achieve MedAE of 1.82-2.31 days vs 6.46 days for OLS/Ridge, indicating
  dramatically better typical-case predictions.

#### MAE vs RMSE interpretation
- RandomForest has both the lowest MAE (9.79d) and the lowest RMSE (21.78d), making it the
  dominant regressor on this test set across both metrics.
- ExtraTrees achieves the lowest MedAE (1.82d) and lowest extreme-delay MAE (42.35d) but
  higher RMSE (22.78d) than RandomForest — it is marginally better on the typical/median
  prediction and extreme-delay MAE, but worse at avoiding individual large blunders
  (higher RMSE indicates more variance in prediction errors).
- **A model with lower MAE but higher RMSE** is better at typical-case accuracy but produces
  more occasional large-magnitude blunders. For supply chain buffer planning, MAE is generally
  the more operationally relevant metric; for risk management of catastrophic single-shipment
  failures, RMSE matters more.

#### Remaining limitations
- R²=0.42 means 58% of delay-duration variance remains unexplained by pre-dispatch features.
  The dominant in-transit disruption factors (customs, weather, port congestion) are inherently
  unobservable at order creation.
- Both tree ensembles were tuned with max_features=1.0 (all features used at each split),
  suggesting the best signal-to-noise ratio comes from considering all features simultaneously
  rather than random sub-sampling.
- The 4 year-entry typo records (delay_days≈-370) remain in the dataset and continue to
  distort RMSE, though the ensemble's ability to memorise rare patterns may partially absorb them.

---

## 4. Improvement vs Prior Baselines

### Classification improvement over best prior model

| Metric | Prior Best | Prior Model | New Best | New Model | Change |
| :--- | :---: | :--- | :---: | :--- | :---: |
| F1-Score | 0.3528 | LR Balanced | **0.4490** | RF thr=0.40 | +27.3% |
| ROC-AUC | 0.8062 | LR Default | **0.8576** | RandomForest | +6.4% |
| Avg Precision | 0.3099 | LR Default | **0.4006** | RandomForest | +29.3% |
| FPR at ~60% recall | 17.6% | LR thr=0.20 | **14.1%** | RF thr=0.40 | -3.5pp |

### Regression improvement over best prior model (Ridge/OLS)

| Metric | Prior Best | Prior Model | New Best | New Model | Change |
| :--- | :---: | :--- | :---: | :--- | :---: |
| MAE | 11.13d* | Huber (warned) | **9.79d** | RandomForest | -12.1% vs Huber / -29.9% vs Ridge |
| RMSE | 26.90d | Ridge/OLS | **21.78d** | RandomForest | -19.0% |
| R² | 0.1156 | Ridge | **0.4199** | RandomForest | +263% relative |
| Extreme MAE | 61.51d | Linear Reg. | **42.35d** | ExtraTrees | -31.1% |

*Huber had a convergence warning; Ridge/OLS (13.96d MAE) is the prior reliable best.

---

## 5. Model-Selection Decisions

### Classification

**For maximum recall (catch every delay, accept false alarms):**
- **Logistic Regression (Balanced)** — Recall=87.8%, FPR=40.2%, F1=0.353
- Appropriate when: stockout cost >> alarm inspection cost (ratio > ~4:1)

**For best practical balance (recommended for most deployments):**
- **RandomForest (thr=0.40, CV-tuned)** — Recall=60.3%, FPR=14.1%, F1=0.449, ROC-AUC=0.8576
- Appropriate when: inspection capacity is limited and precision matters
- The CV-selected threshold (0.40) was determined using training-only cross-validation;
  it is not guaranteed optimal on unseen future data but was not selected by inspecting the test set.

**For precision-sensitive operations (lowest false alarm rate):**
- **RandomForest (thr=0.50)** — Recall=41.4%, FPR=6.5%, F1=0.432, Precision=45.2%
- Only 119 false alarms vs 734 for Balanced LR; suitable where alert reviews are expensive.

### Regression

**Recommended:** **RandomForestRegressor (tuned)** — MAE=9.79d, RMSE=21.78d, R²=0.4199
- Dominates all prior models on MAE, RMSE, and R² simultaneously.
- No convergence warnings.
- Best parameters: 400 trees, max_depth=20, min_samples_leaf=2, max_features=1.0.

**Runner-up:** ExtraTreesRegressor — MAE=9.97d, MedAE=1.82d (slightly lower typical error),
  extreme-delay MAE=42.35d (slightly better on tail events), but higher RMSE (22.78d).

---

## 6. Evaluation Integrity Notes

- **No test-set leakage during tuning:** RandomizedSearchCV used 3-fold CV on X_train only.
  Final test set was evaluated once per model, after all tuning was complete.
- **Threshold calibration:** Decision thresholds were selected using 5-fold CV predict_proba
  on X_train; the test set was not consulted for threshold selection.
- **Consistent split:** Both experiments use the identical 80/20 split (random_state=42) as
  Phases 4-5, enabling direct metric comparison.
- **Caveat — repeated test-set evaluation:** Comparing all phases (4, 5, and now 6) on the
  same test partition introduces mild post-hoc selection risk. Improvements should be treated
  as directional findings pending validation on a temporally held-out cohort.
- **Target integrity:** No extreme delay_days values were removed or modified.
- **No feature leakage:** Same conservative leakage exclusion list as all prior phases.

---

## 7. Recommended Next Steps

1. **Phase 6 (Time-Series):** Aggregate monthly shipment volumes, run ADF stationarity tests,
   decompose seasonality, and evaluate ARIMA/SARIMA baselines.
2. **Chronological validation:** Refit RandomForest classifier and regressor using a temporal
   train/test split (e.g. 2006-2012 train, 2013-2015 test) to verify out-of-time generalizability.
3. **Probability calibration:** Apply Platt scaling or isotonic regression to RandomForest
   classification probabilities, which are known to be poorly calibrated in tree ensembles.
4. **Sensitivity analysis:** Re-run RandomForestRegressor with the 4 ERP year-typo records
   removed to quantify their distortion of RMSE.
5. **Deeper hyperparameter search:** Expand RandomizedSearchCV to n_iter=50 or use BayesSearchCV
   to explore the interaction between max_depth, min_samples_leaf, and max_features more thoroughly.
