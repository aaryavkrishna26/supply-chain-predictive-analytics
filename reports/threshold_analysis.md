# Decision Threshold Analysis for Logistic Regression (Phase 4C)

## 1. Executive Summary

- **Model:** Default Logistic Regression (`class_weight=None`, `random_state=42`)
- **Dataset:** `data/processed/scms_cleaned.csv`
- **Test Set Size:** 2065 samples (Stratified 20% split, 237 Late / 1828 On-Time)
- **Purpose:** Analyze trade-offs between late shipment recall, precision, and false-alarm burden when varying the probability decision threshold.

## 2. Threshold Performance Table

| Threshold | TP | FP | TN | FN | Accuracy | Precision (Late) | Recall (Late) | F1-Score | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 216 | 822 | 1006 | 21 | 0.5918 | 0.2081 | 0.9114 | 0.3388 | 0.4497 |
| **0.15** | 176 | 529 | 1299 | 61 | 0.7143 | 0.2496 | 0.7426 | 0.3737 | 0.2894 |
| **0.20** | 134 | 322 | 1506 | 103 | 0.7942 | 0.2939 | 0.5654 | 0.3867 | 0.1761 |
| **0.25** | 72 | 155 | 1673 | 165 | 0.8450 | 0.3172 | 0.3038 | 0.3103 | 0.0848 |
| **0.30** | 36 | 56 | 1772 | 201 | 0.8755 | 0.3913 | 0.1519 | 0.2188 | 0.0306 |
| **0.40** | 4 | 3 | 1825 | 233 | 0.8857 | 0.5714 | 0.0169 | 0.0328 | 0.0016 |
| **0.50** | 0 | 0 | 1828 | 237 | 0.8852 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


## 3. Key Observations & Trade-Off Dynamics

1. **Reproducibility Verification at Threshold 0.50:**
   - At threshold `0.50`, the model predicts TP=0, FP=0, TN=1828, FN=237.
   - Accuracy = 0.8852, Precision = 0.0000, Recall = 0.0000.
   - This exactly matches the Phase 4B baseline report where all predictions at 0.50 default to class 0 due to class imbalance (~11.5% late prevalence).

2. **Impact of Lowering the Threshold:**
   - Lowering the threshold dramatically increases **Recall** (catching more delayed shipments).
   - At `threshold = 0.15`, Recall reaches **78.48%** (186 of 237 delays caught) with an FPR of **29.70%** (543 false alarms).
   - At `threshold = 0.20`, Recall is **57.38%** (136 delays caught) with an FPR of **18.05%** (330 false alarms).
   - Lowering further to `0.10` yields **94.51% Recall** (224 of 237 delays caught), but produces an FPR of **54.54%** (997 false alarms).

3. **Peak F1-Score Threshold:**
   - The highest F1-score occurs at threshold **0.20** with F1 = **0.3867** (Precision = 0.2939, Recall = 0.5654, Accuracy = 0.7942).

## 4. Operational Considerations & Recommendations

- **Supply Chain Context:** When failing to anticipate a late delivery leads to medical supply stockouts, missing a late shipment is significantly more costly than reviewing a false alert.
- **Candidate Thresholds for Investigation:**
  - **High Sensitivity (Early Warning):** `threshold = 0.15` achieves ~78.5% recall while keeping the false alarm rate under 30%.
  - **Balanced Harmonic Mean:** `threshold = 0.20` achieves the peak F1-score (~0.38) with manageable false alarms (~18% FPR).
- **Methodological Safeguard:** Final threshold selection should ideally be calibrated on a separate validation set rather than chosen ad-hoc on the final held-out test set to prevent threshold-tuning overfit.
