# PowerNext-AI 2026 — Methodology Note

**Team:** Kori Rotti  
**Challenge:** The Black-Box Test Bench Challenge (CPRI & MIT Bengaluru)

---

## 1. Executive Summary and Feature Selection Strategy

The objective of this work is to predict the verified hot-spot temperature rise (Reference Parameter) of an electrical test specimen under varying operating conditions, and to classify each test record as Valid or Invalid based on measurement reliability.

Crucial domain insight from data exploration confirmed that Sensor S1, S2, and S3 are already recorded as temperature rises above ambient (in °C), and the Reference Parameter is likewise a temperature rise. Sensors are thus utilized directly as rise features rather than double-subtracting ambient temperature. Ambient temperature is retained as an independent feature affecting external convective cooling.

Correlation analysis of the 1,000-record training set revealed that Load Current has the strongest linear relationship with the Reference Parameter (r = +0.87), followed by Sensor S2 (r = +0.70). Sensor S4 demonstrated unphysical white noise characteristics (max correlation r = 0.0707, lag-1 autocorrelation = 0.0300) and was completely excluded from feature space.

From the raw inputs, a refined 27-feature regression contract was constructed, including apparent electrical power (P = V x I), thermal energy proxy (E = P x t), Joule heating (I^2 x t), spatial sensor statistics (mean, max, min, spread, standard deviation across S1-S3), z-score spatial anomaly descriptors, power-duration saturation terms, impedance proxy (V/I), and pairwise sensor asymmetry ratios.

---

## 2. Anomaly Detection and Sensor Integrity Logic

### 2.1 Sensor S4 Diagnosis
Sensor S4 cross-correlation against S1, S2, S3, and Reference Parameter remained below 0.08 with an autocorrelation of 0.03. Demonstrating near-zero mutual information with true system state, S4 was removed to prevent tree-split noise overfitting.

### 2.2 Validity Classification (Task 01)
Validity classification utilizes a hybrid architecture combining spatial z-score feature representations with a gradient boosting classifier and deterministic physical safety rules.

On 5-fold stratified cross-validation across the training set, the hybrid classifier achieved an out-of-fold accuracy of 99.80%, precision of 98.53%, recall of 100.0% (134/134 invalid records identified), and a weighted F1-score of 0.9926 (±0.0091).

Physical safety overrides enforce invalid status if voltage, current, or duration are negative, if sensor rises exceed 400°C, or if inter-sensor spread exceeds 200°C. On the 350-record test set, the classifier identified 304 Valid and 46 Invalid records (13.1% anomaly rate), closely matching the historical training distribution (13.4%).

---

## 3. Thermal Regression and Validation Strategy (Task 02)

### 3.1 Strict Valid-Only Training Protocol
The regressor was trained strictly on the 866 verified Valid training records, isolating the models from sensor disconnects and corruption artifacts.

### 3.2 Regression Performance
Using 5-fold cross-validation on the filtered valid dataset, the optimized Gradient Boosting Regressor achieved:
- Root Mean Squared Error (RMSE): 0.6976 °C (per-fold mean: 0.6955 ± 0.0526 °C)
- Mean Absolute Error (MAE): 0.4664 °C
- Coefficient of Determination (R²): 0.9958

### 3.3 Prediction Consistency
Predicted hot-spot temperature rises on the 350 test instances range from 12.96 °C to 60.74 °C with a mean of 26.41 °C, aligning with the ground-truth training distribution (range: 11.92 °C to 61.58 °C, mean: 26.75 °C) and verifying absence of scale drift.

---

## 4. Digital Twin Automation Framework

**Stage 1 — Ingestion:** Automated reading of test bench SCADA streams with dynamic header normalization, deduplication, and type validation.

**Stage 2 — Noise Filtering:** Multivariate iterative imputation (MICE) resolves missing readings. Auxiliary channels are subjected to automated autocorrelation screening to prune white-noise sensors.

**Stage 3 — Inference Engine:** Spatial z-score hybrid classifier gates record validity. Valid records pass to the 0.69°C RMSE gradient boosting regressor for hot-spot inference; anomalies are flagged for diagnostic review.

**Stage 4 — Prescriptive Action:** Risk-ranked test summaries are generated automatically. Top attention IDs trigger automated maintenance work orders mapped to spatial sensor signatures.

---

## 5. Prescriptive Circuit Redesign Matrix

| Observed Sensor Signature | Root Cause | Recommended Hardware Intervention |
|---|---|---|
| S2 significantly higher than S1 and S3 | Outgoing terminal contact degradation | Re-torque load clamp to 50 Nm; apply silver-plated conductive compound |
| S1 significantly higher than S2 and S3 | Incoming terminal contact resistance | Clean and deoxidize incoming bushing interface; inspect joint compression |
| High spread across S1–S3 (> 20 °C) | Busbar current crowding / localized joint fault | Verify bolted joint torque; upgrade busbar cross-sectional area |
| Uniformly elevated S1, S2, S3 | Sustained overload or restricted ventilation | Increase forced-air flow by 25%; inspect cooling duct clearances |
| Single sensor floating near zero | Sensor detachment or open-circuit lead | Replace RTD transducer and verify DAQ ground isolation |

---

*Prepared by Team Kori Rotti for PowerNext-AI 2026 (CPRI & MIT Bengaluru).*