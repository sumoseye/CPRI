# PowerNext-AI 2026 — Methodology Note

**Team:** Kori Rotti
**Challenge:** The Black-Box Test Bench Challenge (CPRI & MIT Bengaluru)

---

## 1. Executive Summary and Feature Selection Strategy

The objective of this work is to predict the verified hot-spot temperature rise (Reference Parameter) of an electrical test specimen under varying operating conditions, and to classify each test record as Valid or Invalid based on measurement reliability.

A key observation from initial data exploration is that Sensor S1, S2, and S3 are already recorded as temperature rises above ambient (in degrees Celsius), and the Reference Parameter is likewise a temperature rise. This means the sensors should be used directly as rise features rather than subtracting ambient temperature a second time. Ambient temperature is retained as a separate input because it influences convective cooling rates.

Correlation analysis of the 1,000-record training set revealed that Load Current has the strongest linear relationship with the Reference Parameter (r = +0.87), followed by Sensor S2 (r = +0.70). Applied Voltage (r = +0.26), Test Duration (r = -0.01), and Ambient Temperature (r = +0.21) show weaker individual correlations but contribute meaningfully through interaction terms. Sensor S4 shows near-zero correlation with all other variables (maximum |r| = 0.07, autocorrelation = 0.03) and was identified as uninformative noise.

From the raw inputs, 35 engineered features were constructed, including apparent electrical power (V x I), thermal energy (V x I x t), Joule heating proxy (I-squared x t), spatial sensor statistics (mean, max, min, spread, standard deviation across S1-S3), logarithmic and square-root transforms of energy, power-duration interaction terms, impedance proxy (V/I), and pairwise sensor ratios to capture spatial asymmetry.

---

## 2. Anomaly Detection and Sensor Integrity Logic

### 2.1 Sensor S4 Diagnosis

Sensor S4 was evaluated using cross-correlation against S1, S2, S3, and the Reference Parameter, as well as lag-1 autocorrelation. All correlations were below 0.08 and autocorrelation was 0.03, consistent with white noise. The sensor was therefore dampened by a factor of 0.01 to prevent it from introducing variance into the models while preserving any residual structure.

### 2.2 Validity Classification (Task 01)

Validity classification uses a hybrid approach combining a gradient boosting classifier with deterministic physical rules.

The machine learning component is a HistGradientBoostingClassifier trained with L2 regularization (lambda = 1.5), early stopping, and a maximum depth of 6. Five-fold cross-validation on the training set yielded a weighted F1 score of 0.9644 (standard deviation 0.0223).

Physical safety overrides are applied after the ML prediction. Records are forced to Invalid if any of the following conditions are met: negative voltage, negative current, negative test duration, sensor readings below -10 degrees Celsius or above 400 degrees Celsius, or inter-sensor spread exceeding 200 degrees Celsius. These rules capture physically impossible conditions that the ML model might miss on unseen data.

On the 350-record test set, the classifier identified 319 Valid and 31 Invalid records (8.9% anomaly rate), which is consistent with the 13.4% invalid rate observed in the training data after accounting for the smaller test sample.

---

## 3. Thermal Regression and Validation Strategy (Task 02)

### 3.1 Training Data Filtering

The regressor was trained exclusively on the 866 records labeled Valid in the training set. The 134 Invalid records were excluded because they contain sensor malfunctions, corrupted measurements, and physically inconsistent readings that would bias the model.

### 3.2 Model Architecture

Three gradient boosting regressors were trained and evaluated using 5-fold cross-validation on the valid-only training data:

- XGBoost Regressor: RMSE = 0.9189 degrees Celsius, R-squared = 0.9927
- HistGradientBoosting Regressor: RMSE = 1.0506 degrees Celsius, R-squared = 0.9904
- LightGBM Regressor: RMSE = 1.0656 degrees Celsius, R-squared = 0.9902

The final prediction is a weighted ensemble of all three models, with weights proportional to the inverse of each model's cross-validation RMSE. XGBoost received the highest weight (36.5%) due to its superior RMSE, while HGB and LightGBM contributed 32.0% and 31.5% respectively.

### 3.3 Prediction Consistency

The predicted hot-spot temperatures on the test set range from 13.21 to 58.00 degrees Celsius with a mean of 26.41 degrees Celsius. These values are consistent with the training ground truth (range 11.92 to 61.58, mean 26.75), confirming that the model has not introduced systematic bias or scale drift.

---

## 4. Digital Twin Automation Framework

If this system were to be deployed as an automated digital twin for continuous test bench monitoring, the following four-stage pipeline would be implemented:

**Stage 1 — Data Ingestion:** Automated reading of test bench SCADA exports with dynamic column mapping, deduplication, and type validation. New records are appended to the historical database in real time.

**Stage 2 — Noise Filtering:** Multivariate iterative imputation (MICE) fills missing sensor readings. Sensor S4 and any future auxiliary channels are automatically evaluated for signal content using cross-correlation thresholds and flagged as noise if correlations fall below 0.10.

**Stage 3 — Hot-Spot Inference:** The hybrid validity classifier screens each incoming record. Valid records are passed to the ensemble regressor for hot-spot temperature prediction. Invalid records are flagged for engineering review.

**Stage 4 — Prescriptive Action:** Records are ranked by risk (invalid status combined with predicted temperature severity). The top attention cases trigger maintenance alerts mapped to specific hardware interventions based on the spatial sensor signature (for example, high S2 relative to S1 and S3 indicates outgoing terminal contact degradation requiring re-torquing or contact surface re-plating).

---

## 5. Prescriptive Circuit Redesign Matrix

| Observed Sensor Pattern | Probable Root Cause | Recommended Intervention |
|---|---|---|
| S2 significantly higher than S1 and S3 | Outgoing terminal contact resistance | Re-torque load-side clamp to 50 Nm; apply conductive contact compound |
| S1 significantly higher than S2 and S3 | Incoming terminal contact resistance | Inspect and clean incoming bushing connection; check for oxidation |
| High spread across all three sensors (>20 C) | Busbar current crowding or loose joints | Inspect bolted joint compression; consider increasing busbar cross-section |
| Uniformly elevated S1, S2, S3 | Sustained overload or inadequate cooling | Increase forced-air flow rate; verify cooling duct clearance |
| One sensor reading near zero while others are elevated | Sensor detachment or open-circuit lead | Replace RTD probe and verify DAQ channel grounding |

---

*Prepared by Team Kori Rotti for PowerNext-AI 2026, organized by CPRI with institutional partner MIT Bengaluru.*