# PowerNext-AI 2026 — Black-Box Test Bench Challenge

**Team:** Kori Rotti
**Organized by:** CPRI with institutional partner MIT Bengaluru

## 1. Overview

This repository contains our solution for the Screening Round of PowerNext-AI 2026.

The task is to analyse historical test bench records and develop a reproducible program that performs three functions:

1. Identify abnormal or invalid test records in new data.
2. Predict the Reference Parameter, defined as the verified temperature rise at the critical hot-spot of the test specimen.
3. Generate an automated summary of the test results.

Input data is provided as a single Excel workbook with two sheets: Training_Data with 1000 records and Test_Data with 350 records.

The program reads the workbook and produces two output files: Kori_Rotti.csv and summary.json.

## 2. Repository Contents

- solution.py — Main program that runs the complete pipeline.
- Methodology_Note.md — Short methodology note describing the approach.
- Kori_Rotti.csv — Predicted Reference Parameter and Valid / Invalid label for each test record.
- summary.json — Automated summary of the test results.
- README.md — This file.
- .gitignore — Excludes the virtual environment and local system files.

Note: The input Excel workbook is not included in this repository.

## 3. Method Summary

### 3.1 Data preparation

Column names are standardized to remove units and formatting differences. Duplicate rows are removed from the training set. Missing sensor values are filled using iterative multivariate imputation, with KNN imputation as a fallback.

### 3.2 Feature engineering

A set of 27 features is constructed for regression based on the available inputs:

- Apparent power: Applied_Voltage x Load_Current
- Thermal energy: power x Test_Duration
- Joule heating proxy: Load_Current squared x Test_Duration
- Spatial statistics across Sensor S1, S2 and S3: mean, maximum, minimum, spread, standard deviation and median
- Logarithmic and square root transforms of power, energy and duration
- Power-duration interaction terms and impedance proxy (voltage divided by current)
- Ambient temperature retained as a separate input affecting cooling

Sensor S1, S2 and S3 are provided as temperature rises above ambient, so they are used directly. Ambient temperature is not subtracted a second time.

Sensor S4 was assessed using cross-correlation and lag-1 autocorrelation. The maximum absolute correlation with other variables was 0.07 and the autocorrelation was 0.03, consistent with white noise. It was therefore excluded from the feature set.

### 3.3 Task 01: Validity classification

Validity is determined using a combination of a classifier and physical checks.

The classifier is a HistGradientBoostingClassifier trained on the labeled training records. Input features include the engineered physical features and spatial z-scores that describe how far each sensor deviates from the local mean.

After the classifier prediction, physical rules are applied. A record is marked Invalid if voltage, current or duration is negative, if a sensor reading is below -10 C or above 400 C, or if the spread between sensors exceeds 200 C.

### 3.4 Task 02: Reference Parameter regression

The regressor is trained only on records labeled Valid in the training set (866 records). Invalid records are excluded to avoid fitting sensor errors.

Three gradient boosting models are trained with 5-fold cross-validation: GradientBoostingRegressor, XGBoost and LightGBM. Predictions from the five folds are averaged for each model, and the three model outputs are combined using inverse-variance weighting based on cross-validation error.

A physical lower bound is applied to the final predictions. For each test record, the predicted hot-spot rise is constrained to be at least the maximum of S1, S2 and S3 minus 0.5 C, and at least 0 C.

## 4. Cross-Validation Results

Results below are from 5-fold cross-validation on the training data:

Regression (trained on 866 valid records):
- Combined ensemble RMSE: 0.9060 C
- Combined ensemble MAE: 0.4838 C
- Combined ensemble R-squared: 0.9929

Classification (trained on 1000 labeled records):
- Weighted F1 score: 0.9718
- Accuracy: 0.9730
- Precision on Invalid class: 0.9820

Test set predictions range from 13.26 C to 57.07 C with a mean of 26.42 C, which is consistent with the training target distribution.

## 5. How to Run

Requirements: Python 3.8 or later.

Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, openpyxl

Steps:

```bash
git clone https://github.com/sumoseye/CPRI.git
cd CPRI
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost lightgbm scipy openpyxl
```

Place CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx in the same folder as solution.py, then run:

```bash
python solution.py
```

The run takes under a minute and creates Kori_Rotti.csv and summary.json in the current folder.

## 6. Outputs

Kori_Rotti.csv contains 350 rows with three columns:

- Test_ID
- Predicted_Reference_Parameter
- Valid / Invalid

summary.json contains:

- records_analysed
- abnormal_invalid_records
- min_predicted_reference_parameter
- max_predicted_reference_parameter
- average_predicted_reference_parameter
- top_3_attention_test_ids
- approach_explanation

## 7. Assumptions and Notes

1. Sensor S1, S2, S3 and the Reference Parameter are already expressed as rises above ambient in degrees Celsius.
2. Ambient temperature is treated as a boundary condition for cooling rather than a baseline to subtract.
3. Invalid records are assumed to reflect measurement or logging faults, not genuine operating regimes, and are excluded from regression training.
4. Thermal properties of the test specimen are assumed to be consistent between training and test records.
5. All random seeds are fixed for reproducibility.

## 8. Automation Outline

For continuous use, the same steps can be run as a scheduled pipeline: ingest new test records, apply imputation and channel checks, run the validity classifier, run the regression ensemble for valid records, and write the updated CSV and JSON outputs.

---

Prepared by Team Kori Rotti for PowerNext-AI 2026.
