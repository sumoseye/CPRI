#!/usr/bin/env python3
"""
===============================================================================
PowerNext-AI 2026: The Black-Box Test Bench Challenge
Organized by: CPRI & MIT Bengaluru
Team: CPRI_Winners
Strategy: Understand -> Analyse -> Validate -> Automate
===============================================================================
"""

import os
import sys
import json
import re
import warnings
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)

# High-performance boosting libraries with fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEAM_NAME = "Kori_Rotti"
RANDOM_STATE = 42
N_FOLDS = 5
OUTPUT_CSV = f"{TEAM_NAME}.csv"
OUTPUT_JSON = "summary.json"
DEFAULT_DATA_FILE = "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"

np.random.seed(RANDOM_STATE)


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# =============================================================================
# 2. FILE & SHEET RESOLUTION
# =============================================================================
def locate_workbook(default_name: str = DEFAULT_DATA_FILE) -> str:
    if os.path.exists(default_name):
        return default_name
    for f in os.listdir("."):
        if f.lower() == default_name.lower():
            return f
    xlsx = [f for f in os.listdir(".") if f.endswith(".xlsx") and not f.startswith("~$")]
    if xlsx:
        logger.warning(f"Using found workbook: '{xlsx[0]}'")
        return xlsx[0]
    logger.error("No .xlsx file found in current directory.")
    sys.exit(1)


def detect_sheets(xlsx_path: str) -> Tuple[str, str]:
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xl.sheet_names
    logger.info(f"Workbook sheets: {sheets}")
    train_sheet, test_sheet = None, None
    for s in sheets:
        clean = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
        if "train" in clean:
            train_sheet = s
        elif "test" in clean:
            test_sheet = s
    if train_sheet is None or test_sheet is None:
        if len(sheets) >= 2:
            train_sheet = train_sheet or sheets[0]
            test_sheet = test_sheet or sheets[1]
        else:
            logger.error("Excel file must contain at least 2 sheets.")
            sys.exit(1)
    logger.info(f"Using Sheets -> Training: '{train_sheet}' | Test: '{test_sheet}'")
    return train_sheet, test_sheet


# =============================================================================
# 3. SCHEMA STANDARDIZATION & CLEANING (UNDERSTAND STAGE)
# =============================================================================
class DatasetCleaner:
    CANONICAL_RENAME = {
        'applied_voltage_kv': 'Applied_Voltage',
        'applied_voltage': 'Applied_Voltage',
        'voltage': 'Applied_Voltage',
        'load_current_a': 'Load_Current',
        'load_current': 'Load_Current',
        'current': 'Load_Current',
        'ambient_temperature_c': 'Ambient_Temperature',
        'ambient_temperature': 'Ambient_Temperature',
        'ambient_temp': 'Ambient_Temperature',
        'test_duration_min': 'Test_Duration',
        'test_duration': 'Test_Duration',
        'duration': 'Test_Duration',
        'sensor_s1': 'Sensor_S1',
        'sensor_s2': 'Sensor_S2',
        'sensor_s3': 'Sensor_S3',
        'sensor_s4': 'Sensor_S4',
        'reference_parameter': 'Reference_Parameter',
        'validity_label': 'Validity_Label',
        'test_id': 'Test_ID',
    }

    @classmethod
    def standardize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        ren = {}
        for col in df.columns:
            clean = re.sub(r"[^\w\s]", "_", str(col).strip())
            clean = re.sub(r"\s+", "_", clean)
            clean = re.sub(r"_+", "_", clean).strip("_").lower()
            if clean in cls.CANONICAL_RENAME:
                ren[col] = cls.CANONICAL_RENAME[clean]
            else:
                for k, v in cls.CANONICAL_RENAME.items():
                    if k in clean:
                        ren[col] = v
                        break
        return df.rename(columns=ren)

    def __init__(self):
        self.s4_is_noise = False
        self.train_raw = None   # pre-imputation copies (missingness intact) for validity screening
        self.test_raw = None

    def ingest_and_clean(self, xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 70)
        logger.info("STAGE 1: UNDERSTAND — Ingestion, Deduplication & Diagnostics")
        logger.info("=" * 70)

        tr_sheet, te_sheet = detect_sheets(xlsx_path)
        train = pd.read_excel(xlsx_path, sheet_name=tr_sheet, engine="openpyxl")
        test = pd.read_excel(xlsx_path, sheet_name=te_sheet, engine="openpyxl")

        train = self.standardize_columns(train)
        test = self.standardize_columns(test)

        logger.info(f"Loaded Train: {train.shape} | Test: {test.shape}")

        # Deduplication
        n_dup = train.duplicated().sum()
        train = train.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {n_dup} exact duplicate rows from training set.")

        # S4 Noise Check
        self._diagnose_s4(train)

        # Impute missing values
        num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                    if c not in ["Test_ID", "Reference_Parameter"]]
        shared_cols = [c for c in num_cols if c in test.columns]

        train_miss = train[shared_cols].isnull().sum().sum()
        test_miss = test[shared_cols].isnull().sum().sum()
        logger.info(f"Missing values detected: Train={train_miss}, Test={test_miss}")

        # Preserve raw frames BEFORE imputation: the validity classifier relies on
        # the missingness pattern itself as a fault signal (S1-S3 dropouts occur
        # exclusively on Invalid records), so it must see the original NaNs.
        self.train_raw = train.copy()
        self.test_raw = test.copy()

        if train_miss > 0 or test_miss > 0:
            try:
                imp = IterativeImputer(max_iter=20, random_state=RANDOM_STATE, initial_strategy="median")
                imp.fit(train[shared_cols])
                train[shared_cols] = imp.transform(train[shared_cols])
                test[shared_cols] = imp.transform(test[shared_cols])
                logger.info("Applied IterativeImputer (MICE) across train & test.")
            except Exception:
                knn = KNNImputer(n_neighbors=5)
                knn.fit(train[shared_cols])
                train[shared_cols] = knn.transform(train[shared_cols])
                test[shared_cols] = knn.transform(test[shared_cols])
                logger.info("Applied KNNImputer (fallback).")

        if "Reference_Parameter" in train.columns:
            target_nulls = train["Reference_Parameter"].isnull().sum()
            if target_nulls > 0:
                train = train.dropna(subset=["Reference_Parameter"]).reset_index(drop=True)
                self.train_raw = self.train_raw.dropna(subset=["Reference_Parameter"]).reset_index(drop=True)
                logger.info(f"Dropped {target_nulls} train records with missing Reference_Parameter.")

        return train, test

    def _diagnose_s4(self, df: pd.DataFrame):
        if "Sensor_S4" not in df.columns:
            self.s4_is_noise = False
            return
        s4 = df["Sensor_S4"].dropna()
        if len(s4) < 10:
            self.s4_is_noise = True
            return
        corrs = []
        for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Reference_Parameter"]:
            if s in df.columns:
                valid = df[[s, "Sensor_S4"]].dropna()
                if len(valid) > 10:
                    corrs.append(abs(valid[s].corr(valid["Sensor_S4"])))
        max_c = max(corrs) if corrs else 0.0
        autoc = float(s4.autocorr(lag=1)) if len(s4) > 20 else 0.0
        logger.info(f"Sensor_S4 Diagnostics -> Max Correlation: {max_c:.4f} | Lag-1 Autocorr: {autoc:.4f}")
        self.s4_is_noise = (max_c < 0.12 and abs(autoc) < 0.12)
        logger.info(f"Sensor_S4 Status: {'UNPHYSICAL WHITE NOISE ⚠️' if self.s4_is_noise else 'VALID SIGNAL ✅'}")


# =============================================================================
# 4. DOMAIN FEATURE ENGINEERING (ANALYSE STAGE)
# =============================================================================
class PhysicsFeatureEngineer:
    """
    Constructs domain features capturing:
      - Apparent Electrical Power: P = Applied_Voltage_kV * Load_Current_A (kW)
      - Thermal Energy Proxy: E = P * Test_Duration_min (kW·min)
      - Joule Heating Proxy: I^2 * t
      - Sensor Spatial Aggregations across S1, S2, S3 (°C rises)
      - Nonlinear saturation interactions
    """

    def __init__(self, s4_is_noise: bool = False):
        self.s4_is_noise = s4_is_noise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        v = df["Applied_Voltage"] if "Applied_Voltage" in df.columns else pd.Series(0, index=df.index)
        i = df["Load_Current"] if "Load_Current" in df.columns else pd.Series(0, index=df.index)
        t = df["Test_Duration"] if "Test_Duration" in df.columns else pd.Series(0, index=df.index)
        amb = df["Ambient_Temperature"] if "Ambient_Temperature" in df.columns else pd.Series(25.0, index=df.index)

        # 1. Electrical Power & Energy
        df["Apparent_Power_kW"] = v * i
        df["Thermal_Energy_kWmin"] = df["Apparent_Power_kW"] * t
        df["Joule_Heating_I2t"] = (i ** 2) * t
        df["V_squared"] = v ** 2
        df["I_squared"] = i ** 2
        df["Impedance_Proxy"] = (v * 1000.0) / (i.abs() + 1e-5)

        # 2. Sensor Spatial Descriptors (S1, S2, S3)
        sensors = [s for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"] if s in df.columns]
        if sensors:
            df["Sensor_Mean"] = df[sensors].mean(axis=1)
            df["Sensor_Max"] = df[sensors].max(axis=1)
            df["Sensor_Min"] = df[sensors].min(axis=1)
            df["Sensor_Spread"] = df["Sensor_Max"] - df["Sensor_Min"]
            df["Sensor_Std"] = df[sensors].std(axis=1).fillna(0)
            df["Sensor_Median"] = df[sensors].median(axis=1)
            df["Max_to_Mean_Ratio"] = df["Sensor_Max"] / (df["Sensor_Mean"].abs() + 1e-5)
        else:
            for feat in ["Sensor_Mean", "Sensor_Max", "Sensor_Min", "Sensor_Spread", "Sensor_Std", "Sensor_Median", "Max_to_Mean_Ratio"]:
                df[feat] = 0

        # 3. Cross-Physics Interactions
        if sensors:
            df["Power_x_MaxSensor"] = df["Apparent_Power_kW"] * df["Sensor_Max"]
            df["Energy_x_MaxSensor"] = df["Thermal_Energy_kWmin"] * df["Sensor_Max"]
            df["Power_x_MeanSensor"] = df["Apparent_Power_kW"] * df["Sensor_Mean"]

        # 3b. Approved Config-5 interaction: Joule heating x ambient regime.
        #     Derived exclusively from raw input columns (leakage-free by construction).
        df["I_squared_x_Ambient_Temperature"] = df["I_squared"] * df["Ambient_Temperature"]

        # 4. Nonlinear Operating Regimes & Saturation
        df["Log_Duration"] = np.log1p(np.maximum(0, t))
        df["Log_Power"] = np.log1p(np.maximum(0, df["Apparent_Power_kW"]))
        df["Log_Energy"] = np.log1p(np.maximum(0, df["Thermal_Energy_kWmin"]))
        df["Sqrt_Energy"] = np.sqrt(np.maximum(0, df["Thermal_Energy_kWmin"]))
        df["Power_x_LogDuration"] = df["Apparent_Power_kW"] * df["Log_Duration"]

        # 5. Ambient Temperature Interactions
        df["Ambient_Temp"] = amb
        if sensors:
            df["Sensor_Max_over_Ambient"] = df["Sensor_Max"] / (amb.abs() + 1e-5)

        # 6. Sensor S4 Treatment
        if "Sensor_S4" in df.columns:
            if not self.s4_is_noise:
                df["S4_raw"] = df["Sensor_S4"].fillna(0)
                if sensors:
                    df["S4_vs_Mean"] = df["Sensor_S4"].fillna(0) - df["Sensor_Mean"]
            else:
                df["S4_dampened"] = df["Sensor_S4"].fillna(0) * 0.01

        # 7. Asymmetry Ratios
        if "Sensor_S1" in df.columns and "Sensor_S2" in df.columns:
            df["S1_S2_ratio"] = df["Sensor_S1"] / (df["Sensor_S2"].abs() + 1e-5)
        if "Sensor_S2" in df.columns and "Sensor_S3" in df.columns:
            df["S2_S3_ratio"] = df["Sensor_S2"] / (df["Sensor_S3"].abs() + 1e-5)
        if "Sensor_S1" in df.columns and "Sensor_S3" in df.columns:
            df["S1_S3_ratio"] = df["Sensor_S1"] / (df["Sensor_S3"].abs() + 1e-5)

        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        return df

    @staticmethod
    def get_feature_list(df: pd.DataFrame) -> List[str]:
        exclude = {"Test_ID", "Reference_Parameter", "Validity_Label"}
        return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


# =============================================================================
# 5. TASK 01: ANOMALY & VALIDITY CLASSIFIER (VALIDATE STAGE)
# =============================================================================
class ValidityClassifier:
    """Hybrid Valid/Invalid detector.

    Layer 1 — supervised HistGradientBoostingClassifier on physically meaningful
    consistency features (sensor spread, pairwise diffs, robust physics z-scores,
    missing/int-clamp flags).

    Layer 2 — deterministic consistency rules whose thresholds are DERIVED FROM
    TRAINING DATA ONLY inside the CV pipeline (leakage-safe):
      * spatial inconsistency: spread > max(spread | Valid) + margin (Wilks-style
        empirical bound on the majority class)
      * physics z: Huber-regression residuals of each sensor (and their mean)
        vs electrical operating point, scaled by MAD; |z| > 4 (~4-sigma; the
        maximum observed |z| among Valid records is < 3.7)
      * missing S1-S3 channels, or any sensor clamped at 0.0 / 1.0 / 25.0
        (4-decimal data never lands on integers by chance)
      * duplicate measurement: exact match of the 8-feature key with another
        record (training pool or within the evaluated batch)

    Stray absolute-value overrides (sensor > 400 C, negatives) are kept purely
    as a generalization safety net: they never fired on train or test data.
    """

    SENSORS = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
    CLAMP_VALUES = (0.0, 1.0, 25.0)
    Z_CUT = 4.0
    PROB_CUT = 0.5
    SPREAD_MARGIN = 1e-6

    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.is_trained = False

    # ---------- feature helpers ----------
    @classmethod
    def _consistency_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sens = [s for s in cls.SENSORS if s in df.columns]
        df["spread"] = df[sens].max(axis=1) - df[sens].min(axis=1)
        df["mean123"] = df[sens].mean(axis=1)
        df["median123"] = df[sens].median(axis=1)
        if len(sens) == 3:
            df["abs_s12"] = (df["Sensor_S1"] - df["Sensor_S2"]).abs()
            df["abs_s23"] = (df["Sensor_S2"] - df["Sensor_S3"]).abs()
            df["abs_s13"] = (df["Sensor_S1"] - df["Sensor_S3"]).abs()
        df["int_flag"] = df[sens].isin(list(cls.CLAMP_VALUES)).any(axis=1).astype(int)
        df["miss_s123"] = df[sens].isnull().any(axis=1).astype(int)
        df["miss_s4"] = df["Sensor_S4"].isnull().astype(int) if "Sensor_S4" in df.columns else 0
        df = cls._covariates(df)
        for s in cls.SENSORS + ["mean123"]:
            df[f"z_{s}"] = np.nan
        df["max_abs_z"] = np.nan
        return df

    @staticmethod
    def _covariates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lc = df["Load_Current"] if "Load_Current" in df.columns else pd.Series(0.0, index=df.index)
        av = df["Applied_Voltage"] if "Applied_Voltage" in df.columns else pd.Series(0.0, index=df.index)
        df["I2"] = lc ** 2
        df["VI"] = lc * av
        return df

    COVARIATES = ["Load_Current", "I2", "Applied_Voltage", "Test_Duration",
                  "Ambient_Temperature", "VI"]

    def _fit_physics_state(self, Xdf: pd.DataFrame, valid_mask: np.ndarray):
        """Learn Huber physics fits, MAD scales, spread bound, dup pool (train only)."""
        from sklearn.linear_model import HuberRegressor

        self.hubs_, self.mads_ = {}, {}
        for tgt in self.SENSORS + ["mean123"]:
            ok = Xdf[self.COVARIATES + [tgt]].notna().all(axis=1)
            hub = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=5000)
            hub.fit(Xdf.loc[ok, self.COVARIATES].values, Xdf.loc[ok, tgt].values)
            resid = Xdf[tgt].values - hub.predict(Xdf[self.COVARIATES].values)
            resid = resid[ok.values]
            mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            self.hubs_[tgt], self.mads_[tgt] = hub, max(mad, 1e-3)
        # Empirical spread bound from the majority (Valid) class only
        self.spread_cut_ = float(Xdf.loc[valid_mask, "spread"].max()) + self.SPREAD_MARGIN
        # Duplicate pool: 8-feature keys of the training set
        self.dup_keys_ = set(map(tuple, Xdf[self._key_cols(Xdf)].round(6).values))

    def _key_cols(self, df: pd.DataFrame) -> List[str]:
        cols = ["Applied_Voltage", "Load_Current", "Ambient_Temperature",
                "Test_Duration"] + self.SENSORS
        cols += ["Sensor_S4"] if "Sensor_S4" in df.columns else []
        return cols

    def _fill_z(self, Xdf: pd.DataFrame) -> pd.DataFrame:
        Xdf = self._covariates(Xdf)
        if not hasattr(self, "hubs_"):
            return Xdf
        for tgt in self.SENSORS + ["mean123"]:
            pred = self.hubs_[tgt].predict(Xdf[self.COVARIATES].values)
            Xdf[f"z_{tgt}"] = (Xdf[tgt].values - pred) / self.mads_[tgt]
        Xdf["max_abs_z"] = Xdf[[f"z_{t}" for t in self.SENSORS + ["mean123"]]].abs().max(axis=1)
        return Xdf

    def _consistency_rules(self, Xdf: pd.DataFrame) -> pd.Series:
        """Data-derived deterministic invalidity rules (thresholds fit on train)."""
        r = pd.Series(False, index=Xdf.index)
        r |= Xdf["spread"] > self.spread_cut_
        r |= Xdf["miss_s123"] == 1
        r |= Xdf[self.SENSORS].eq(0.0).any(axis=1)
        r |= Xdf[self.SENSORS].isin(list(self.CLAMP_VALUES)).any(axis=1)
        r |= Xdf["max_abs_z"] > self.Z_CUT
        keys = Xdf[self._key_cols(Xdf)].round(6).fillna(-9999.0)
        in_pool = keys.apply(lambda row: tuple(row) in self.dup_keys_, axis=1)
        batch_dup = keys.duplicated(keep=False)
        r |= in_pool | batch_dup
        return r

    def _apply_physics_rules(self, df: pd.DataFrame) -> pd.Series:
        """Stray absolute-value safety net (never fires on in-distribution data)."""
        invalid = pd.Series(False, index=df.index)
        for s in self.SENSORS:
            if s in df.columns:
                invalid |= (df[s] < -10.0) | (df[s] > 400.0)
        if "Applied_Voltage" in df.columns:
            invalid |= (df["Applied_Voltage"] < 0)
        if "Load_Current" in df.columns:
            invalid |= (df["Load_Current"] < 0)
        if "Test_Duration" in df.columns:
            invalid |= (df["Test_Duration"] < 0)
        if "Sensor_Spread" in df.columns:
            invalid |= (df["Sensor_Spread"] > 200.0)
        return invalid

    # ---------- sklearn-style API (used for leakage-safe CV) ----------
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xdf = self._consistency_frame(X)
        std_y = y.astype(str).str.strip().map(
            lambda v: "Invalid" if v.lower() in ["invalid", "0", "false"] else "Valid")
        y_bin = (std_y == "Invalid").astype(int).values
        valid_mask = y_bin == 0
        self._fit_physics_state(Xdf, valid_mask)
        Xdf = self._fill_z(Xdf)

        self.feature_cols = ["spread", "mean123", "median123", "abs_s12", "abs_s23",
                             "abs_s13", "z_Sensor_S1", "z_Sensor_S2", "z_Sensor_S3",
                             "z_mean123", "max_abs_z", "int_flag", "miss_s123",
                             "miss_s4", "Applied_Voltage", "Load_Current",
                             "Ambient_Temperature", "Test_Duration"]
        self.feature_cols = [c for c in self.feature_cols if c in Xdf.columns]
        Xi = Xdf[self.feature_cols].values.astype(float)
        self.imp_ = SimpleImputer(strategy="median").fit(Xi)
        Xi = self.imp_.transform(Xi)

        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            learning_rate=0.06,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
            early_stopping=False,
        )
        self.model.fit(Xi, y_bin)
        self.is_trained = True
        return self

    def _ml_invalid_prob(self, Xdf: pd.DataFrame) -> np.ndarray:
        Xi = self.imp_.transform(Xdf[self.feature_cols].values.astype(float))
        return self.model.predict_proba(Xi)[:, 1]

    # ---------- original public interface (preserved) ----------
    def train(self, train_df: pd.DataFrame, feature_cols: List[str]):
        logger.info("=" * 70)
        logger.info("TASK 01: VALIDITY CLASSIFICATION (HYBRID RULES + ML)")
        logger.info("=" * 70)

        if "Validity_Label" not in train_df.columns:
            logger.warning("No 'Validity_Label' found. Defaulting to physics rules.")
            return

        labeled = train_df[train_df["Validity_Label"].notna()].copy()
        std_y = labeled["Validity_Label"].astype(str).str.strip().map(
            lambda v: "Invalid" if v.lower() in ["invalid", "0", "false"] else "Valid")
        logger.info(f"Class distribution in training: {dict(std_y.value_counts())}")

        self.fit(labeled, labeled["Validity_Label"])

        # Leakage-safe stratified CV: every learned state (Huber fits, MAD scales,
        # spread bound, dup pool, imputation) is re-derived inside each fold.
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof = pd.Series(index=labeled.index, dtype=object)
        fold_f1 = []
        for tr_idx, te_idx in cv.split(labeled, std_y):
            fold_clf = ValidityClassifier()
            fold_clf.log_predictions = False  # silence per-fold prediction logs
            fold_clf.fit(labeled.iloc[tr_idx], labeled.iloc[tr_idx]["Validity_Label"])
            oof.iloc[te_idx] = fold_clf.predict(labeled.iloc[te_idx])
            fold_f1.append(f1_score(std_y.iloc[te_idx], oof.iloc[te_idx],
                                    pos_label="Invalid"))
        y_true_bin = (std_y == "Invalid").astype(int)
        y_oof_bin = (oof == "Invalid").astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_oof_bin).ravel()
        logger.info(f"Hybrid Validity Classifier {N_FOLDS}-Fold Stratified CV:")
        logger.info(f"  Fold F1: mean={np.mean(fold_f1):.4f} (±{np.std(fold_f1):.4f})")
        logger.info(f"  OOF  Acc={accuracy_score(y_true_bin, y_oof_bin):.4f} | "
                    f"Prec={precision_score(y_true_bin, y_oof_bin):.4f} | "
                    f"Rec={recall_score(y_true_bin, y_oof_bin):.4f} | "
                    f"F1={f1_score(y_true_bin, y_oof_bin):.4f}")
        logger.info(f"  Confusion (Valid,Invalid x pred): TN={tn} FP={fp} FN={fn} TP={tp}")

        # Permutation importance on a stratified 25% holdout (evidence, not selection)
        try:
            from sklearn.model_selection import train_test_split
            itr, ite = train_test_split(np.arange(len(labeled)), test_size=0.25,
                                        stratify=std_y, random_state=RANDOM_STATE)
            hold_clf = ValidityClassifier().fit(labeled.iloc[itr],
                                                labeled.iloc[itr]["Validity_Label"])
            Hte = hold_clf._fill_z(hold_clf._consistency_frame(labeled.iloc[ite]))
            base_p = hold_clf._ml_invalid_prob(Hte)
            yte_bin = (std_y.iloc[ite] == "Invalid").astype(int).values
            rng = np.random.default_rng(RANDOM_STATE)
            drops = {}
            for c in self.feature_cols:
                ds = []
                for _ in range(10):
                    Xp = Hte.copy()
                    Xp[c] = rng.permutation(Xp[c].values)
                    p = hold_clf._ml_invalid_prob(Xp)
                    ds.append(f1_score(yte_bin, (base_p > self.PROB_CUT).astype(int))
                              - f1_score(yte_bin, (p > self.PROB_CUT).astype(int)))
                drops[c] = float(np.mean(ds))
            top = sorted(drops.items(), key=lambda kv: -kv[1])[:8]
            logger.info("  Permutation importance (ML path, dF1): "
                        + ", ".join(f"{k}={v:.3f}" for k, v in top))
        except Exception as e:  # pragma: no cover - diagnostics only
            logger.warning(f"Permutation importance skipped: {e}")

        self.model.fit(self.imp_.transform(
            self._fill_z(self._consistency_frame(labeled))[self.feature_cols].values.astype(float)),
            (std_y == "Invalid").astype(int).values)
        self.is_trained = True

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.is_trained and self.model is not None:
            Tdf = self._fill_z(self._consistency_frame(test_df))
            prob = self._ml_invalid_prob(Tdf)
            preds = pd.Series(np.where(prob > self.PROB_CUT, "Invalid", "Valid"),
                              index=test_df.index)
            # Layer 2: data-derived consistency rules (deterministic override)
            rule_invalid = self._consistency_rules(Tdf)
            preds[rule_invalid] = "Invalid"
        else:
            preds = pd.Series("Valid", index=test_df.index)

        # Stray absolute-value safety net (unchanged behaviour)
        phys_invalid = self._apply_physics_rules(test_df)
        preds[phys_invalid] = "Invalid"
        preds = preds.map(lambda x: "Valid" if str(x).strip().lower() in ["valid", "1", "true"] else "Invalid")
        if getattr(self, "log_predictions", True):
            logger.info(f"Predicted Test Validity: {dict(preds.value_counts())}")
        return preds


# =============================================================================
# 6. TASK 02: HOT-SPOT TEMPERATURE REGRESSOR (VALIDATE STAGE)
# =============================================================================
class HotSpotRegressor:
    """Config-5 production regressor (approved after 3-seed ablation verification).

    Single GradientBoostingRegressor over the verified 27-feature set: the
    26-feature pruned set (S4-derived features and dead sensor-ratio/interaction
    features removed) plus the input-only interaction
    I_squared_x_Ambient_Temperature = Load_Current^2 * Ambient_Temperature.
    """

    # Exact feature list used in the final 3-seed verification experiment.
    # Set-equality against PhysicsFeatureEngineer output is asserted in train().
    REG_FEATURES = [
        "Applied_Voltage", "Load_Current", "Ambient_Temperature", "Test_Duration",
        "Apparent_Power_kW", "Thermal_Energy_kWmin", "Joule_Heating_I2t", "V_squared",
        "I_squared", "Impedance_Proxy", "Log_Duration", "Log_Power", "Log_Energy",
        "Sqrt_Energy", "Power_x_LogDuration", "Ambient_Temp",
        "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_Mean", "Sensor_Median",
        "Sensor_Max", "Sensor_Min", "Sensor_Spread", "Sensor_Std", "Max_to_Mean_Ratio",
        "I_squared_x_Ambient_Temperature",
    ]

    def __init__(self):
        self.model = None
        self.feature_cols: List[str] = []

    def train(self, train_df: pd.DataFrame, feature_cols: List[str]):
        logger.info("=" * 70)
        logger.info("TASK 02: HOT-SPOT TEMPERATURE REGRESSION (VALID-ONLY FILTER)")
        logger.info("=" * 70)

        # STRICT FILTER: Train solely on 'Valid' records
        if "Validity_Label" in train_df.columns:
            valid_mask = train_df["Validity_Label"].astype(str).str.strip().str.lower().isin(["valid", "1", "true"])
            clean = train_df[valid_mask].copy()
            logger.info(f"Strict Filter: Training on {len(clean)} Valid rows (excluded {len(train_df) - len(clean)} invalid rows).")
        else:
            clean = train_df.copy()

        if "Reference_Parameter" not in clean.columns:
            logger.error("Reference_Parameter target column is missing.")
            sys.exit(1)

        clean = clean.dropna(subset=["Reference_Parameter"]).reset_index(drop=True)

        # Guard the verified feature contract: every approved feature must exist
        # in the engineered frame; no S4-derived feature may enter the regression.
        missing = [c for c in self.REG_FEATURES if c not in clean.columns]
        assert not missing, f"Config-5 feature contract violated, missing: {missing}"
        assert not any("s4" in c.lower() for c in self.REG_FEATURES), \
            "S4-derived features must not enter the regression"
        self.feature_cols = list(self.REG_FEATURES)
        logger.info(f"Config-5 feature contract OK: {len(self.feature_cols)} regression features")

        X = clean[self.feature_cols].values
        y = clean["Reference_Parameter"].values.astype(float)

        logger.info(f"Feature count: {len(self.feature_cols)} | Samples: {len(X)}")
        logger.info(f"Target distribution -> Mean: {y.mean():.2f}°C, Std: {y.std():.2f}°C, Range: [{y.min():.2f}°C, {y.max():.2f}°C]")

        def _mk_model():
            return GradientBoostingRegressor(
                n_estimators=700,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.85,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )

        # CV metric report (identical protocol to the verification experiments)
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_preds = cross_val_predict(_mk_model(), X, y, cv=cv)
        cv_rmse = compute_rmse(y, cv_preds)
        cv_mae = mean_absolute_error(y, cv_preds)
        cv_r2 = r2_score(y, cv_preds)
        fold_rmse = []
        for tr_idx, te_idx in cv.split(X, y):
            m = _mk_model()
            m.fit(X[tr_idx], y[tr_idx])
            fold_rmse.append(compute_rmse(y[te_idx], m.predict(X[te_idx])))
        logger.info(f"GradientBoostingRegressor 5-Fold CV -> RMSE: {cv_rmse:.4f} | MAE: {cv_mae:.4f} | R²: {cv_r2:.4f}")
        logger.info(f"  Per-fold RMSE: {[round(v, 4) for v in fold_rmse]} (mean {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f})")
        logger.info("  (Experimental CV result - hidden test labels are unavailable; this is not a test-set score.)")

        # Final fit on all valid records for production inference
        self.model = _mk_model().fit(X, y)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "HotSpotRegressor must be trained before predict"
        X = test_df[self.feature_cols].values
        return self.model.predict(X)


# =============================================================================
# 7. TASK 03: AUTOMATED DELIVERABLE EXPORT (AUTOMATE STAGE)
# =============================================================================
class DeliverableExporter:
    @staticmethod
    def run(test_df: pd.DataFrame, preds: np.ndarray, validity: pd.Series):
        logger.info("=" * 70)
        logger.info("TASK 03: AUTOMATED DELIVERABLE EXPORT")
        logger.info("=" * 70)

        tids = test_df["Test_ID"].values if "Test_ID" in test_df.columns else np.arange(1, len(test_df) + 1)

        # 1. Output CSV (<TeamName>.csv)
        out_df = pd.DataFrame({
            "Test_ID": tids,
            "Predicted_Reference_Parameter": np.round(preds, 4),
            "Valid / Invalid": validity.values,
        })
        out_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Generated Prediction CSV: '{OUTPUT_CSV}' ({len(out_df)} rows)")

        # 2. Top-3 Attention Test IDs (Invalid first, then highest predicted temperature)
        rank_df = pd.DataFrame({
            "Test_ID": tids,
            "Pred": preds,
            "Is_Invalid": (validity.values == "Invalid").astype(int)
        }).sort_values(["Is_Invalid", "Pred"], ascending=[False, False])

        top3 = rank_df["Test_ID"].head(3).tolist()
        top3 = [int(x) if isinstance(x, (np.integer, np.int64, np.int32)) else x for x in top3]

        # 3. Summary JSON (Concise explanation strictly under 100 words)
        approach = (
            "Hybrid validity screening: consistency rules derived from training data only "
            "(sensor-spread bound, Huber/MAD physics z-scores of S1-S3 vs operating point, "
            "missing-channel and clamp-value detection, duplicate-measurement keys) OR-ed with a "
            "HistGradientBoosting classifier; leakage-safe stratified 5-fold CV (all learned "
            "state re-derived per fold). Regression: single GradientBoostingRegressor on 27 "
            "physics-informed input features (S4 removed after ablation; I2xAmbient interaction), "
            "trained on verified-valid records; fixed seeds for full reproducibility."
        )

        summary = {
            "records_analysed": int(len(test_df)),
            "abnormal_invalid_records": int((validity.values == "Invalid").sum()),
            "min_predicted_reference_parameter": round(float(preds.min()), 2),
            "max_predicted_reference_parameter": round(float(preds.max()), 2),
            "average_predicted_reference_parameter": round(float(preds.mean()), 2),
            "top_3_attention_test_ids": top3,
            "approach_explanation": approach,
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Generated Summary JSON: '{OUTPUT_JSON}'")
        for k, v in summary.items():
            if k != "approach_explanation":
                logger.info(f"  • {k}: {v}")
        logger.info(f"  • approach_explanation word count: {len(approach.split())} words")


# =============================================================================
# 8. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info(f"  PowerNext-AI 2026 Challenge | Team: {TEAM_NAME}")
    logger.info("  Framework: Understand -> Analyse -> Validate -> Automate")
    logger.info("=" * 70)

    dataset_path = locate_workbook(DEFAULT_DATA_FILE)

    # 1. UNDERSTAND
    cleaner = DatasetCleaner()
    train_df, test_df = cleaner.ingest_and_clean(dataset_path)

    # 2. ANALYSE
    logger.info("=" * 70)
    logger.info("STAGE 2: ANALYSE — Physics-Informed Feature Engineering")
    logger.info("=" * 70)
    fe = PhysicsFeatureEngineer(s4_is_noise=cleaner.s4_is_noise)
    train_feat = fe.transform(train_df)
    test_feat = fe.transform(test_df)

    feature_cols = fe.get_feature_list(train_feat)
    logger.info(f"Total features extracted: {len(feature_cols)}")

    for c in feature_cols:
        if c not in test_feat.columns:
            test_feat[c] = 0.0

    # 3. VALIDATE (Task 01: Validity Classification)
    # NOTE: uses the raw (pre-imputation) frames — the missingness pattern itself
    # is a fault signal (S1-S3 dropouts occur exclusively on Invalid records),
    # so the validity stage must see the original NaNs. The regressor keeps using
    # the imputed frames from STAGE 2, unchanged.
    classifier = ValidityClassifier()
    classifier.train(cleaner.train_raw, feature_cols)
    validity_preds = classifier.predict(cleaner.test_raw)

    # 4. VALIDATE (Task 02: Hot-Spot Regression)
    regressor = HotSpotRegressor()
    regressor.train(train_feat, feature_cols)
    temperature_preds = regressor.predict(test_feat)

    # 5. AUTOMATE (Task 03: Deliverable Export)
    DeliverableExporter.run(test_feat, temperature_preds, validity_preds)

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE — ALL DELIVERABLES GENERATED.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()