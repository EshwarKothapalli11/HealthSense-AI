"""
validate_pipeline.py -- End-to-end ML pipeline validation for HealthSense AI.

Checks:
  1. Feature order match (training vs UI hardcoded)
  2. Scaler dimensionality match (scaler.n_features_in_ vs model expectation)
  3. OHE encoding correctness (one-hot matching logic)
  4. Model loading (correct file type: .pkl for XGBoost, .h5 for LSTM)
  5. predict_proba / predict logic (XGBoost vs Keras dispatch)
  6. High-risk and low-risk test case validation
"""

import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f"  -- {detail}" if detail else ""))
    return condition


# =========================================================
#  1. Feature order: training vs UI
# =========================================================
print("\n=== 1. FEATURE ORDER ===")

# Diabetes
df_d = pd.read_csv("data/raw/diabetes.csv")
training_d_features = [c for c in df_d.columns if c != "Outcome"]
ui_d_features = config.DIABETES_COLUMNS[:-1]
check("Diabetes feature order",
      training_d_features == ui_d_features,
      f"training={training_d_features} vs ui={ui_d_features}")

# Heart
df_h = pd.read_csv("data/raw/heart.csv")
cats = []
for c in df_h.columns:
    if c == "target":
        continue
    if df_h[c].nunique() <= 5 and c not in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        cats.append(c)
df_h_enc = pd.get_dummies(df_h, columns=cats, drop_first=True)
df_h_enc = df_h_enc.apply(pd.to_numeric, errors="coerce").dropna()
training_h_features = [c for c in df_h_enc.columns if c != "target"]

ui_h_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex_1", "cp_2", "cp_3", "cp_4", "fbs_1",
    "restecg_1", "restecg_2", "exang_1",
    "slope_2", "slope_3",
    "ca_1.0", "ca_2.0", "ca_3.0",
    "thal_6.0", "thal_7.0",
]
check("Heart feature order",
      training_h_features == ui_h_features,
      f"count: training={len(training_h_features)} vs ui={len(ui_h_features)}")

# =========================================================
#  2. Scaler dimensionality
# =========================================================
print("\n=== 2. SCALER DIMENSIONALITY ===")

with open(os.path.join(config.SCALERS_DIR, "diabetes_scaler.pkl"), "rb") as f:
    d_scaler = pickle.load(f)
check("Diabetes scaler n_features",
      d_scaler.n_features_in_ == 8,
      f"n_features_in_={d_scaler.n_features_in_}, expected=8")

with open(os.path.join(config.SCALERS_DIR, "heart_scaler.pkl"), "rb") as f:
    h_scaler = pickle.load(f)
check("Heart scaler n_features",
      h_scaler.n_features_in_ == 20,
      f"n_features_in_={h_scaler.n_features_in_}, expected=20")

# =========================================================
#  3. OHE encoding correctness
# =========================================================
print("\n=== 3. ONE-HOT ENCODING LOGIC ===")


def build_heart_vector(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """Reproduce the UI's _prepare_heart_input logic."""
    raw_dict = {
        "age": float(age), "sex": float(sex), "cp": float(cp),
        "trestbps": float(trestbps), "chol": float(chol),
        "fbs": float(fbs), "restecg": float(restecg),
        "thalach": float(thalach), "exang": float(exang),
        "oldpeak": float(oldpeak), "slope": float(slope),
        "ca": float(ca), "thal": float(thal),
    }
    arr = np.zeros(20, dtype=np.float32)
    for i, col in enumerate(ui_h_features):
        if col in raw_dict:
            arr[i] = raw_dict[col]
        elif "_" in col:
            try:
                base, val = col.rsplit("_", 1)
                if base in raw_dict and raw_dict[base] == float(val):
                    arr[i] = 1.0
            except ValueError:
                pass
    return arr


# Male, cp=3 (Non-Anginal), slope=2 (Flat)
vec = build_heart_vector(50, 1, 3, 120, 200, 0, 0, 150, 0, 1.0, 2, 0, 3)
check("OHE: sex_1 activated for Male", vec[5] == 1.0)
check("OHE: cp_3 activated for cp=3", vec[7] == 1.0)
check("OHE: cp_2 NOT activated for cp=3", vec[6] == 0.0)
check("OHE: cp_4 NOT activated for cp=3", vec[8] == 0.0)
check("OHE: slope_2 activated for slope=2", vec[13] == 1.0)
check("OHE: slope_3 NOT activated for slope=2", vec[14] == 0.0)

# Female, cp=4 (Asymptomatic), slope=3 (Down)
vec2 = build_heart_vector(60, 0, 4, 140, 300, 1, 0, 120, 1, 3.0, 3, 2.0, 7.0)
check("OHE: sex_1 NOT activated for Female", vec2[5] == 0.0)
check("OHE: cp_4 activated for cp=4", vec2[8] == 1.0)
check("OHE: fbs_1 activated for fbs=1", vec2[9] == 1.0)
check("OHE: exang_1 activated for exang=1", vec2[12] == 1.0)
check("OHE: slope_3 activated for slope=3", vec2[14] == 1.0)
check("OHE: ca_2.0 activated for ca=2", vec2[16] == 1.0)
check("OHE: thal_7.0 activated for thal=7", vec2[19] == 1.0)

# =========================================================
#  4. Model loading (correct types)
# =========================================================
print("\n=== 4. MODEL LOADING ===")

d_model_path = os.path.join(config.MODELS_DIR, "diabetes_best.pkl")
h_model_path = os.path.join(config.MODELS_DIR, "heart_best.pkl")
m_model_path = os.path.join(config.MODELS_DIR, "mental_best.h5")

check("Diabetes model file exists (.pkl)", os.path.exists(d_model_path))
check("Heart model file exists (.pkl)", os.path.exists(h_model_path))
check("Mental model file exists (.h5)", os.path.exists(m_model_path))

with open(d_model_path, "rb") as f:
    d_model = pickle.load(f)
check("Diabetes model has predict_proba", hasattr(d_model, "predict_proba"))
check("Diabetes model has feature_importances_", hasattr(d_model, "feature_importances_"))

with open(h_model_path, "rb") as f:
    h_model = pickle.load(f)
check("Heart model has predict_proba", hasattr(h_model, "predict_proba"))
check("Heart model has feature_importances_", hasattr(h_model, "feature_importances_"))

check("Diabetes feature_importances_ count = 8",
      len(d_model.feature_importances_) == 8,
      f"got {len(d_model.feature_importances_)}")
check("Heart feature_importances_ count = 20",
      len(h_model.feature_importances_) == 20,
      f"got {len(h_model.feature_importances_)}")

import tensorflow as tf
m_model = tf.keras.models.load_model(m_model_path, compile=False)
check("Mental model is Keras Model", isinstance(m_model, tf.keras.Model))
check("Mental model NOT has predict_proba", not hasattr(m_model, "predict_proba"))

# =========================================================
#  5. predict_proba / predict dispatch
# =========================================================
print("\n=== 5. PROBABILITY DISPATCH ===")


def _predict_prob(model, X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    else:
        return float(model.predict(X, verbose=0)[0][0])


# Quick smoke test with default values
d_test = d_scaler.transform(np.array([[1, 100, 70, 20, 80, 25.0, 0.5, 30]], dtype=np.float32))
d_prob = _predict_prob(d_model, d_test)
check("Diabetes predict_proba returns float in [0,1]",
      isinstance(d_prob, float) and 0.0 <= d_prob <= 1.0,
      f"got {d_prob:.4f}")

h_test = h_scaler.transform(build_heart_vector(50, 1, 3, 120, 200, 0, 0, 150, 0, 1.0, 2, 0, 3).reshape(1, -1))
h_prob = _predict_prob(h_model, h_test)
check("Heart predict_proba returns float in [0,1]",
      isinstance(h_prob, float) and 0.0 <= h_prob <= 1.0,
      f"got {h_prob:.4f}")

from src.data.preprocess_text import clean_text, texts_to_padded
tokenizer_path = os.path.join(config.SCALERS_DIR, "mental_tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
m_text = "I feel very depressed and hopeless"
m_cleaned = clean_text(m_text)
m_padded = texts_to_padded([m_cleaned], tokenizer, maxlen=config.MAX_TEXT_LEN)
m_prob = _predict_prob(m_model, m_padded)
check("Mental predict returns float in [0,1]",
      isinstance(m_prob, float) and 0.0 <= m_prob <= 1.0,
      f"got {m_prob:.4f}")

# =========================================================
#  6. HIGH-RISK vs LOW-RISK test cases
# =========================================================
print("\n=== 6. HIGH-RISK vs LOW-RISK VALIDATION ===")

# --- Diabetes ---
# High-risk: high glucose, high BMI, high age, high insulin, multiple pregnancies
high_d = np.array([[8, 180, 90, 45, 500, 42.0, 1.8, 55]], dtype=np.float32)
high_d_scaled = d_scaler.transform(high_d)
high_d_prob = _predict_prob(d_model, high_d_scaled)

# Low-risk: young, low glucose, normal BMI
low_d = np.array([[0, 85, 65, 15, 50, 22.0, 0.2, 25]], dtype=np.float32)
low_d_scaled = d_scaler.transform(low_d)
low_d_prob = _predict_prob(d_model, low_d_scaled)

check("Diabetes: high-risk > low-risk",
      high_d_prob > low_d_prob,
      f"high={high_d_prob:.4f}, low={low_d_prob:.4f}")
check("Diabetes: high-risk prob > 0.5",
      high_d_prob > 0.5,
      f"got {high_d_prob:.4f}")
check("Diabetes: low-risk prob < 0.5",
      low_d_prob < 0.5,
      f"got {low_d_prob:.4f}")

# --- Heart ---
# High-risk: old male, asymptomatic cp=4, high chol, low thalach, exercise angina, high oldpeak
high_h = build_heart_vector(65, 1, 4, 160, 350, 1, 2, 100, 1, 4.0, 3, 3.0, 7.0)
high_h_scaled = h_scaler.transform(high_h.reshape(1, -1))
high_h_prob = _predict_prob(h_model, high_h_scaled)

# Low-risk: young female, non-anginal cp=1 (Typical), low chol, high thalach, no angina
low_h = build_heart_vector(35, 0, 1, 110, 180, 0, 0, 180, 0, 0.0, 1, 0.0, 3.0)
low_h_scaled = h_scaler.transform(low_h.reshape(1, -1))
low_h_prob = _predict_prob(h_model, low_h_scaled)

check("Heart: high-risk > low-risk",
      high_h_prob > low_h_prob,
      f"high={high_h_prob:.4f}, low={low_h_prob:.4f}")
check("Heart: high-risk prob > 0.5",
      high_h_prob > 0.5,
      f"got {high_h_prob:.4f}")
check("Heart: low-risk prob < 0.5",
      low_h_prob < 0.5,
      f"got {low_h_prob:.4f}")

# --- Mental ---
depressed_text = "I feel so depressed and hopeless. I cannot sleep at night and I have lost interest in everything. Life feels meaningless and I cry every day."
happy_text = "I had a wonderful day today. I feel happy and grateful for my family and friends. Life is beautiful and I am looking forward to tomorrow."

dep_padded = texts_to_padded([clean_text(depressed_text)], tokenizer, maxlen=config.MAX_TEXT_LEN)
dep_prob = _predict_prob(m_model, dep_padded)
hap_padded = texts_to_padded([clean_text(happy_text)], tokenizer, maxlen=config.MAX_TEXT_LEN)
hap_prob = _predict_prob(m_model, hap_padded)

check("Mental: depressed > happy",
      dep_prob > hap_prob,
      f"depressed={dep_prob:.4f}, happy={hap_prob:.4f}")

# =========================================================
#  7. Summary
# =========================================================
print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
warned = sum(1 for s, _, _ in results if s == WARN)
print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {warned} warnings")
if failed == 0:
    print("  ALL CHECKS PASSED!")
else:
    print("\n  FAILURES:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"    {FAIL} {name}  -- {detail}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
