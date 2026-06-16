import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.ui.app import create_app
from src.fusion.fusion_engine import HealthFusionEngine

# Load models and scalers
with open(os.path.join(config.MODELS_DIR, 'diabetes_best.pkl'), 'rb') as f:
    d_model = pickle.load(f)
with open(os.path.join(config.MODELS_DIR, 'heart_best.pkl'), 'rb') as f:
    h_model = pickle.load(f)
with open(os.path.join(config.SCALERS_DIR, 'diabetes_scaler.pkl'), 'rb') as f:
    d_scaler = pickle.load(f)
with open(os.path.join(config.SCALERS_DIR, 'heart_scaler.pkl'), 'rb') as f:
    h_scaler = pickle.load(f)

# Need mental model for create_app
import tensorflow as tf
class DummyModel:
    def predict(self, X, **kwargs):
        return np.array([[0.1]])
m_model = DummyModel()

# Fake tokenizer
class DummyTokenizer:
    pass

engine = HealthFusionEngine()
app = create_app(d_model, h_model, m_model, d_scaler, h_scaler, DummyTokenizer(), engine)

# To test predict_physical, we can import it from the current context? No, it's nested inside create_app.
# Let's just redefine _prepare_heart_input and _predict_prob exactly as they are in app.py

def _predict_prob(model, X):
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0, 1])
    else:
        return float(model.predict(X, verbose=0)[0][0])

def _prepare_heart_input(age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak, slope):
    raw_dict = {
        'age': float(age),
        'sex': 1.0 if sex == "Male" else 0.0,
        'cp': float({"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal": 3, "Asymptomatic": 4}.get(cp, 1)),
        'trestbps': float(trestbps),
        'chol': float(chol),
        'fbs': 1.0 if fbs else 0.0,
        'restecg': 0.0,
        'thalach': float(thalach),
        'exang': 1.0 if exang else 0.0,
        'oldpeak': float(oldpeak),
        'slope': float({"Up": 1, "Flat": 2, "Down": 3}.get(slope, 1)),
        'ca': 0.0,
        'thal': 3.0
    }
    feature_names = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1', 'cp_2', 'cp_3', 'cp_4', 'fbs_1', 'restecg_1', 'restecg_2', 'exang_1', 'slope_2', 'slope_3', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0']
    input_array = np.zeros(len(feature_names), dtype=np.float32)
    for i, col in enumerate(feature_names):
        if col in raw_dict:
            input_array[i] = raw_dict[col]
        elif '_' in col:
            try:
                base_feat, val = col.rsplit('_', 1)
                if base_feat in raw_dict and raw_dict[base_feat] == float(val):
                    input_array[i] = 1.0
            except ValueError:
                pass
    return h_scaler.transform(input_array.reshape(1, -1))

# Test Cases
print("--- Heart Disease Tests ---")
# Very low risk: young female, typical angina (0 from UI), low BP, low chol, high thalach, no exang, 0 oldpeak, Up slope
# Note: from UI, cp is an integer. Let's pass what UI passes.
h_low = _prepare_heart_input(35, "Female", 0, 110, 150, False, 180, False, 0.0, "Up")
print(f"Very Low Risk: {_predict_prob(h_model, h_low):.4f}")

# Moderate risk: middle age male, atypical angina (1 from UI), mod BP, mod chol
h_mod = _prepare_heart_input(55, "Male", 1, 130, 220, False, 150, False, 1.0, "Flat")
print(f"Moderate Risk: {_predict_prob(h_model, h_mod):.4f}")

# Very high risk: old male, asymptomatic (3 from UI), high BP, high chol, exang=True, oldpeak=3.0, Down slope
h_high = _prepare_heart_input(65, "Male", 3, 160, 280, True, 100, True, 3.0, "Down")
print(f"Very High Risk: {_predict_prob(h_model, h_high):.4f}")

