import sys
import os
from unittest.mock import MagicMock
import numpy as np

class MockKerasModel:
    def predict(self, *args, **kwargs):
        return np.array([[0.5]])

tf_mock = MagicMock()
tf_mock.keras.models.load_model.return_value = MockKerasModel()
tf_mock.keras.Model = MockKerasModel

keras_mock = MagicMock()
sys.modules['keras'] = keras_mock
sys.modules['tensorflow'] = tf_mock
sys.modules['tensorflow.keras'] = tf_mock.keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

captured_fn = None
class MockButton:
    def click(self, fn, *args, **kwargs):
        global captured_fn
        if fn.__name__ == 'generate_full_report':
            captured_fn = fn
        return self
    def then(self, *args, **kwargs):
        return self

class MockGradio:
    def Button(self, *args, **kwargs):
        return MockButton()
    def __getattr__(self, name):
        return MagicMock()

sys.modules['gradio'] = MockGradio()

import config
import pickle
with open(os.path.join(config.MODELS_DIR, 'diabetes_best.pkl'), 'rb') as f:
    d_model = pickle.load(f)
with open(os.path.join(config.MODELS_DIR, 'heart_best.pkl'), 'rb') as f:
    h_model = pickle.load(f)
with open(os.path.join(config.SCALERS_DIR, 'diabetes_scaler.pkl'), 'rb') as f:
    d_scaler = pickle.load(f)
with open(os.path.join(config.SCALERS_DIR, 'heart_scaler.pkl'), 'rb') as f:
    h_scaler = pickle.load(f)

class MockTokenizer:
    def texts_to_sequences(self, *args):
        return [[1]]
tok = MockTokenizer()

from src.fusion.fusion_engine import HealthFusionEngine
engine = HealthFusionEngine()

from src.ui.app import create_app
demo = create_app(d_model, h_model, MockKerasModel(), d_scaler, h_scaler, tok, engine)

print("Running generate_full_report...")
if captured_fn:
    # 26 arguments based on inputs_list
    res = captured_fn(
        1, 120, 80, 20, 80, 25.0, 0.5, 30, # d 8
        50, "Male", "Typical Angina", 120, 200, False, "Normal", 150, False, 0.0, "Flat", 0, "Normal", # h 13
        "I feel sad today", # mental
        33, 33, 33, 0.5 # fusion
    )
    print("Result generated successfully!")
else:
    print("Could not capture generate_full_report")
