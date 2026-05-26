"""
app.py — Main entry point for Hugging Face Spaces.
"""

import os
import sys
import pickle
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from src.ui.app import create_app
from src.fusion.fusion_engine import HealthFusionEngine

# Check and generate output directories if they miss
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.SCALERS_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)

# Paths
diabetes_model_path = os.path.join(config.MODELS_DIR, 'diabetes_best.h5')
heart_model_path = os.path.join(config.MODELS_DIR, 'heart_best.h5')
mental_model_path = os.path.join(config.MODELS_DIR, 'mental_best.h5')

diabetes_scaler_path = os.path.join(config.SCALERS_DIR, 'diabetes_scaler.pkl')
heart_scaler_path = os.path.join(config.SCALERS_DIR, 'heart_scaler.pkl')
tokenizer_path = os.path.join(config.SCALERS_DIR, 'mental_tokenizer.pkl')

# Check if models exist
missing_files = []
for path in [diabetes_model_path, heart_model_path, mental_model_path, 
             diabetes_scaler_path, heart_scaler_path, tokenizer_path]:
    if not os.path.exists(path):
        missing_files.append(path)
        
if missing_files:
    print("\n[ERROR] Missing required model or scaler files:")
    for f in missing_files:
        print(f"  - {f}")
    sys.exit(1)

try:
    print("[INFO] Loading models...")
    diabetes_model = tf.keras.models.load_model(diabetes_model_path, compile=False)
    heart_model = tf.keras.models.load_model(heart_model_path, compile=False)
    mental_model = tf.keras.models.load_model(mental_model_path, compile=False)
    
    print("[INFO] Loading preprocessors...")
    with open(diabetes_scaler_path, 'rb') as f:
        diabetes_scaler = pickle.load(f)
    with open(heart_scaler_path, 'rb') as f:
        heart_scaler = pickle.load(f)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
except Exception as e:
    print(f"\n[ERROR] Failed to load models or scalers: {e}")
    sys.exit(1)
    
print("[INFO] Initializing Multi-Modal Fusion Engine...")
fusion_engine = HealthFusionEngine()

print("[INFO] Building Web UI...")
demo = create_app(
    diabetes_model=diabetes_model,
    heart_model=heart_model,
    mental_model=mental_model,
    diabetes_scaler=diabetes_scaler,
    heart_scaler=heart_scaler,
    tokenizer=tokenizer,
    fusion_engine=fusion_engine
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 READY! Launching web server...")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7860)
