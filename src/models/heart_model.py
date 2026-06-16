"""
heart_model.py — XGBoost model for Heart Disease prediction.

Re-exports from training/models/heart_model.py for import compatibility.
"""

import os
import sys

# Ensure training/models is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from training.models.heart_model import build_heart_model, train_heart_model

__all__ = ['build_heart_model', 'train_heart_model']
