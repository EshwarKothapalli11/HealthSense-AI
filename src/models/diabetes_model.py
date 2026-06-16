"""
diabetes_model.py — XGBoost model for Pima Indians Diabetes prediction.

Re-exports from training/models/diabetes_model.py for import compatibility.
"""

import os
import sys

# Ensure training/models is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from training.models.diabetes_model import build_diabetes_model, train_diabetes_model

__all__ = ['build_diabetes_model', 'train_diabetes_model']
