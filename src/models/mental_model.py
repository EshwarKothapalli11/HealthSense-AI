"""
mental_model.py — LSTM model for Mental Health text classification.

Re-exports from training/models/mental_model.py for import compatibility.
"""

import os
import sys

# Ensure training/models is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from training.models.mental_model import build_mental_model, train_mental_model

__all__ = ['build_mental_model', 'train_mental_model']
