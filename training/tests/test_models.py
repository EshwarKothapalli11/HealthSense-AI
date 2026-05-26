"""
test_models.py — Unit tests for model architectures and output shapes.
"""

import numpy as np
import pytest


def test_diabetes_model_output_shape():
    """Test that diabetes model produces correct output shape."""
    from src.models.diabetes_model import build_diabetes_model
    model = build_diabetes_model(input_dim=8)
    dummy = np.random.rand(4, 8).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_heart_model_output_shape():
    """Test that heart model produces correct output shape."""
    from src.models.heart_model import build_heart_model
    model = build_heart_model(input_dim=13)
    dummy = np.random.rand(4, 13).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_mental_model_output_shape():
    """Test that mental health model produces correct output shape."""
    from src.models.mental_model import build_mental_model
    model = build_mental_model(vocab_size=10000, maxlen=200)
    dummy = np.random.randint(0, 10000, (4, 200))
    out = model.predict(dummy, verbose=0)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_diabetes_model_output_range():
    """Test that diabetes model output is between 0 and 1 (sigmoid)."""
    from src.models.diabetes_model import build_diabetes_model
    model = build_diabetes_model(input_dim=8)
    dummy = np.random.rand(10, 8).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"


def test_heart_model_output_range():
    """Test that heart model output is between 0 and 1 (sigmoid)."""
    from src.models.heart_model import build_heart_model
    model = build_heart_model(input_dim=13)
    dummy = np.random.rand(10, 13).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"


def test_mental_model_output_range():
    """Test that mental model output is between 0 and 1 (sigmoid)."""
    from src.models.mental_model import build_mental_model
    model = build_mental_model(vocab_size=10000, maxlen=200)
    dummy = np.random.randint(0, 10000, (10, 200))
    out = model.predict(dummy, verbose=0)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"
