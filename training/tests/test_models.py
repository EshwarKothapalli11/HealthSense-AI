"""
test_models.py — Unit tests for model architectures and output shapes.
"""

import numpy as np
import pytest


def test_diabetes_model_output_shape():
    """Test that diabetes XGBoost model produces correct output shape."""
    from src.models.diabetes_model import build_diabetes_model
    model = build_diabetes_model(input_dim=8)
    # Fit on minimal data so predict works
    X_dummy = np.random.rand(10, 8).astype(np.float32)
    y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model.fit(X_dummy, y_dummy)
    X_test = np.random.rand(4, 8).astype(np.float32)
    proba = model.predict_proba(X_test)
    assert proba.shape == (4, 2), f"Expected (4, 2), got {proba.shape}"
    pred = model.predict(X_test)
    assert pred.shape == (4,), f"Expected (4,), got {pred.shape}"


def test_heart_model_output_shape():
    """Test that heart XGBoost model produces correct output shape."""
    from src.models.heart_model import build_heart_model
    model = build_heart_model(input_dim=13)
    # Fit on minimal data so predict works
    X_dummy = np.random.rand(10, 13).astype(np.float32)
    y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model.fit(X_dummy, y_dummy)
    X_test = np.random.rand(4, 13).astype(np.float32)
    proba = model.predict_proba(X_test)
    assert proba.shape == (4, 2), f"Expected (4, 2), got {proba.shape}"
    pred = model.predict(X_test)
    assert pred.shape == (4,), f"Expected (4,), got {pred.shape}"


def test_mental_model_output_shape():
    """Test that mental health model produces correct output shape."""
    from src.models.mental_model import build_mental_model
    model = build_mental_model(vocab_size=10000, maxlen=200)
    dummy = np.random.randint(0, 10000, (4, 200))
    out = model.predict(dummy, verbose=0)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_diabetes_model_output_range():
    """Test that diabetes XGBoost predict_proba is between 0 and 1."""
    from src.models.diabetes_model import build_diabetes_model
    model = build_diabetes_model(input_dim=8)
    X_dummy = np.random.rand(20, 8).astype(np.float32)
    y_dummy = np.array([0, 1] * 10)
    model.fit(X_dummy, y_dummy)
    X_test = np.random.rand(10, 8).astype(np.float32)
    proba = model.predict_proba(X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all(), "Output should be in [0, 1]"


def test_heart_model_output_range():
    """Test that heart XGBoost predict_proba is between 0 and 1."""
    from src.models.heart_model import build_heart_model
    model = build_heart_model(input_dim=13)
    X_dummy = np.random.rand(20, 13).astype(np.float32)
    y_dummy = np.array([0, 1] * 10)
    model.fit(X_dummy, y_dummy)
    X_test = np.random.rand(10, 13).astype(np.float32)
    proba = model.predict_proba(X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all(), "Output should be in [0, 1]"


def test_mental_model_output_range():
    """Test that mental model output is between 0 and 1 (sigmoid)."""
    from src.models.mental_model import build_mental_model
    model = build_mental_model(vocab_size=10000, maxlen=200)
    dummy = np.random.randint(0, 10000, (10, 200))
    out = model.predict(dummy, verbose=0)
    assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0, 1]"
