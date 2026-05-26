"""
test_preprocessing.py — Unit tests for data preprocessing functions.
"""

import numpy as np
import pandas as pd
import pytest


def test_impute_zeros():
    """Test that zeros are replaced with column medians."""
    from src.data.preprocess_tabular import impute_zeros
    df = pd.DataFrame({'Glucose': [0, 120, 0], 'BMI': [0, 25.0, 30.0]})
    result = impute_zeros(df, ['Glucose', 'BMI'])
    assert (result['Glucose'] != 0).all(), "Zeros should be replaced in Glucose"


def test_impute_preserves_nonzero():
    """Test that non-zero values are not changed by imputation."""
    from src.data.preprocess_tabular import impute_zeros
    df = pd.DataFrame({'Glucose': [100, 120, 80], 'BMI': [22.0, 25.0, 30.0]})
    result = impute_zeros(df, ['Glucose', 'BMI'])
    pd.testing.assert_frame_equal(result, df)


def test_clean_text():
    """Test text cleaning removes URLs and special characters."""
    from src.data.preprocess_text import clean_text
    assert clean_text("Hello WORLD!! https://example.com") == "hello world"


def test_clean_text_empty():
    """Test text cleaning handles empty strings."""
    from src.data.preprocess_text import clean_text
    assert clean_text("") == ""


def test_clean_text_non_string():
    """Test text cleaning handles non-string input."""
    from src.data.preprocess_text import clean_text
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_padding_shape():
    """Test that padded sequences have correct shape."""
    from src.data.preprocess_text import build_tokenizer, texts_to_padded
    tokenizer = build_tokenizer(["hello world", "test sentence"], vocab_size=100)
    padded = texts_to_padded(["hello"], tokenizer, maxlen=10)
    assert padded.shape == (1, 10), f"Expected (1, 10), got {padded.shape}"


def test_padding_multiple_texts():
    """Test padding with multiple texts."""
    from src.data.preprocess_text import build_tokenizer, texts_to_padded
    texts = ["hello world", "test sentence", "another example"]
    tokenizer = build_tokenizer(texts, vocab_size=100)
    padded = texts_to_padded(texts, tokenizer, maxlen=10)
    assert padded.shape == (3, 10), f"Expected (3, 10), got {padded.shape}"


def test_scale_features():
    """Test that scaling produces zero mean unit variance."""
    from src.data.preprocess_tabular import scale_features
    X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    X_test = np.array([[2, 3]], dtype=np.float64)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    # Scaled train should have ~zero mean
    assert abs(X_train_s.mean()) < 1e-10


def test_encode_categoricals():
    """Test one-hot encoding of categorical columns."""
    from src.data.preprocess_tabular import encode_categoricals
    df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'value': [1, 2, 3]})
    result = encode_categoricals(df, ['color'])
    assert 'color' not in result.columns
    assert result.shape[1] == 2  # value + one dummy column (drop_first)


def test_split_data():
    """Test that data split maintains proportions."""
    from src.data.preprocess_tabular import split_data
    X = np.random.rand(100, 5)
    y = np.array([0] * 50 + [1] * 50)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
