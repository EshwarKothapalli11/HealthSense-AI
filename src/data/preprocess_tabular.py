"""
preprocess_tabular.py — Preprocessing functions for tabular datasets (diabetes, heart).

Provides imputation, encoding, scaling, splitting, and full preparation pipelines
for the Pima Indians Diabetes and Heart Disease datasets.
"""

import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def impute_zeros(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace 0s with column median in specified biological columns.

    Args:
        df: Input DataFrame.
        columns: List of column names where 0 is biologically impossible
                 and should be replaced with the median.

    Returns:
        pd.DataFrame: DataFrame with zeros replaced by column medians.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col][df[col] != 0].median()
            df[col] = df[col].replace(0, median_val)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    One-hot encode specified categorical columns.

    Args:
        df: Input DataFrame.
        columns: List of categorical column names to encode.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns (drop_first=True).
    """
    existing_cols = [c for c in columns if c in df.columns]
    if existing_cols:
        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
    return df


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data and transform both train and test.

    Args:
        X_train: Training feature array.
        X_test: Test feature array.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Feature array.
        y: Target array.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def save_scaler(scaler: Any, path: str) -> None:
    """
    Save a fitted scaler to disk using pickle.

    Args:
        scaler: The fitted scaler object.
        path: File path to save the scaler.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved to {path}")


def load_scaler(path: str) -> Any:
    """
    Load a saved scaler from disk.

    Args:
        path: File path to the saved scaler.

    Returns:
        The loaded scaler object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_diabetes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Full preprocessing pipeline for the Pima Indians Diabetes dataset.

    Pipeline: load → impute zeros → split → scale → save scaler.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    """
    print("\n[INFO] Preparing Diabetes dataset...")

    df = pd.read_csv("data/raw/diabetes.csv")

    # Impute biologically impossible zeros
    df = impute_zeros(df, config.DIABETES_ZERO_IMPUTE_COLS)

    # Separate features and target
    feature_names = [c for c in df.columns if c != 'Outcome']
    X = df[feature_names].values
    y = df['Outcome'].values

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Save scaler
    save_scaler(scaler, os.path.join(config.SCALERS_DIR, "diabetes_scaler.pkl"))

    print(f"  ✓ Diabetes data ready — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


def prepare_heart() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Full preprocessing pipeline for the Heart Disease dataset.

    Pipeline: load → impute → encode categoricals → split → scale → save scaler.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    """
    print("\n[INFO] Preparing Heart Disease dataset...")

    df = pd.read_csv("data/raw/heart.csv")

    # Identify categorical columns (typically: sex, cp, fbs, restecg, exang, slope, ca, thal)
    categorical_cols = []
    for col in df.columns:
        if col == 'target':
            continue
        if df[col].nunique() <= 5 and col not in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            categorical_cols.append(col)

    # Impute zeros in continuous columns if applicable
    continuous_cols = [c for c in df.columns if c not in categorical_cols and c != 'target']
    zero_impute = [c for c in continuous_cols if (df[c] == 0).sum() > 0 and c != 'age']
    if zero_impute:
        df = impute_zeros(df, zero_impute)

    # Encode categoricals
    if categorical_cols:
        df = encode_categoricals(df, categorical_cols)

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Separate features and target
    feature_names = [c for c in df.columns if c != 'target']
    X = df[feature_names].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Save scaler
    save_scaler(scaler, os.path.join(config.SCALERS_DIR, "heart_scaler.pkl"))

    print(f"  ✓ Heart data ready — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    prepare_diabetes()
    prepare_heart()
