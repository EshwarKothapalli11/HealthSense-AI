"""
heart_model.py — XGBoost model for Heart Disease prediction.

Architecture: Gradient-boosted tree ensemble with regularization and
early stopping on a held-out validation set.
"""

import os
import sys
import pickle
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def build_heart_model(input_dim: int = 13) -> xgb.XGBClassifier:
    """
    Build an XGBoost classifier for heart disease prediction.

    Args:
        input_dim: Number of input features (unused, kept for API parity).

    Returns:
        Configured (unfitted) XGBClassifier.
    """
    model = xgb.XGBClassifier(
        n_estimators=config.XGB_H_N_ESTIMATORS,
        max_depth=config.XGB_H_MAX_DEPTH,
        learning_rate=config.XGB_H_LEARNING_RATE,
        subsample=config.XGB_H_SUBSAMPLE,
        colsample_bytree=config.XGB_H_COLSAMPLE_BYTREE,
        min_child_weight=config.XGB_H_MIN_CHILD_WEIGHT,
        gamma=config.XGB_H_GAMMA,
        reg_alpha=config.XGB_H_REG_ALPHA,
        reg_lambda=config.XGB_H_REG_LAMBDA,
        objective='binary:logistic',
        eval_metric=['logloss', 'auc'],
        use_label_encoder=False,
        random_state=config.RANDOM_STATE,
        verbosity=1,
    )
    return model


def train_heart_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tuple[Any, dict]:
    """
    Train the heart disease XGBoost model with CV and probability calibration.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (calibrated_model, eval_results_dict).
    """
    from typing import Any
    print("\n[INFO] Tuning Heart Disease XGBoost model with GridSearchCV (5-fold)...")
    base_model = build_heart_model(input_dim=X_train.shape[1])

    # Compute scale_pos_weight for class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    scale_pos = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
    base_model.set_params(scale_pos_weight=scale_pos)
    print(f"  Class weights: {class_weights}")
    print(f"  scale_pos_weight: {scale_pos:.4f}")

    # Search over a compact, robust grid for best ROC-AUC
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best_base = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")

    # Train with early stopping on best estimator to get evals_result for plots
    print("  Extracting training history curves...")
    history_model = build_heart_model(input_dim=X_train.shape[1])
    history_model.set_params(**grid.best_params_)
    history_model.set_params(scale_pos_weight=scale_pos)
    history_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    eval_results = history_model.evals_result()

    # Fit probability calibrated classifier using 5-fold cross-validation
    print("  Fitting probability calibrator (CalibratedClassifierCV)...")
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(estimator=best_base, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)

    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, 'heart_best.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(calibrated_model, f)
    print(f"  [SUCCESS] Model saved to {model_path}")

    print("  [SUCCESS] Heart Disease XGBoost model training complete!")
    return calibrated_model, eval_results



if __name__ == "__main__":
    from src.data.preprocess_tabular import prepare_heart
    X_train, X_test, y_train, y_test, _ = prepare_heart()
    model, eval_results = train_heart_model(X_train, y_train, X_test, y_test)
