"""
evaluator.py — Model evaluation with metrics, plots, and feature importance.

Provides comprehensive evaluation reporting for all three HealthSense AI models.
Supports both XGBoost (tabular) and Keras (LSTM) model types.
Uses a light glassmorphism-inspired plot style.
"""

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def _is_keras_model(model: Any) -> bool:
    """Check if a model is a Keras model (vs XGBoost/sklearn)."""
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Get probability predictions from either a Keras or XGBoost model.

    Args:
        model: Trained model (Keras or XGBoost).
        X: Input features.

    Returns:
        1D array of positive-class probabilities.
    """
    if _is_keras_model(model):
        return model.predict(X, verbose=0).flatten()
    else:
        # XGBoost / sklearn API
        return model.predict_proba(X)[:, 1]


def apply_light_glass_style(fig: plt.Figure, ax_list: list) -> None:
    """Apply the light glassmorphism style to matplotlib figures and axes."""
    fig.patch.set_facecolor("#eef4fc")
    fig.patch.set_alpha(0.0)
    for ax in ax_list:
        ax.set_facecolor("#f8fbff")
        ax.patch.set_alpha(0.6)
        for spine in ax.spines.values():
            spine.set_color("#b4cdeb")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#4a6080", labelsize=10)
        ax.xaxis.label.set_color("#2d4060")
        ax.yaxis.label.set_color("#2d4060")
        ax.title.set_color("#1a3a7a")
        ax.title.set_fontweight("bold")
        ax.grid(True, color="#b4cdeb", alpha=0.35, linestyle="--", linewidth=0.7)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization toolkit."""

    def __init__(self) -> None:
        """Initialize evaluator and ensure output directories exist."""
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        threshold: float = 0.5
    ) -> dict:
        """
        Evaluate a trained model on test data.

        Supports both Keras and XGBoost/sklearn models.

        Args:
            model: Trained model (Keras or XGBoost).
            X_test: Test feature array.
            y_test: True labels.
            model_name: Name identifier for the model.
            threshold: Classification threshold for binary prediction.

        Returns:
            Dictionary containing all computed metrics.
        """
        print(f"\n{'='*50}")
        print(f"  Evaluation Report: {model_name}")
        print(f"{'='*50}")

        # Predict probabilities (model-agnostic)
        y_prob = _predict_proba(model, X_test)
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n  Accuracy:  {acc:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

        metrics = {
            'model_name': model_name,
            'accuracy': acc,
            'auc': auc,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        return metrics

    def plot_training_history(
        self, history: Any, model_name: str
    ) -> str:
        """
        Plot training and validation curves.

        Supports both:
        - Keras History object (has .history dict with 'loss', 'accuracy', etc.)
        - XGBoost eval_results dict (has 'validation_0'/'validation_1' with 'logloss', 'auc')

        Args:
            history: Keras History or XGBoost evals_result dict.
            model_name: Name identifier for saved plot.

        Returns:
            File path of the saved plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.set_facecolor("#eef4fc")
        apply_light_glass_style(fig, [ax1, ax2])

        if hasattr(history, 'history'):
            # --- Keras History ---
            ax1.plot(history.history['loss'], color="#4a90d9", linewidth=2.5, label='Train Loss')
            ax1.plot(history.history['val_loss'], color="#9b72d4", linewidth=2.5,
                     label='Val Loss', linestyle="--")
            ax1.fill_between(range(len(history.history['loss'])),
                             history.history['loss'], alpha=0.08, color="#4a90d9")
            ax1.set_title(f'{model_name} — Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend(facecolor="#ffffff", labelcolor="#2d4060",
                       framealpha=0.9, edgecolor="#b4cdeb")

            ax2.plot(history.history['accuracy'], color="#4a90d9", linewidth=2.5, label='Train Accuracy')
            ax2.plot(history.history['val_accuracy'], color="#9b72d4", linewidth=2.5,
                     label='Val Accuracy', linestyle="--")
            ax2.fill_between(range(len(history.history['accuracy'])),
                             history.history['accuracy'], alpha=0.08, color="#4a90d9")
            ax2.set_title(f'{model_name} — Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend(facecolor="#ffffff", labelcolor="#2d4060",
                       framealpha=0.9, edgecolor="#b4cdeb")
        else:
            # --- XGBoost evals_result dict ---
            # Keys are 'validation_0' (train) and 'validation_1' (val)
            train_key = 'validation_0'
            val_key = 'validation_1'

            # Plot 1: Log Loss
            train_logloss = history[train_key]['logloss']
            val_logloss = history[val_key]['logloss']
            rounds = range(len(train_logloss))

            ax1.plot(rounds, train_logloss, color="#4a90d9", linewidth=2.5, label='Train LogLoss')
            ax1.plot(rounds, val_logloss, color="#9b72d4", linewidth=2.5,
                     label='Val LogLoss', linestyle="--")
            ax1.fill_between(rounds, train_logloss, alpha=0.08, color="#4a90d9")
            ax1.set_title(f'{model_name} — Log Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Boosting Round')
            ax1.set_ylabel('Log Loss')
            ax1.legend(facecolor="#ffffff", labelcolor="#2d4060",
                       framealpha=0.9, edgecolor="#b4cdeb")

            # Plot 2: AUC
            train_auc = history[train_key]['auc']
            val_auc = history[val_key]['auc']

            ax2.plot(rounds, train_auc, color="#4a90d9", linewidth=2.5, label='Train AUC')
            ax2.plot(rounds, val_auc, color="#9b72d4", linewidth=2.5,
                     label='Val AUC', linestyle="--")
            ax2.fill_between(rounds, train_auc, alpha=0.08, color="#4a90d9")
            ax2.set_title(f'{model_name} — AUC', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Boosting Round')
            ax2.set_ylabel('AUC')
            ax2.legend(facecolor="#ffffff", labelcolor="#2d4060",
                       framealpha=0.9, edgecolor="#b4cdeb")

        plt.tight_layout()
        path = os.path.join(config.PLOTS_DIR, f'{model_name}_history.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor="#eef4fc")
        plt.close(fig)
        print(f"  [SUCCESS] Training history plot saved: {path}")
        return path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        labels: list[str] | None = None
    ) -> str:
        """
        Plot a confusion matrix heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name identifier for saved plot.
            labels: Optional class label names.

        Returns:
            File path of the saved plot.
        """
        if labels is None:
            labels = ['Negative', 'Positive']

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.set_facecolor("#eef4fc")
        apply_light_glass_style(fig, [ax])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar_kws={'shrink': 0.8},
            linecolor="#b4cdeb", linewidths=0.8,
            annot_kws={"size": 14, "color": "#1a3a7a", "weight": "bold"},
        )
        ax.set_title(f'{model_name} — Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)

        plt.tight_layout()
        path = os.path.join(config.PLOTS_DIR, f'{model_name}_confusion.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor="#eef4fc")
        plt.close(fig)
        print(f"  [SUCCESS] Confusion matrix saved: {path}")
        return path

    def compute_permutation_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str]
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance for any model.

        For XGBoost/sklearn models, uses their native API directly.
        For Keras models, wraps them in a sklearn-compatible interface.

        Args:
            model: Trained model (Keras or XGBoost).
            X_test: Test features.
            y_test: True labels.
            feature_names: List of feature names.

        Returns:
            pd.DataFrame: Sorted by mean importance descending.
        """
        if _is_keras_model(model):
            # Wrap Keras model for sklearn compatibility
            class _KerasWrapper:
                def __init__(self, keras_model):
                    self._model = keras_model

                def fit(self, X, y):
                    return self  # No-op: model is already trained

                def predict(self, X):
                    return (self._model.predict(X, verbose=0).flatten() >= 0.5).astype(int)

                def score(self, X, y):
                    y_pred = self.predict(X)
                    return accuracy_score(y, y_pred)

            estimator = _KerasWrapper(model)
        else:
            # XGBoost/sklearn models already have .score()
            estimator = model

        result = permutation_importance(
            estimator, X_test, y_test,
            n_repeats=10,
            random_state=config.RANDOM_STATE,
            scoring=None  # Use estimator's score method
        )

        importance_df = pd.DataFrame({
            'feature': feature_names[:X_test.shape[1]],
            'importance_mean': result.importances_mean[:len(feature_names)],
            'importance_std': result.importances_std[:len(feature_names)]
        })
        importance_df = importance_df.sort_values(
            'importance_mean', ascending=False
        ).reset_index(drop=True)

        return importance_df

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, model_name: str
    ) -> str:
        """
        Plot top 10 features by permutation importance.

        Args:
            importance_df: DataFrame with feature importance data.
            model_name: Name identifier for saved plot.

        Returns:
            File path of the saved plot.
        """
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor("#eef4fc")
        apply_light_glass_style(fig, [ax])

        # Clinical blue gradient colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, top_n))

        bars = ax.barh(
            range(top_n), top_features['importance_mean'].values,
            xerr=top_features['importance_std'].values,
            color=colors, edgecolor='white', linewidth=0.5
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_title(
            f'{model_name} — Feature Importance (Top {top_n})',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Permutation Importance', fontsize=12)

        plt.tight_layout()
        path = os.path.join(config.PLOTS_DIR, f'{model_name}_importance.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor="#eef4fc")
        plt.close(fig)
        print(f"  [SUCCESS] Feature importance plot saved: {path}")
        return path
