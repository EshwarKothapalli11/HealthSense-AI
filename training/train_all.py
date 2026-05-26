"""
train_all.py — Training entry point for HealthSense AI.

Runs the complete pipeline:
1. Download all datasets
2. Preprocess and train Diabetes DNN
3. Preprocess and train Heart Disease DNN
4. Preprocess and train Mental Health LSTM
5. Evaluate all models and generate reports
6. Print final summary table
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import config
from src.data.download_datasets import download_all
from src.data.preprocess_tabular import prepare_diabetes, prepare_heart
from src.data.preprocess_text import prepare_mental_health
from src.models.diabetes_model import train_diabetes_model
from src.models.heart_model import train_heart_model
from src.models.mental_model import train_mental_model
from src.evaluation.evaluator import ModelEvaluator


def main() -> None:
    """Run the complete training and evaluation pipeline."""
    print("=" * 70)
    print("  🏥 HealthSense AI — Full Training Pipeline")
    print("=" * 70)

    # Ensure output directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.SCALERS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    evaluator = ModelEvaluator()
    results = []

    # ---- Step 1: Download datasets ----
    print("\n" + "=" * 70)
    print("  STEP 1: Downloading Datasets")
    print("=" * 70)
    download_all()

    # ---- Step 2: Diabetes Model ----
    print("\n" + "=" * 70)
    print("  STEP 2: Diabetes DNN")
    print("=" * 70)

    X_train_d, X_test_d, y_train_d, y_test_d, diabetes_features = prepare_diabetes()
    diabetes_model, diabetes_history = train_diabetes_model(
        X_train_d, y_train_d, X_test_d, y_test_d
    )

    d_metrics = evaluator.evaluate(
        diabetes_model, X_test_d, y_test_d, "Diabetes"
    )
    evaluator.plot_training_history(diabetes_history, "Diabetes")
    evaluator.plot_confusion_matrix(
        y_test_d, d_metrics['y_pred'], "Diabetes",
        labels=['No Diabetes', 'Diabetes']
    )

    # Feature importance for diabetes
    d_importance = evaluator.compute_permutation_importance(
        diabetes_model, X_test_d, y_test_d, diabetes_features
    )
    evaluator.plot_feature_importance(d_importance, "Diabetes")

    results.append({
        'Model': 'Diabetes DNN',
        'Accuracy': f"{d_metrics['accuracy']:.4f}",
        'AUC': f"{d_metrics['auc']:.4f}",
        'F1-Score': f"{d_metrics['f1_score']:.4f}"
    })

    # ---- Step 3: Heart Disease Model ----
    print("\n" + "=" * 70)
    print("  STEP 3: Heart Disease DNN")
    print("=" * 70)

    X_train_h, X_test_h, y_train_h, y_test_h, heart_features = prepare_heart()
    heart_model, heart_history = train_heart_model(
        X_train_h, y_train_h, X_test_h, y_test_h
    )

    h_metrics = evaluator.evaluate(
        heart_model, X_test_h, y_test_h, "Heart"
    )
    evaluator.plot_training_history(heart_history, "Heart")
    evaluator.plot_confusion_matrix(
        y_test_h, h_metrics['y_pred'], "Heart",
        labels=['No Disease', 'Heart Disease']
    )

    # Feature importance for heart
    h_importance = evaluator.compute_permutation_importance(
        heart_model, X_test_h, y_test_h, heart_features
    )
    evaluator.plot_feature_importance(h_importance, "Heart")

    results.append({
        'Model': 'Heart Disease DNN',
        'Accuracy': f"{h_metrics['accuracy']:.4f}",
        'AUC': f"{h_metrics['auc']:.4f}",
        'F1-Score': f"{h_metrics['f1_score']:.4f}"
    })

    # ---- Step 4: Mental Health Model ----
    print("\n" + "=" * 70)
    print("  STEP 4: Mental Health LSTM")
    print("=" * 70)

    X_train_m, X_test_m, y_train_m, y_test_m, tokenizer = prepare_mental_health()
    mental_model, mental_history = train_mental_model(
        X_train_m, y_train_m, X_test_m, y_test_m
    )

    m_metrics = evaluator.evaluate(
        mental_model, X_test_m, y_test_m, "Mental"
    )
    evaluator.plot_training_history(mental_history, "Mental")
    evaluator.plot_confusion_matrix(
        y_test_m, m_metrics['y_pred'], "Mental",
        labels=['Healthy', 'Depressed/Stressed']
    )

    results.append({
        'Model': 'Mental Health LSTM',
        'Accuracy': f"{m_metrics['accuracy']:.4f}",
        'AUC': f"{m_metrics['auc']:.4f}",
        'F1-Score': f"{m_metrics['f1_score']:.4f}"
    })

    # ---- Step 5: Final Summary ----
    print("\n" + "=" * 70)
    print("  📊 FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))

    # Save performance summary
    summary_df.to_csv(
        os.path.join(config.PLOTS_DIR, "performance_summary.csv"), index=False
    )

    print("\n" + "=" * 70)
    print("  ✅ All models trained and evaluated successfully!")
    print(f"  📁 Models saved to: {config.MODELS_DIR}")
    print(f"  📁 Scalers saved to: {config.SCALERS_DIR}")
    print(f"  📁 Plots saved to: {config.PLOTS_DIR}")
    print("=" * 70)
    print("\n  🚀 Run 'python main.py' to launch the HealthSense AI dashboard!")


if __name__ == "__main__":
    main()
