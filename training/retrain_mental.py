"""
retrain_mental.py — Retrain only the Mental Health LSTM model.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from src.data.download_datasets import download_mental_health
from src.data.preprocess_text import prepare_mental_health
from src.models.mental_model import train_mental_model
from src.evaluation.evaluator import ModelEvaluator

print("=" * 60)
print("  Re-downloading Mental Health data...")
print("=" * 60)
download_mental_health()

print("\n" + "=" * 60)
print("  Preprocessing...")
print("=" * 60)
X_train, X_test, y_train, y_test, tokenizer = prepare_mental_health()

print("\n" + "=" * 60)
print("  Training Mental Health LSTM...")
print("=" * 60)
model, history = train_mental_model(X_train, y_train, X_test, y_test)

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test, y_test, 'Mental')
evaluator.plot_training_history(history, 'Mental')
evaluator.plot_confusion_matrix(
    y_test, metrics['y_pred'], 'Mental',
    labels=['Healthy', 'Depressed/Stressed']
)

acc = metrics['accuracy']
auc = metrics['auc']
f1 = metrics['f1_score']

print("\n" + "=" * 60)
print(f"  FINAL: Accuracy={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
print("=" * 60)
