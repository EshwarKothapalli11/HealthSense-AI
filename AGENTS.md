# AGENTS.md

## Project: HealthSense AI

## Code Style
- Follow PEP 8 strictly
- Type hints required on all function signatures
- Docstrings required on all public methods and classes
- No monolithic scripts — use modular functions and classes
- Models in src/models/, data in src/data/, UI in src/ui/, fusion in src/fusion/

## Tech Stack
- Python 3.10+
- TensorFlow 2.x / Keras for all deep learning models
- scikit-learn for preprocessing, metrics, permutation importance
- Gradio 4.x for the web UI
- matplotlib + seaborn for evaluation plots
- No BERT, no transformers — LSTM only for NLP
- All models saved as .h5, all preprocessors saved as .pkl

## Agent Roles
- Agent 1: Data pipeline — download, clean, preprocess all three datasets
- Agent 2: Model training — train all three DL models independently
- Agent 3: Evaluation — metrics, confusion matrices, training plots, explainability
- Agent 4: Fusion engine + Gradio UI — combine predictions, build full interface

## Testing
- pytest unit tests for each model's forward pass shape
- pytest unit tests for fusion engine edge cases
- pytest unit tests for preprocessing functions
