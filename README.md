# 🏥 HealthSense AI — Multi-Modal Deep Learning Healthcare System

> **Intelligent health risk assessment powered by three deep learning models and a multi-modal fusion engine.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Gradio 4.x](https://img.shields.io/badge/Gradio-4.x-green.svg)](https://www.gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📋 Overview

HealthSense AI is a comprehensive healthcare assessment platform that uses **three independent deep learning models** to evaluate:

1. **🩸 Diabetes Risk** — Feedforward DNN trained on the Pima Indians Diabetes dataset
2. **❤️ Heart Disease Risk** — DNN with Batch Normalization trained on the UCI Heart Disease dataset
3. **🧠 Mental Health Status** — LSTM recurrent network for text-based depression/stress detection

These models are combined through a **Multi-Modal Fusion Engine** that generates unified health reports with risk tiers, summaries, and personalized recommendations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio Web UI (5 Tabs)                │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ Physical │  Mental  │ Unified  │  Model   │   About &   │
│  Health  │  Health  │  Report  │ Insights │ Architecture│
└────┬─────┴────┬─────┴────┬─────┴──────────┴─────────────┘
     │          │          │
     ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌──────────────────┐
│Diabetes │ │ Mental │ │  Health Fusion   │
│  DNN    │ │ LSTM   │ │     Engine       │
│  (8→1)  │ │(text→1)│ │ (3 models → 1)  │
└─────────┘ └────────┘ └──────────────────┘
     │          │          │
┌─────────┐    │     ┌─────────┐
│  Heart  │    │     │ Risk    │
│  DNN    │    │     │ Tiering │
│ (13→1)  │    │     └─────────┘
└─────────┘    │
               │
     ┌─────────┘
     ▼
┌──────────────┐
│ Explainability│
│ (Gradients)   │
└──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthsense-ai.git
cd healthsense-ai

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train all three models (downloads data automatically)
python train_all.py
```

This will:
1. Download all three datasets
2. Preprocess and train the Diabetes DNN
3. Preprocess and train the Heart Disease DNN
4. Preprocess and train the Mental Health LSTM
5. Generate evaluation plots and metrics

### Launching the UI

```bash
# Launch the Gradio web application
python main.py
```

The application will be available at `http://localhost:7860`

---

## 📁 Project Structure

```
healthsense-ai/
├── AGENTS.md                    # Agent instructions and code style
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Hyperparameters and paths
├── main.py                      # Application entry point
├── train_all.py                 # Training pipeline entry point
├── src/
│   ├── data/
│   │   ├── download_datasets.py # Dataset download and validation
│   │   ├── preprocess_tabular.py# Tabular data preprocessing
│   │   └── preprocess_text.py   # Text data preprocessing (NLP)
│   ├── models/
│   │   ├── diabetes_model.py    # Diabetes DNN architecture
│   │   ├── heart_model.py       # Heart Disease DNN architecture
│   │   └── mental_model.py      # Mental Health LSTM architecture
│   ├── fusion/
│   │   └── fusion_engine.py     # Multi-modal prediction fusion
│   ├── evaluation/
│   │   ├── evaluator.py         # Model evaluation and plotting
│   │   └── explainability.py    # Gradient saliency explanations
│   └── ui/
│       ├── app.py               # Gradio Blocks application
│       ├── components.py        # Reusable UI components
│       └── visualizations.py    # Gauge, radar, and bar charts
├── artifacts/
│   ├── models/                  # Saved .h5 model files
│   ├── scalers/                 # Saved .pkl preprocessors
│   └── plots/                   # Generated evaluation plots
└── tests/
    ├── test_models.py           # Model architecture tests
    ├── test_fusion.py           # Fusion engine tests
    └── test_preprocessing.py    # Preprocessing function tests
```

---

## 📊 Model Performance

| Model | Accuracy | AUC | F1-Score |
|-------|----------|-----|----------|
| Diabetes DNN | — | — | — |
| Heart Disease DNN | — | — | — |
| Mental Health LSTM | — | — | — |

*Performance metrics are populated after running `train_all.py`.*

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_models.py -v
pytest tests/test_fusion.py -v
pytest tests/test_preprocessing.py -v
```

---

## 🔬 Technical Details

### Fusion Engine
- **Mental Amplification:** `min(1.0, disease_prob + α × mental_prob × disease_prob)` where α = 0.12
- **Composite Score:** `0.35 × diabetes + 0.35 × heart + 0.30 × mental`
- **Risk Tiers:** Low (<25), Moderate (25-49), High (50-74), Critical (≥75)

### Explainability
- **Tabular Models:** Gradient saliency + Permutation importance
- **LSTM Model:** Token-level gradient saliency through embedding layer

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

HealthSense AI is an **educational and research project**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers for medical concerns.
