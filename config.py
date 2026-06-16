import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'artifacts', 'models')
SCALERS_DIR = os.path.join(BASE_DIR, 'artifacts', 'scalers')
PLOTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'plots')

# Dataset URLs
DIABETES_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
HEART_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
MENTAL_HEALTH_URL = 'https://raw.githubusercontent.com/AhmedSSoliman/sentiment-analysis-for-mental-health-Combined-Data/main/sentiment-analysis-for-mental-health-Combined%20Data.csv'

# Preprocessing Constants
DIABETES_COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
DIABETES_ZERO_IMPUTE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

MAX_TEXT_LEN = 100
VOCAB_SIZE = 5000
OOV_TOKEN = '<OOV>'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Deep Learning Hyperparameters (Mental Health LSTM)
LEARNING_RATE = 0.001
MAX_EPOCHS = 15
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# XGBoost Hyperparameters (Diabetes)
XGB_D_N_ESTIMATORS = 150
XGB_D_MAX_DEPTH = 3
XGB_D_LEARNING_RATE = 0.01
XGB_D_SUBSAMPLE = 0.8
XGB_D_COLSAMPLE_BYTREE = 0.7
XGB_D_MIN_CHILD_WEIGHT = 5
XGB_D_GAMMA = 0.1
XGB_D_REG_ALPHA = 0.01
XGB_D_REG_LAMBDA = 1.0

# XGBoost Hyperparameters (Heart Disease)
XGB_H_N_ESTIMATORS = 100
XGB_H_MAX_DEPTH = 4
XGB_H_LEARNING_RATE = 0.05
XGB_H_SUBSAMPLE = 0.9
XGB_H_COLSAMPLE_BYTREE = 0.8
XGB_H_MIN_CHILD_WEIGHT = 5
XGB_H_GAMMA = 0.1
XGB_H_REG_ALPHA = 0.01
XGB_H_REG_LAMBDA = 1.0

# Multi-Modal Fusion Configuration
FUSION_WEIGHT_DIABETES = 0.35
FUSION_WEIGHT_HEART = 0.35
FUSION_WEIGHT_MENTAL = 0.30
MENTAL_AMPLIFICATION_ALPHA = 0.12
