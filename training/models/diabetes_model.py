"""
diabetes_model.py — DNN model for Pima Indians Diabetes prediction.

Architecture: 3-layer dense network with dropout regularization.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def build_diabetes_model(input_dim: int) -> tf.keras.Model:
    """
    Build a deep neural network for diabetes prediction.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(input_dim,), name='diabetes_input')
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', name='diabetes_output')(x)
    return Model(inputs, outputs, name='DiabetesDNN')


def train_diabetes_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train the diabetes DNN model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (trained_model, training_history).
    """
    print("\n[INFO] Building Diabetes model...")
    model = build_diabetes_model(input_dim=X_train.shape[1])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    model.summary()

    # Compute class weights for imbalanced data
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    print(f"  Class weights: {class_weights}")

    # Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=config.REDUCE_LR_PATIENCE,
            factor=config.REDUCE_LR_FACTOR,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODELS_DIR, 'diabetes_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n[INFO] Training Diabetes model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("  ✓ Diabetes model training complete!")
    return model, history


if __name__ == "__main__":
    from src.data.preprocess_tabular import prepare_diabetes
    X_train, X_test, y_train, y_test, _ = prepare_diabetes()
    model, history = train_diabetes_model(X_train, y_train, X_test, y_test)
