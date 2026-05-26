"""
mental_model.py — LSTM model for Mental Health text classification.

Architecture: Embedding -> SpatialDropout1D -> Bidirectional LSTM -> Dense -> Sigmoid.
Optimized for high accuracy with proper regularization.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Embedding, LSTM, SpatialDropout1D,
    Bidirectional, GlobalMaxPooling1D, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def build_mental_model(
    vocab_size: int = 5000,
    embedding_dim: int = 128,
    maxlen: int = 100
) -> tf.keras.Model:
    """
    Build a Bidirectional LSTM model for mental health text classification.

    Uses dual pooling (max + average) from the LSTM output for richer
    feature extraction, leading to higher classification accuracy.

    Args:
        vocab_size: Size of the token vocabulary.
        embedding_dim: Dimensionality of word embeddings.
        maxlen: Maximum sequence length.

    Returns:
        Keras Model (uncompiled).
    """
    inputs = Input(shape=(maxlen,), name='text_input')

    # Embedding layer
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = SpatialDropout1D(0.3)(x)

    # Bidirectional LSTM (return full sequence for pooling)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)

    # Dual pooling for richer feature extraction
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])

    # Classification head
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', name='mental_output')(x)

    return Model(inputs, outputs, name='MentalHealthLSTM')


def train_mental_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train the mental health LSTM model.

    Args:
        X_train: Training padded sequences.
        y_train: Training labels.
        X_val: Validation padded sequences.
        y_val: Validation labels.

    Returns:
        Tuple of (trained_model, training_history).
    """
    print("\n[INFO] Building Mental Health LSTM model...")
    model = build_mental_model(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=128,
        maxlen=config.MAX_TEXT_LEN
    )

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    model.summary()

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    print(f"  Class weights: {class_weights}")

    # Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=config.REDUCE_LR_PATIENCE,
            factor=config.REDUCE_LR_FACTOR,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODELS_DIR, 'mental_best.h5'),
            monitor='val_auc',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]

    print("\n[INFO] Training Mental Health LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("  -> Mental Health LSTM model training complete!")
    return model, history


if __name__ == "__main__":
    from src.data.preprocess_text import prepare_mental_health
    X_train, X_test, y_train, y_test, _ = prepare_mental_health()
    model, history = train_mental_model(X_train, y_train, X_test, y_test)
