"""
explainability.py — Model explainability for HealthSense AI.

Provides gradient-based saliency for Keras models and built-in feature
importance for XGBoost models, plus influential token extraction for LSTM.
"""

import numpy as np
import tensorflow as tf


def _is_keras_model(model) -> bool:
    """Check if a model is a Keras model (vs XGBoost/sklearn)."""
    return isinstance(model, tf.keras.Model)


def compute_gradient_saliency(
    model: tf.keras.Model, input_sample: np.ndarray
) -> np.ndarray:
    """
    Compute gradient saliency of model output with respect to input.

    Uses tf.GradientTape to compute the absolute gradient values,
    which indicate how sensitive the output is to each input feature.

    Args:
        model: Trained Keras model.
        input_sample: Single input sample as numpy array (1D or 2D with batch dim).

    Returns:
        np.ndarray: Absolute gradient values (saliency scores).
    """
    if input_sample.ndim == 1:
        input_sample = input_sample.reshape(1, -1)

    input_tensor = tf.cast(input_sample, tf.float32)
    input_var = tf.Variable(input_tensor)

    with tf.GradientTape() as tape:
        tape.watch(input_var)
        prediction = model(input_var, training=False)

    gradients = tape.gradient(prediction, input_var)
    saliency = np.abs(gradients.numpy()).flatten()

    return saliency


def get_top_influential_features(
    model,
    input_sample: np.ndarray,
    feature_names: list[str],
    top_n: int = 5
) -> list[dict]:
    """
    Get the top-N most influential features for a prediction.

    For Keras models: uses gradient saliency.
    For XGBoost models: uses built-in feature_importances_.

    Args:
        model: Trained model (Keras or XGBoost).
        input_sample: Single input sample.
        feature_names: List of feature names corresponding to input dimensions.
        top_n: Number of top features to return.

    Returns:
        List of dicts with 'feature' and 'score' keys, sorted by importance.
    """
    if _is_keras_model(model):
        # Gradient-based saliency for Keras
        saliency = compute_gradient_saliency(model, input_sample)
    else:
        # XGBoost: use built-in feature importances, handling calibration wrappers
        if hasattr(model, 'feature_importances_'):
            saliency = model.feature_importances_
        elif hasattr(model, 'calibrated_classifiers_'):
            importances = []
            for c in model.calibrated_classifiers_:
                estimator = getattr(c, 'base_estimator', getattr(c, 'estimator', None))
                if estimator is not None and hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            if importances:
                saliency = np.mean(importances, axis=0)
            else:
                saliency = np.zeros(len(feature_names))
        else:
            saliency = np.zeros(len(feature_names))

    # Ensure we don't exceed available features
    n_features = min(len(saliency), len(feature_names))
    top_n = min(top_n, n_features)

    # Get top indices
    top_indices = np.argsort(saliency[:n_features])[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            'feature': feature_names[idx],
            'score': float(saliency[idx])
        })

    return results


def get_top_influential_tokens(
    tokenizer: object,
    text_sequence: np.ndarray,
    model: tf.keras.Model,
    top_n: int = 5
) -> list[str]:
    """
    Get the top-N most influential tokens for an LSTM text prediction.

    Computes gradient saliency directly against the Embedding weights
    used in the prediction, making it robust to any model architecture.

    Args:
        tokenizer: Fitted Keras Tokenizer with index_word attribute.
        text_sequence: Padded integer sequence (1D or 2D with batch dim).
        model: Trained LSTM Keras model.
        top_n: Number of top tokens to return.

    Returns:
        List of top-N influential word strings.
    """
    if text_sequence.ndim == 1:
        text_sequence = text_sequence.reshape(1, -1)

    input_tensor = tf.constant(text_sequence, dtype=tf.int32)

    # Get the embedding layer
    embedding_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            embedding_layer = layer
            break

    if embedding_layer is None:
        return []

    # Compute gradients with respect to the embedding matrix.
    # By passing model.trainable_variables, we guarantee TF accepts the variable type.
    with tf.GradientTape() as tape:
        prediction = model(input_tensor, training=False)

    all_grads = tape.gradient(prediction, model.trainable_variables)
    
    if not all_grads or all_grads[0] is None:
        return []
        
    # The embedding matrix is the first variable in trainable_variables
    gradients = all_grads[0]

    # Convert IndexedSlices to dense tensor if needed
    if isinstance(gradients, tf.IndexedSlices):
        gradients = tf.convert_to_tensor(gradients)

    # Compute absolute saliency for each unit in the vocabulary
    gradients_np = np.abs(gradients.numpy())
    vocab_saliency = np.sum(gradients_np, axis=-1)  # Shape: (vocab_size,)

    sequence = text_sequence.flatten()
    index_word = getattr(tokenizer, 'index_word', {})
    
    token_scores = []
    # Only score unique non-pad tokens present in the current input
    for token_id in set(sequence):
        if token_id == 0:  # Skip padding
            continue
        word = index_word.get(int(token_id), f'<UNK-{token_id}>')
        # Ensure we don't index out of bounds if maxlen > vocab_size somehow
        if token_id < len(vocab_saliency):
            token_scores.append((word, float(vocab_saliency[token_id])))

    # Sort by descending saliency
    token_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in token_scores[:top_n]]

    return top_words
