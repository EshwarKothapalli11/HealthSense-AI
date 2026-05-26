"""
preprocess_text.py — Preprocessing functions for mental health text data.

Provides text cleaning, tokenization, padding, and the full preparation pipeline
for the mental health classification dataset.
"""

import os
import pickle
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config


def clean_text(text: str) -> str:
    """
    Clean a text string for NLP processing.

    Operations: lowercase, remove URLs, remove special characters, strip extra whitespace.

    Args:
        text: Raw input text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_tokenizer(texts: list[str], vocab_size: int = 5000) -> Tokenizer:
    """
    Build and fit a Keras Tokenizer on the given texts.

    Args:
        texts: List of text strings to fit the tokenizer on.
        vocab_size: Maximum number of words to keep.

    Returns:
        Fitted Keras Tokenizer.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    print(f"  ✓ Tokenizer built — vocabulary size: {min(len(tokenizer.word_index), vocab_size)}")
    return tokenizer


def texts_to_padded(
    texts: list[str], tokenizer: Tokenizer, maxlen: int = 100
) -> np.ndarray:
    """
    Convert texts to padded sequences.

    Args:
        texts: List of text strings.
        tokenizer: Fitted Keras Tokenizer.
        maxlen: Maximum sequence length.

    Returns:
        np.ndarray: Padded integer sequences of shape (n_samples, maxlen).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    return padded


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """
    Save a fitted tokenizer to disk using pickle.

    Args:
        tokenizer: Fitted Keras Tokenizer.
        path: File path to save the tokenizer.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  ✓ Tokenizer saved to {path}")


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a saved tokenizer from disk.

    Args:
        path: File path to the saved tokenizer.

    Returns:
        Loaded Keras Tokenizer.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_mental_health() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    """
    Full preprocessing pipeline for the mental health text dataset.

    Pipeline: load → clean text → build tokenizer → pad sequences → split → save tokenizer.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, tokenizer).
    """
    print("\n[INFO] Preparing Mental Health text dataset...")

    df = pd.read_csv("data/raw/mental_health.csv")

    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)

    # Remove empty texts
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)

    # Build tokenizer
    tokenizer = build_tokenizer(
        df['text_clean'].tolist(), vocab_size=config.VOCAB_SIZE
    )

    # Pad sequences
    X = texts_to_padded(
        df['text_clean'].tolist(), tokenizer, maxlen=config.MAX_TEXT_LEN
    )
    y = df['label'].values.astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # Save tokenizer
    save_tokenizer(tokenizer, os.path.join(config.SCALERS_DIR, "mental_tokenizer.pkl"))

    print(f"  ✓ Mental Health data ready — Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, tokenizer


if __name__ == "__main__":
    prepare_mental_health()
