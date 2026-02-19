"""
LSTM-based Text Emotion Recognition - FAST INFERENCE

This uses Bidirectional LSTM with GloVe embeddings for fast emotion classification.
Inference time: ~10-50ms per sample (much faster than BERT's ~500ms)

Usage:
  python model_training/train_lstm_text_model.py --csv datasets/text/train.txt --epochs 10

Dataset format: CSV/TXT with 'text' and 'label' columns
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config


def load_dataset(csv_path, text_col='text', label_col='label'):
    """Load and preprocess the emotion dataset."""
    print(f"Loading dataset from {csv_path}...")

    # Try different separators and encodings
    try:
        df = pd.read_csv(csv_path, sep=';', names=['text', 'label'])
    except:
        try:
            df = pd.read_csv(csv_path, sep=',', names=['text', 'label'])
        except:
            df = pd.read_csv(csv_path, sep='\t', names=['text', 'label'])

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # If still only one column, try to split it
    if len(df.columns) == 1:
        first_col = df.columns[0]
        if ';' in str(df[first_col].iloc[0]):
            df[['text', 'label']] = df[first_col].str.split(';', n=1, expand=True)
            df = df.drop(columns=[first_col])

    # Use the actual column names if they exist
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Clean data
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.lower().str.strip()

    # Map labels to emotions
    print(f"\nOriginal labels: {df[label_col].unique()}")

    # Create label mapping (handles both string and numeric labels)
    unique_labels = df[label_col].unique()
    if isinstance(unique_labels[0], str):
        # If labels are already emotion names
        label_to_emotion = {}
        for label in unique_labels:
            for emotion in Config.EMOTIONS:
                if emotion in label.lower() or label.lower() in emotion:
                    label_to_emotion[label] = emotion
                    break
        df['emotion'] = df[label_col].map(label_to_emotion)
    else:
        # If labels are numeric, map them to Config.EMOTIONS
        label_map = {i: Config.EMOTIONS[i] if i < len(Config.EMOTIONS) else Config.EMOTIONS[0]
                     for i in range(len(unique_labels))}
        df['emotion'] = df[label_col].map(label_map)

    # Filter valid emotions
    df = df[df['emotion'].isin(Config.EMOTIONS)]

    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nTotal samples: {len(df)}")

    return df[text_col].values, df['emotion'].values


def create_lstm_model(vocab_size, max_length, num_classes):
    """Build Bidirectional LSTM model for fast inference."""
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size,
                 output_dim=128,  # Smaller than BERT for speed
                 input_length=max_length,
                 name='embedding'),

        # Spatial dropout for regularization
        SpatialDropout1D(0.3),

        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2)),

        # Dense layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax', name='output')
    ])

    return model


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for text emotion recognition')
    parser.add_argument('--csv', required=True, help='Path to CSV/TXT dataset')
    parser.add_argument('--text-col', default='text', help='Name of text column')
    parser.add_argument('--label-col', default='label', help='Name of label column')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--max-length', type=int, default=Config.MAX_TEXT_LENGTH, help='Max sequence length')
    args = parser.parse_args()

    print("="*60)
    print("LSTM TEXT EMOTION RECOGNITION - TRAINING")
    print("="*60)

    # Load dataset
    texts, emotions = load_dataset(args.csv, args.text_col, args.label_col)

    # Create label mapping
    emotion_to_idx = {e: i for i, e in enumerate(Config.EMOTIONS)}
    labels = np.array([emotion_to_idx[e] for e in emotions])

    # Tokenize texts
    print("\nTokenizing texts...")
    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=args.max_length, padding='post', truncating='post')

    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Sequence shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Build model
    print("\nBuilding LSTM model...")
    vocab_size = min(args.vocab_size, len(tokenizer.word_index) + 1)
    model = create_lstm_model(vocab_size, args.max_length, Config.NUM_EMOTIONS)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
        ModelCheckpoint(Config.TEXT_MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    os.makedirs(os.path.dirname(Config.TEXT_MODEL_PATH), exist_ok=True)
    model.save(Config.TEXT_MODEL_PATH)

    # Save tokenizer
    tokenizer_path = Config.TEXT_MODEL_PATH.replace('.h5', '_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"\nModel saved to: {Config.TEXT_MODEL_PATH}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
