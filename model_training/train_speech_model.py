"""
Speech model training CLI - OPTIMIZED FOR GPU (RTX 4070)
Trains a deep DNN on engineered audio features (MFCC+Chroma+Spectral = 56 dims).

Usage:
  python model_training/train_speech_model.py --data-root datasets/speech --pattern "**/*.wav" --label-from parent

Supported label extraction:
  - parent: uses parent directory name as label (recommended)
  - name: parses emotion from filename substring
"""

from typing import Tuple, List, Dict
import os
import sys
import argparse
import glob
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU Configuration - MUST be at the top before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        print(f"  Using: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    raise RuntimeError("No GPU detected! This script requires GPU. Please check CUDA installation.")

from tensorflow import keras
from config import Config
from preprocessing.audio_preprocessing import preprocess_audio


def create_speech_model():
    """Create an optimized DNN for speech emotion recognition."""
    model = keras.Sequential([
        keras.layers.Input(shape=(56,)),  # 40 MFCC + 12 Chroma + 4 Spectral = 56
        
        # Block 1: Initial feature expansion
        keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.4),
        
        # Block 2: Deep feature learning
        keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.4),
        
        # Block 3: Feature compression
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),
        
        # Block 4: Further compression
        keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),
        
        # Block 5: Final representation
        keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.1),
        
        # Output layer
        keras.layers.Dense(Config.NUM_EMOTIONS, activation='softmax'),
    ])
    
    # Optimizer with gradient clipping for stability
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def one_hot(labels: List[int], num_classes: int) -> np.ndarray:
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, idx in enumerate(labels):
        y[i, idx] = 1.0
    return y


def load_dataset(data_root: str, pattern: str, label_from: str, name_map: Dict[str, str] = None):
    """Load and preprocess the audio dataset."""
    files = glob.glob(os.path.join(data_root, pattern), recursive=True)
    print(f"Found {len(files)} audio files")
    
    X: List[np.ndarray] = []
    y_labels: List[str] = []
    
    for i, fp in enumerate(files):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(files)}...", end='\r')
        try:
            feat = preprocess_audio(fp)
            X.append(feat)
            if label_from == 'parent':
                lbl = os.path.basename(os.path.dirname(fp)).lower()
            elif label_from == 'name':
                base = os.path.basename(fp).lower()
                lbl = None
                if name_map:
                    for key, val in name_map.items():
                        if key.lower() in base:
                            lbl = val
                            break
                if lbl is None:
                    raise ValueError(f"Could not map filename to label: {base}")
            else:
                raise ValueError('label_from must be "parent" or "name"')
            y_labels.append(lbl)
        except Exception as e:
            print(f'\nSkip {fp}: {e}')
    
    print(f"\nSuccessfully processed {len(X)} files")
    
    # Map labels to indices based on Config.EMOTIONS order
    label_to_idx = {e: i for i, e in enumerate(Config.EMOTIONS)}
    y_idx = [label_to_idx[lbl] for lbl in y_labels if lbl in label_to_idx]
    X = [x for x, lbl in zip(X, y_labels) if lbl in label_to_idx]
    X = np.array(X, dtype=np.float32)
    y = one_hot(y_idx, Config.NUM_EMOTIONS)
    
    # Print class distribution
    print("\nClass distribution:")
    for emotion in Config.EMOTIONS:
        count = y_labels.count(emotion)
        print(f"  {emotion}: {count} samples")
    
    return X, y


def augment_features(X: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
    """Add Gaussian noise for data augmentation."""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='datasets/speech', help='Root folder containing audio files')
    parser.add_argument('--pattern', default='**/*.wav', help='Glob pattern to find audio files')
    parser.add_argument('--label-from', default='parent', choices=['parent', 'name'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--augment', action='store_true', default=True, help='Apply data augmentation')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SPEECH EMOTION MODEL TRAINING")
    print("="*60)

    # Load data
    X, y = load_dataset(args.data_root, args.pattern, args.label_from)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, 
        stratify=np.argmax(y, axis=1)
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Data augmentation
    if args.augment:
        print("\nApplying data augmentation (3x)...")
        X_train_aug = np.vstack([
            X_train,
            augment_features(X_train, 0.05),
            augment_features(X_train, 0.1),
        ])
        y_train_aug = np.vstack([y_train, y_train, y_train])
        X_train, y_train = X_train_aug, y_train_aug
        print(f"Augmented training set: {len(X_train)} samples")

    # Build model
    print("\nBuilding model...")
    model = create_speech_model()
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/speech_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    os.makedirs('models', exist_ok=True)
    model.save(Config.SPEECH_MODEL_PATH)
    joblib.dump(scaler, Config.SPEECH_SCALER_PATH)
    print(f"\n✓ Model saved to: {Config.SPEECH_MODEL_PATH}")
    print(f"✓ Scaler saved to: {Config.SPEECH_SCALER_PATH}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    preds = model.predict(X_val)
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(preds, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.EMOTIONS))
    
    # Final accuracy
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✓ Final Validation Loss: {val_loss:.4f}")


if __name__ == '__main__':
    main()
