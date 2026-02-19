"""
Speech inference: loads a trained TF model when available.
If not available, falls back to a lightweight heuristic based on audio features.
"""

from typing import Dict, List
import numpy as np
from config import Config
from preprocessing.audio_preprocessing import preprocess_audio, extract_spectral_features, load_audio


class SpeechInference:
    def __init__(self):
        self.emotions = Config.EMOTIONS
        self.model = None
        self.scaler = None
        try:
            import tensorflow as tf  # local import to avoid import error at app import time
            self.tf = tf
            try:
                self.model = tf.keras.models.load_model(Config.SPEECH_MODEL_PATH)
            except Exception:
                self.model = None
            # Try to load feature scaler if available
            try:
                import joblib  # type: ignore
                if joblib and Config.SPEECH_SCALER_PATH:
                    self.scaler = joblib.load(Config.SPEECH_SCALER_PATH)
            except Exception:
                self.scaler = None
        except Exception:
            self.tf = None
            self.model = None
            self.scaler = None

    def _heuristic_predict(self, audio_path: str) -> Dict:
        # Simple heuristic: use RMS energy and spectral centroid to approximate arousal/valence
        audio, sr = load_audio(audio_path)
        spectral = extract_spectral_features(audio, sr)
        zcr, centroid, rolloff, rms = spectral

        # Heuristic mapping
        if rms > 0.06 and centroid > 2000:
            label = 'angry'
        elif rms < 0.02 and centroid < 1500:
            label = 'sad'
        else:
            label = 'neutral'

        probs = np.ones(len(self.emotions)) * (0.1 / (len(self.emotions) - 1))
        idx = self.emotions.index(label)
        probs[idx] = 0.9

        return {
            'emotion': label,
            'confidence': float(probs[idx]),
            'all_probabilities': probs.tolist(),
        }

    def predict(self, audio_file_path: str) -> Dict:
        if self.model is None or self.tf is None:
            return self._heuristic_predict(audio_file_path)

        # Use model
        features = preprocess_audio(audio_file_path)
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).squeeze(0)
        features = np.expand_dims(features, axis=0)
        predictions = self.model.predict(features)
        emotion_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][emotion_idx])
        emotion = self.emotions[emotion_idx]
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_probabilities': predictions[0].tolist(),
        }

    def extract_features(self, audio_file_path: str):
        """Extract features for fusion model."""
        if self.model is None or self.tf is None:
            return None, None
        
        # Get features
        features = preprocess_audio(audio_file_path)
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Get intermediate features (before final layer)
        try:
            feature_model = self.tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-3].output  # Before final dense+softmax
            )
            intermediate = feature_model.predict(features_scaled, verbose=0)[0]
        except Exception:
            # Fallback: use raw features
            intermediate = features_scaled[0][:64]  # Take first 64 features
        
        # Get predictions
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        
        return intermediate, predictions
