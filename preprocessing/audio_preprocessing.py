"""
Audio preprocessing utilities.
- Load audio (pad/trim to fixed duration)
- Extract MFCC, Chroma, and spectral features
"""

import numpy as np
import librosa
from config import Config


def load_audio(file_path, sr=Config.SAMPLE_RATE, duration=Config.AUDIO_DURATION):
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]
    return audio, sr


def extract_mfcc(audio, sr, n_mfcc=Config.N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def extract_chroma(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return np.mean(chroma.T, axis=0)


def extract_spectral_features(audio, sr):
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    return np.array([zcr, spectral_centroid, spectral_rolloff, rms], dtype=np.float32)


def preprocess_audio(file_path):
    audio, sr = load_audio(file_path)
    mfcc = extract_mfcc(audio, sr)
    chroma = extract_chroma(audio, sr)
    spectral = extract_spectral_features(audio, sr)
    features = np.concatenate([mfcc, chroma, spectral])
    return features.astype(np.float32)

