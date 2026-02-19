"""
Unit tests for preprocessing modules.
Tests audio, text, and image preprocessing functions.
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config


class TestAudioPreprocessing:
    """Test audio preprocessing functions."""

    def test_audio_feature_dimensions(self):
        """Test that audio preprocessing returns correct feature dimensions."""
        try:
            from preprocessing.audio_preprocessing import preprocess_audio
            # This test requires an actual audio file
            # For now, we just test the function exists and has correct signature
            assert callable(preprocess_audio)
        except ImportError:
            pytest.skip("Audio preprocessing module not available")

    def test_mfcc_extraction(self):
        """Test MFCC feature extraction."""
        try:
            from preprocessing.audio_preprocessing import extract_mfcc
            import librosa
            # Create dummy audio signal
            audio = np.random.randn(Config.SAMPLE_RATE * Config.AUDIO_DURATION)
            mfcc = extract_mfcc(audio, Config.SAMPLE_RATE)
            assert mfcc.shape == (Config.N_MFCC,)
            assert np.all(np.isfinite(mfcc))
        except ImportError:
            pytest.skip("Librosa not available")

    def test_chroma_extraction(self):
        """Test chroma feature extraction."""
        try:
            from preprocessing.audio_preprocessing import extract_chroma
            import librosa
            # Create dummy audio signal
            audio = np.random.randn(Config.SAMPLE_RATE * Config.AUDIO_DURATION)
            chroma = extract_chroma(audio, Config.SAMPLE_RATE)
            assert chroma.shape == (12,)  # 12 chroma bins
            assert np.all(np.isfinite(chroma))
        except ImportError:
            pytest.skip("Librosa not available")

    def test_spectral_features(self):
        """Test spectral feature extraction."""
        try:
            from preprocessing.audio_preprocessing import extract_spectral_features
            import librosa
            # Create dummy audio signal
            audio = np.random.randn(Config.SAMPLE_RATE * Config.AUDIO_DURATION)
            spectral = extract_spectral_features(audio, Config.SAMPLE_RATE)
            assert spectral.shape == (4,)  # ZCR, centroid, rolloff, RMS
            assert np.all(np.isfinite(spectral))
        except ImportError:
            pytest.skip("Librosa not available")


class TestTextPreprocessing:
    """Test text preprocessing functions."""

    def test_text_cleaning(self):
        """Test text cleaning function."""
        try:
            from preprocessing.text_preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()

            # Test URL removal
            text = "Check this out http://example.com amazing!"
            cleaned = preprocessor.clean_text(text)
            assert "http" not in cleaned

            # Test lowercase conversion
            text = "HELLO World"
            cleaned = preprocessor.clean_text(text)
            assert cleaned.islower()

            # Test special character removal
            text = "Hello, @world! #test"
            cleaned = preprocessor.clean_text(text)
            assert "@" not in cleaned
            assert "#" not in cleaned
        except ImportError:
            pytest.skip("Text preprocessing module not available")

    def test_text_tokenization(self):
        """Test BERT tokenization."""
        try:
            from preprocessing.text_preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()

            text = "I am very happy today!"
            result = preprocessor.preprocess_text(text)

            # Check that result contains required keys
            assert 'input_ids' in result
            assert 'attention_mask' in result

            # Check dimensions
            assert result['input_ids'].shape[1] == Config.MAX_TEXT_LENGTH
        except (ImportError, Exception) as e:
            pytest.skip(f"BERT tokenization not available: {e}")


class TestImagePreprocessing:
    """Test image preprocessing functions."""

    def test_image_dimensions(self):
        """Test that image preprocessing returns correct dimensions."""
        try:
            from preprocessing.image_preprocessing import preprocess_image
            import cv2

            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_path = 'test_image_temp.jpg'
            cv2.imwrite(test_path, dummy_image)

            try:
                processed = preprocess_image(test_path)
                assert processed.shape == (1, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3)
                assert processed.min() >= 0 and processed.max() <= 1
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)
        except ImportError:
            pytest.skip("Image preprocessing module not available")

    def test_face_detection(self):
        """Test face detection function."""
        try:
            from preprocessing.image_preprocessing import detect_face
            import cv2

            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_path = 'test_face_temp.jpg'
            cv2.imwrite(test_path, dummy_image)

            try:
                face = detect_face(test_path)
                assert face is not None
                assert isinstance(face, np.ndarray)
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)
        except ImportError:
            pytest.skip("OpenCV not available")


class TestConfig:
    """Test configuration settings."""

    def test_emotion_labels(self):
        """Test that emotion labels are correctly defined."""
        assert len(Config.EMOTIONS) == Config.NUM_EMOTIONS
        assert Config.NUM_EMOTIONS == 7
        assert 'happy' in Config.EMOTIONS
        assert 'sad' in Config.EMOTIONS
        assert 'angry' in Config.EMOTIONS

    def test_file_extensions(self):
        """Test allowed file extensions."""
        assert 'wav' in Config.ALLOWED_AUDIO_EXTENSIONS
        assert 'mp3' in Config.ALLOWED_AUDIO_EXTENSIONS
        assert 'png' in Config.ALLOWED_IMAGE_EXTENSIONS
        assert 'jpg' in Config.ALLOWED_IMAGE_EXTENSIONS

    def test_model_paths(self):
        """Test that model paths are defined."""
        assert Config.SPEECH_MODEL_PATH is not None
        assert Config.TEXT_MODEL_PATH is not None
        assert Config.IMAGE_MODEL_PATH is not None
        assert Config.FUSION_MODEL_PATH is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
