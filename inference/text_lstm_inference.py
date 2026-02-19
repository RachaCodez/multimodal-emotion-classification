"""
Fast LSTM Text Emotion Inference

Inference time: ~10-50ms per sample (10x faster than BERT!)

Usage:
  python inference/text_lstm_inference.py "I am so happy today!"
  python inference/text_lstm_inference.py --text "This is terrible, I'm angry"
"""

import argparse
import os
import sys
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config


class FastTextEmotionPredictor:
    """Fast LSTM-based text emotion predictor."""

    def __init__(self, model_path=None, tokenizer_path=None):
        """Initialize the predictor with model and tokenizer."""
        self.model_path = model_path or Config.TEXT_MODEL_PATH
        self.tokenizer_path = tokenizer_path or Config.TEXT_MODEL_PATH.replace('.h5', '_tokenizer.pkl')

        print("Loading LSTM model...")
        self.model = load_model(self.model_path)

        print("Loading tokenizer...")
        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.emotions = Config.EMOTIONS
        self.max_length = Config.MAX_TEXT_LENGTH

        print("Model ready for inference!")

    def predict(self, text):
        """
        Predict emotion from text.

        Args:
            text: Input text string

        Returns:
            dict with 'emotion', 'confidence', and 'probabilities'
        """
        # Preprocess text
        text_clean = text.lower().strip()

        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')

        # Predict
        start_time = time.time()
        predictions = self.model.predict(padded, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_emotion = self.emotions[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Create probabilities dict
        probabilities = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotions, predictions)
        }

        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'inference_time_ms': inference_time
        }

    def predict_batch(self, texts):
        """
        Predict emotions for a batch of texts (even faster!).

        Args:
            texts: List of text strings

        Returns:
            List of prediction dicts
        """
        # Preprocess
        texts_clean = [t.lower().strip() for t in texts]

        # Tokenize and pad
        sequences = self.tokenizer.texts_to_sequences(texts_clean)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        # Predict
        start_time = time.time()
        predictions = self.model.predict(padded, verbose=0)
        total_time = (time.time() - start_time) * 1000

        # Process results
        results = []
        for i, preds in enumerate(predictions):
            predicted_idx = np.argmax(preds)
            predicted_emotion = self.emotions[predicted_idx]
            confidence = float(preds[predicted_idx])

            probabilities = {
                emotion: float(prob)
                for emotion, prob in zip(self.emotions, preds)
            }

            results.append({
                'text': texts[i],
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': probabilities
            })

        avg_time = total_time / len(texts)
        print(f"Batch inference: {total_time:.2f}ms total, {avg_time:.2f}ms per sample")

        return results


def main():
    parser = argparse.ArgumentParser(description='Fast LSTM text emotion prediction')
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--text', dest='text_arg', help='Text to analyze (alternative)')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--batch', action='store_true', help='Run batch demo')
    args = parser.parse_args()

    # Get text from either positional or named argument
    text = args.text or args.text_arg

    if not text and not args.batch:
        print("Error: Please provide text to analyze")
        print("\nUsage examples:")
        print('  python inference/text_lstm_inference.py "I am so happy!"')
        print('  python inference/text_lstm_inference.py --text "I am angry"')
        print('  python inference/text_lstm_inference.py --batch')
        return

    # Initialize predictor
    try:
        predictor = FastTextEmotionPredictor(model_path=args.model)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python model_training/train_lstm_text_model.py --csv datasets/text/train.txt --epochs 10")
        return

    print("\n" + "="*60)

    if args.batch:
        # Batch demo
        print("BATCH PREDICTION DEMO")
        print("="*60)

        test_texts = [
            "I am so happy and excited!",
            "This makes me really angry and frustrated",
            "I'm feeling very sad and depressed today",
            "That's hilarious! What a surprise!",
            "I'm terrified and scared of this",
            "This is disgusting and gross"
        ]

        results = predictor.predict_batch(test_texts)

        for result in results:
            print(f"\nText: '{result['text']}'")
            print(f"Emotion: {result['emotion'].upper()} ({result['confidence']:.2%})")
            print("All probabilities:")
            for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion:10s}: {prob:.2%}")

    else:
        # Single prediction
        print("TEXT EMOTION PREDICTION")
        print("="*60)
        print(f"Input: '{text}'")
        print("-"*60)

        result = predictor.predict(text)

        print(f"\nPredicted Emotion: {result['emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
        print("\nAll Emotion Probabilities:")
        for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 50)
            print(f"  {emotion:10s}: {prob:.2%} {bar}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
