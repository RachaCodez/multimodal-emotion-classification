"""
Text inference using BERT sequence classification (PyTorch).
Fallback: simple keyword heuristic mapping.
"""

from typing import Dict
import numpy as np
from config import Config
from preprocessing.text_preprocessing import TextPreprocessor


KEYWORD_MAP = {
    'happy': ['happy', 'joy', 'glad', 'pleased', 'delighted', 'cheerful', 'love', 'excited'],
    'sad': ['sad', 'down', 'unhappy', 'depressed', 'blue', 'disappointed', 'heartbroken'],
    'angry': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'frustrated'],
    'fear': ['scared', 'afraid', 'fear', 'terrified', 'anxious', 'nervous', 'worried'],
    'disgust': ['disgust', 'gross', 'nasty', 'revolting', 'sick'],
    'surprise': ['surprised', 'amazed', 'astonished', 'wow', 'shocked'],
    'neutral': []
}


class TextInference:
    def __init__(self):
        self.emotions = Config.EMOTIONS
        self.model = None
        self.tokenizer = None
        self.device = None
        self.preprocessor = TextPreprocessor()
        
        # Try to initialize PyTorch BERT model
        try:
            import torch
            from transformers import BertTokenizer, BertForSequenceClassification
            
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_PATH)
                self.model = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_PATH)
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.model = None
                self.tokenizer = None
        except ImportError:
            self.torch = None
            self.model = None
            self.tokenizer = None

    def _keyword_heuristic(self, text: str) -> Dict:
        cleaned = self.preprocessor.clean_text(text) if hasattr(self.preprocessor, 'clean_text') else text.lower()
        selected = 'neutral'
        for label, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if f" {kw} " in f" {cleaned} ":
                    selected = label
                    break
            if selected != 'neutral':
                break
        probs = np.ones(len(self.emotions)) * (0.1 / (len(self.emotions) - 1))
        idx = self.emotions.index(selected)
        probs[idx] = 0.9
        return {
            'emotion': selected,
            'confidence': float(probs[idx]),
            'all_probabilities': probs.tolist(),
        }

    def predict(self, text: str) -> Dict:
        if self.model is None or self.tokenizer is None or self.torch is None:
            return self._keyword_heuristic(text)

        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=Config.MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with self.torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = predictions[0].cpu().numpy()
            
            idx = int(np.argmax(preds))
            return {
                'emotion': self.emotions[idx],
                'confidence': float(preds[idx]),
                'all_probabilities': preds.tolist(),
            }
        except Exception as e:
            print(f"Text inference error: {e}")
            return self._keyword_heuristic(text)
    
    def extract_features(self, text: str):
        """Extract BERT [CLS] embedding for fusion."""
        if self.model is None or self.tokenizer is None:
            return None, None
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with self.torch.no_grad():
            outputs = self.model.bert(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            predictions = self.torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return cls_embedding, predictions
