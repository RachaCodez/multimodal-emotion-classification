"""
Text preprocessing utilities with BERT tokenizer support.
Includes simple cleaning and a fallback if transformers are unavailable.
"""

import re

try:
    from transformers import BertTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    BertTokenizer = None
    _TRANSFORMERS_AVAILABLE = False


class TextPreprocessor:
    def __init__(self, model_type='bert', max_length=128):
        self.model_type = model_type
        self.max_length = max_length
        self.tokenizer = None
        if model_type == 'bert' and _TRANSFORMERS_AVAILABLE:
            try:
                # Prefer local model path if available
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            except Exception:
                self.tokenizer = None

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()
        return text

    def tokenize_bert(self, text):
        if not self.tokenizer:
            return None
        text = self.clean_text(text)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
        )
        return encoding

    def preprocess_text(self, text):
        return self.tokenize_bert(text)

