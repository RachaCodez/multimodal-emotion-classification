"""
Text model training CLI using BERT fine-tuning - OPTIMIZED FOR GPU (RTX 4070)
Uses PyTorch BERT (HuggingFace Transformers)

Usage:
  python model_training/train_text_model.py --csv datasets/text/emotion_dataset.csv --text-col text --label-col label

CSV expected columns: text + label (label in Config.EMOTIONS)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from config import Config

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected! This script requires GPU. Please check CUDA installation.")
print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"  Using: {device}")


class EmotionDataset(Dataset):
    """Custom dataset for emotion classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV dataset')
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=128)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("TEXT EMOTION MODEL TRAINING (BERT - PyTorch)")
    print("="*60)

    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(args.csv)
    df = df[[args.text_col, args.label_col]].dropna()
    
    # Map labels
    label_to_idx = {e: i for i, e in enumerate(Config.EMOTIONS)}
    df = df[df[args.label_col].isin(label_to_idx.keys())]
    df['label_idx'] = df[args.label_col].map(label_to_idx)
    
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    for emotion in Config.EMOTIONS:
        count = (df[args.label_col] == emotion).sum()
        if count > 0:
            print(f"  {emotion}: {count} samples")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df[args.text_col].values,
        df['label_idx'].values,
        test_size=0.15,
        random_state=42,
        stratify=df['label_idx'].values
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Load tokenizer and create datasets
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset = EmotionDataset(X_val, y_val, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Load BERT model
    print("\nLoading BERT model...")
    # Get number of unique labels in training data
    num_labels = len(set(y_train))
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=Config.NUM_EMOTIONS
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            os.makedirs(Config.BERT_MODEL_PATH, exist_ok=True)
            model.save_pretrained(Config.BERT_MODEL_PATH)
            tokenizer.save_pretrained(Config.BERT_MODEL_PATH)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    model = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_PATH)
    model.to(device)
    
    val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, device)
    
    # Only use emotions that are in the dataset
    present_emotions = [e for e in Config.EMOTIONS if (np.array(y_true) == label_to_idx[e]).any()]
    present_indices = [label_to_idx[e] for e in present_emotions]
    
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_indices,
        target_names=present_emotions
    ))
    
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"✓ Model saved to: {Config.BERT_MODEL_PATH}")


if __name__ == '__main__':
    main()
