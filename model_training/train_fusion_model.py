"""
Advanced Multimodal Fusion Model for Emotion Classification
Combines Speech, Text, and Image features using attention-based fusion.

Features:
- Late fusion with learned attention weights per modality
- Confidence-weighted prediction combination
- Cross-modal attention for better feature interaction
- Ensemble of multiple fusion strategies

Usage:
  python model_training/train_fusion_model.py --data-root datasets --epochs 100
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import joblib

from config import Config

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  Using: {device}")
else:
    print("⚠ No GPU detected, using CPU")


# ============== Feature Extractors (load pre-trained models) ==============

class SpeechFeatureExtractor:
    """Extract features from speech using pre-trained speech model."""
    
    def __init__(self, model_path, scaler_path):
        import tensorflow as tf
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Create feature extractor (before final classification layer)
        self.feature_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output  # Before final dense+softmax
        )
        self.num_features = 64  # From the model architecture
        
    def extract_features(self, audio_path):
        """Extract features from audio file."""
        from preprocessing.audio_preprocessing import preprocess_audio
        features = preprocess_audio(audio_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get intermediate features
        intermediate = self.feature_model.predict(features_scaled, verbose=0)
        
        # Also get predictions for decision-level fusion
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        
        return intermediate[0], predictions


class TextFeatureExtractor:
    """Extract features from text using pre-trained BERT model."""
    
    def __init__(self, model_path):
        from transformers import BertTokenizer, BertForSequenceClassification
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device)
        self.num_features = 768  # BERT hidden size
        
    def extract_features(self, text):
        """Extract features from text."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = self.model.bert(input_ids, attention_mask=attention_mask)
            # Get [CLS] token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Get predictions
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            predictions = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return cls_embedding, predictions


class ImageFeatureExtractor:
    """Extract features from images using pre-trained ResNet model."""
    
    def __init__(self, model_path):
        from torchvision import transforms
        
        # Load the model architecture (same as training)
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.num_features = 512  # Feature size after ResNet + FC
        
    def _build_model(self):
        """Build the same model architecture as used in training."""
        from torchvision import models
        
        class ImageEmotionModel(nn.Module):
            def __init__(self, num_classes=7):
                super().__init__()
                self.base = models.resnet50(weights=None)
                in_features = self.base.fc.in_features
                self.base.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                return self.base(x)
            
            def extract_features(self, x):
                # Get features before final classification
                for name, module in self.base.named_children():
                    if name == 'fc':
                        # Only go through first layers of FC
                        x = module[0](x)  # Dropout
                        x = module[1](x)  # Linear -> 512
                        x = module[2](x)  # ReLU
                        return x
                    else:
                        x = module(x)
                return x
        
        return ImageEmotionModel()
    
    def extract_features(self, image_path):
        """Extract features from image file."""
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get intermediate features
            features = self.model.extract_features(image_tensor).cpu().numpy()[0]
            
            # Get predictions
            logits = self.model(image_tensor)
            predictions = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return features, predictions


# ============== Fusion Model Architecture ==============

class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion layer."""
    
    def __init__(self, feature_dims, hidden_dim=256):
        super().__init__()
        self.num_modalities = len(feature_dims)
        
        # Project each modality to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for dim in feature_dims
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of tensors, one per modality
        Returns:
            Fused features tensor
        """
        # Project all modalities
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]
        
        # Concatenate for attention computation
        concat = torch.cat(projected, dim=-1)
        
        # Compute attention weights
        attention_weights = self.attention(concat)
        
        # Stack projected features
        stacked = torch.stack(projected, dim=1)  # (batch, num_modalities, hidden_dim)
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, num_modalities, 1)
        fused = (stacked * attention_weights).sum(dim=1)  # (batch, hidden_dim)
        
        return fused, attention_weights.squeeze(-1)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for better feature interaction."""
    
    def __init__(self, hidden_dim=256, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query, key_value):
        """Apply cross-attention where query attends to key_value."""
        attn_out, _ = self.attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))


class MultiModalFusionModel(nn.Module):
    """
    Advanced multimodal fusion model combining:
    1. Feature-level fusion with learned attention
    2. Cross-modal attention
    3. Decision-level fusion with confidence weighting
    """
    
    def __init__(self, speech_dim=64, text_dim=768, image_dim=512, num_classes=7, hidden_dim=256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Feature projections
        self.speech_proj = nn.Sequential(
            nn.Linear(speech_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cross-modal attention (each modality attends to others)
        self.cross_attn_speech = CrossModalAttention(hidden_dim)
        self.cross_attn_text = CrossModalAttention(hidden_dim)
        self.cross_attn_image = CrossModalAttention(hidden_dim)
        
        # Attention fusion
        self.attention_fusion = AttentionFusion([hidden_dim, hidden_dim, hidden_dim], hidden_dim)
        
        # Decision-level fusion (probability weighting)
        self.decision_weights = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final classifier combining both fusion strategies
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, speech_feat, text_feat, image_feat, speech_pred, text_pred, image_pred):
        """
        Args:
            speech_feat, text_feat, image_feat: Feature vectors from each modality
            speech_pred, text_pred, image_pred: Prediction probabilities from each modality
        """
        batch_size = speech_feat.size(0)
        
        # Project features
        speech_proj = self.speech_proj(speech_feat).unsqueeze(1)  # (batch, 1, hidden)
        text_proj = self.text_proj(text_feat).unsqueeze(1)
        image_proj = self.image_proj(image_feat).unsqueeze(1)
        
        # Cross-modal attention (each modality attends to the concatenation of others)
        other_speech = torch.cat([text_proj, image_proj], dim=1)  # (batch, 2, hidden)
        other_text = torch.cat([speech_proj, image_proj], dim=1)
        other_image = torch.cat([speech_proj, text_proj], dim=1)
        
        speech_enhanced = self.cross_attn_speech(speech_proj, other_speech).squeeze(1)
        text_enhanced = self.cross_attn_text(text_proj, other_text).squeeze(1)
        image_enhanced = self.cross_attn_image(image_proj, other_image).squeeze(1)
        
        # Attention-based feature fusion
        fused_features, attention_weights = self.attention_fusion([speech_enhanced, text_enhanced, image_enhanced])
        
        # Decision-level fusion (weighted combination of predictions)
        all_preds = torch.cat([speech_pred, text_pred, image_pred], dim=-1)
        decision_weights = self.decision_weights(all_preds)  # (batch, 3)
        
        # Weighted prediction combination
        stacked_preds = torch.stack([speech_pred, text_pred, image_pred], dim=1)  # (batch, 3, num_classes)
        weighted_preds = (stacked_preds * decision_weights.unsqueeze(-1)).sum(dim=1)  # (batch, num_classes)
        
        # Combine feature-level and decision-level fusion
        combined = torch.cat([fused_features, weighted_preds], dim=-1)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits, attention_weights, decision_weights


# ============== Dataset ==============

class FusionDataset(Dataset):
    """Dataset for multimodal fusion training."""
    
    def __init__(self, speech_features, text_features, image_features,
                 speech_preds, text_preds, image_preds, labels):
        self.speech_features = torch.FloatTensor(speech_features)
        self.text_features = torch.FloatTensor(text_features)
        self.image_features = torch.FloatTensor(image_features)
        self.speech_preds = torch.FloatTensor(speech_preds)
        self.text_preds = torch.FloatTensor(text_preds)
        self.image_preds = torch.FloatTensor(image_preds)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.speech_features[idx],
            self.text_features[idx],
            self.image_features[idx],
            self.speech_preds[idx],
            self.text_preds[idx],
            self.image_preds[idx],
            self.labels[idx]
        )


# ============== Training Functions ==============

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        speech_feat, text_feat, image_feat, speech_pred, text_pred, image_pred, labels = [
            x.to(device) for x in batch
        ]
        
        optimizer.zero_grad()
        logits, _, _ = model(speech_feat, text_feat, image_feat, speech_pred, text_pred, image_pred)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attn_weights = []
    all_decision_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            speech_feat, text_feat, image_feat, speech_pred, text_pred, image_pred, labels = [
                x.to(device) for x in batch
            ]
            
            logits, attn_weights, decision_weights = model(
                speech_feat, text_feat, image_feat, speech_pred, text_pred, image_pred
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attn_weights.append(attn_weights.cpu().numpy())
            all_decision_weights.append(decision_weights.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    attn_weights = np.concatenate(all_attn_weights, axis=0).mean(axis=0)
    decision_weights = np.concatenate(all_decision_weights, axis=0).mean(axis=0)
    
    return total_loss / len(dataloader), accuracy, all_preds, all_labels, attn_weights, decision_weights


def generate_synthetic_data(num_samples=5000):
    """
    Generate synthetic fusion data for training.
    In production, you would extract features from actual multimodal samples.
    """
    print("\nGenerating synthetic training data...")
    print("(In production, use actual multimodal samples)")
    
    np.random.seed(42)
    num_classes = Config.NUM_EMOTIONS
    
    # Simulate feature dimensions
    speech_dim = 64
    text_dim = 768
    image_dim = 512
    
    # Generate data with some class structure
    speech_features = []
    text_features = []
    image_features = []
    speech_preds = []
    text_preds = []
    image_preds = []
    labels = []
    
    for i in range(num_samples):
        label = i % num_classes
        
        # Create features with some class-correlated signal
        speech_feat = np.random.randn(speech_dim) + label * 0.3
        text_feat = np.random.randn(text_dim) + label * 0.2
        image_feat = np.random.randn(image_dim) + label * 0.25
        
        # Create somewhat realistic predictions (peaked at correct class with noise)
        speech_pred = np.random.dirichlet(np.ones(num_classes) * 0.5)
        speech_pred[label] += np.random.uniform(0.3, 0.6)
        speech_pred /= speech_pred.sum()
        
        text_pred = np.random.dirichlet(np.ones(num_classes) * 0.5)
        text_pred[label] += np.random.uniform(0.4, 0.7)
        text_pred /= text_pred.sum()
        
        image_pred = np.random.dirichlet(np.ones(num_classes) * 0.5)
        image_pred[label] += np.random.uniform(0.2, 0.5)
        image_pred /= image_pred.sum()
        
        speech_features.append(speech_feat)
        text_features.append(text_feat)
        image_features.append(image_feat)
        speech_preds.append(speech_pred)
        text_preds.append(text_pred)
        image_preds.append(image_pred)
        labels.append(label)
    
    return (
        np.array(speech_features, dtype=np.float32),
        np.array(text_features, dtype=np.float32),
        np.array(image_features, dtype=np.float32),
        np.array(speech_preds, dtype=np.float32),
        np.array(text_preds, dtype=np.float32),
        np.array(image_preds, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of synthetic samples')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("MULTIMODAL FUSION MODEL TRAINING")
    print("="*60)
    print("\nArchitecture: Attention-based Cross-Modal Fusion")
    print("Features:")
    print("  - Cross-modal attention for feature interaction")
    print("  - Learned attention weights per modality")
    print("  - Decision-level fusion with confidence weighting")
    print("  - Combined feature + decision fusion")

    # Generate training data
    (speech_features, text_features, image_features,
     speech_preds, text_preds, image_preds, labels) = generate_synthetic_data(args.num_samples)
    
    print(f"\nDataset size: {len(labels)} samples")
    
    # Create dataset
    dataset = FusionDataset(
        speech_features, text_features, image_features,
        speech_preds, text_preds, image_preds, labels
    )
    
    # Split into train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"Training: {train_size}, Validation: {val_size}")

    # Create model
    model = MultiModalFusionModel(
        speech_dim=64,
        text_dim=768,
        image_dim=512,
        num_classes=Config.NUM_EMOTIONS,
        hidden_dim=256
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, attn_weights, decision_weights = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Attention weights: Speech={attn_weights[0]:.3f}, Text={attn_weights[1]:.3f}, Image={attn_weights[2]:.3f}")
        print(f"  Decision weights: Speech={decision_weights[0]:.3f}, Text={decision_weights[1]:.3f}, Image={decision_weights[2]:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'speech_dim': 64,
                    'text_dim': 768,
                    'image_dim': 512,
                    'num_classes': Config.NUM_EMOTIONS,
                    'hidden_dim': 256
                }
            }, Config.FUSION_MODEL_PATH.replace('.pkl', '.pt'))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n  Early stopping triggered!")
                break

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(Config.FUSION_MODEL_PATH.replace('.pkl', '.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, y_pred, y_true, attn_weights, decision_weights = evaluate(
        model, val_loader, criterion, device
    )
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.EMOTIONS))
    
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"\nLearned Modality Weights:")
    print(f"  Feature Attention: Speech={attn_weights[0]:.3f}, Text={attn_weights[1]:.3f}, Image={attn_weights[2]:.3f}")
    print(f"  Decision Weights:  Speech={decision_weights[0]:.3f}, Text={decision_weights[1]:.3f}, Image={decision_weights[2]:.3f}")
    print(f"\n✓ Model saved to: {Config.FUSION_MODEL_PATH.replace('.pkl', '.pt')}")


if __name__ == '__main__':
    main()
