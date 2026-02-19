"""
Image model training CLI using transfer learning - OPTIMIZED FOR GPU (RTX 4070)
Uses PyTorch ResNet50 (avoids TensorFlow zlibwapi.dll issues on Windows)

Usage:
  python model_training/train_image_model.py --data-root datasets/images --epochs 30

Directory structure expected:
  data_root/
    happy/ ... images ...
    sad/ ...
    angry/ ...
    fear/ ...
    disgust/ ...
    surprise/ ...
    neutral/ ...
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from config import Config

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected! This script requires GPU. Please check CUDA installation.")
print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"  Using: {device}")


class ImageEmotionModel(nn.Module):
    """ResNet50-based image emotion classifier."""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        self.base = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Replace the final FC layer
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(dataloader), accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='Folder with class subdirectories')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("IMAGE EMOTION MODEL TRAINING (ResNet50 - PyTorch)")
    print("="*60)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\nLoading dataset...")
    full_dataset = datasets.ImageFolder(args.data_root, transform=train_transform)
    
    # Split into train/val
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform
    
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training: {train_size}, Validation: {val_size}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    print("\nBuilding ResNet50 model...")
    model = ImageEmotionModel(num_classes=len(class_names))
    model = model.to(device)
    
    # Freeze base layers initially
    for param in model.base.parameters():
        param.requires_grad = False
    # Unfreeze FC layer
    for param in model.base.fc.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.base.fc.parameters(), lr=args.learning_rate * 10, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Phase 1: Train only classifier
    print("\n" + "="*60)
    print("PHASE 1: TRAINING CLASSIFIER (Base Frozen)")
    print("="*60)
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(min(10, args.epochs)):
        print(f"\nEpoch {epoch + 1}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/image_model_best.pt')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered!")
                break

    # Phase 2: Fine-tune entire model
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING ENTIRE MODEL")
    print("="*60)
    
    # Unfreeze all layers
    for param in model.base.parameters():
        param.requires_grad = True
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 10)
    patience_counter = 0
    
    for epoch in range(10, args.epochs):
        print(f"\nEpoch {epoch + 1}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/image_model_best.pt')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered!")
                break

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load('models/image_model_best.pt'))
    
    val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, device)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"\n✓ Final Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save in a format compatible with the inference code
    # The inference code expects a Keras model, so we'll save the PyTorch model
    # and update the inference code later
    torch.save(model.state_dict(), Config.IMAGE_MODEL_PATH.replace('.h5', '.pt'))
    print(f"✓ Model saved to: {Config.IMAGE_MODEL_PATH.replace('.h5', '.pt')}")


if __name__ == '__main__':
    main()
