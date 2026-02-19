"""
Image inference using PyTorch ResNet50 model.
Fallback returns a neutral prediction to keep the app usable before training.
"""

from typing import Dict
import numpy as np
from config import Config



class ImageInference:
    def __init__(self):
        self.emotions = Config.EMOTIONS
        self.model = None
        self.transform = None
        self.device = None
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Set up transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Try to load PyTorch model
            model_path = Config.IMAGE_MODEL_PATH.replace('.h5', '.pt')
            try:
                self.model = self._build_model()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load image model: {e}")
                self.model = None
        except ImportError:
            self.torch = None
            self.model = None
    
    def _build_model(self):
        """Build the same model architecture as used in training."""
        import torch
        import torch.nn as nn
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
                """Extract features before final classification."""
                x = self.base.conv1(x)
                x = self.base.bn1(x)
                x = self.base.relu(x)
                x = self.base.maxpool(x)

                x = self.base.layer1(x)
                x = self.base.layer2(x)
                x = self.base.layer3(x)
                x = self.base.layer4(x)

                x = self.base.avgpool(x)
                x = torch.flatten(x, 1)
                
                # Enter custom fc
                # fc[0] is Dropout, fc[1] is Linear(2048, 512), fc[2] is ReLU
                x = self.base.fc[0](x)
                x = self.base.fc[1](x)
                x = self.base.fc[2](x)
                return x
        
        return ImageEmotionModel(num_classes=len(self.emotions))

    def _fallback(self) -> Dict:
        probs = np.ones(len(self.emotions)) * (0.1 / (len(self.emotions) - 1))
        idx = self.emotions.index('neutral')
        probs[idx] = 0.9
        return {
            'emotion': 'neutral',
            'confidence': float(probs[idx]),
            'all_probabilities': probs.tolist(),
        }

    def predict(self, image_file_path: str) -> Dict:
        if self.model is None or self.torch is None:
            return self._fallback()

        try:
            from PIL import Image
            
            # Load and transform image
            image = Image.open(image_file_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with self.torch.no_grad():
                outputs = self.model(image_tensor)
                predictions = self.torch.nn.functional.softmax(outputs, dim=-1)
                preds = predictions[0].cpu().numpy()
            
            idx = int(np.argmax(preds))
            return {
                'emotion': self.emotions[idx],
                'confidence': float(preds[idx]),
                'all_probabilities': preds.tolist(),
            }
        except Exception as e:
            print(f"Image inference error: {e}")
            return self._fallback()
    
    def extract_features(self, image_file_path: str):
        """Extract features for fusion."""
        if self.model is None:
            return None, None
        
        from PIL import Image
        
        image = Image.open(image_file_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with self.torch.no_grad():
            features = self.model.extract_features(image_tensor).cpu().numpy()[0]
            outputs = self.model(image_tensor)
            predictions = self.torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()[0]
        
        return features, predictions
