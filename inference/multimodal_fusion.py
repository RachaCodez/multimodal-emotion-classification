"""
Multimodal fusion using attention-based PyTorch fusion model.
Falls back to weighted average when the model is not available.
"""

from typing import Dict, Optional
import numpy as np
from config import Config

try:
    import joblib
except Exception:
    joblib = None

from inference.speech_inference import SpeechInference
from inference.text_inference import TextInference
from inference.image_inference import ImageInference


class MultimodalFusion:
    def __init__(self):
        self.emotions = Config.EMOTIONS
        self.weights = [0.3, 0.35, 0.35]  # speech, text, image
        self.fusion_model = None
        self.device = None
        self.torch = None
        
        # Initialize modality-specific inference modules
        self.speech_inference = SpeechInference()
        self.text_inference = TextInference()
        self.image_inference = ImageInference()
        
        # Try to load PyTorch fusion model
        try:
            import torch
            import torch.nn as nn
            
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            fusion_model_path = Config.FUSION_MODEL_PATH.replace('.pkl', '.pt')
            try:
                checkpoint = torch.load(fusion_model_path, map_location=self.device)
                config = checkpoint['config']
                
                # Build fusion model
                self.fusion_model = self._build_fusion_model(
                    speech_dim=config['speech_dim'],
                    text_dim=config['text_dim'],
                    image_dim=config['image_dim'],
                    num_classes=config['num_classes'],
                    hidden_dim=config['hidden_dim']
                )
                self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
                self.fusion_model.eval()
                self.fusion_model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load fusion model: {e}")
                self.fusion_model = None
        except ImportError:
            self.torch = None
    
    def _build_fusion_model(self, speech_dim, text_dim, image_dim, num_classes, hidden_dim):
        """Build the same fusion model architecture as used in training."""
        import torch
        import torch.nn as nn
        
        class CrossModalAttention(nn.Module):
            def __init__(self, hidden_dim=256, num_heads=4):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, query, key_value):
                attn_out, _ = self.attention(query, key_value, key_value)
                return self.norm(query + self.dropout(attn_out))
        
        class AttentionFusion(nn.Module):
            def __init__(self, feature_dims, hidden_dim=256):
                super().__init__()
                self.num_modalities = len(feature_dims)
                self.projections = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ) for dim in feature_dims
                ])
                self.attention = nn.Sequential(
                    nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, self.num_modalities),
                    nn.Softmax(dim=-1)
                )
                self.output_dim = hidden_dim
            
            def forward(self, features_list):
                projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]
                concat = torch.cat(projected, dim=-1)
                attention_weights = self.attention(concat)
                stacked = torch.stack(projected, dim=1)
                attention_weights = attention_weights.unsqueeze(-1)
                fused = (stacked * attention_weights).sum(dim=1)
                return fused, attention_weights.squeeze(-1)
        
        class MultiModalFusionModel(nn.Module):
            def __init__(self, speech_dim, text_dim, image_dim, num_classes, hidden_dim):
                super().__init__()
                self.num_classes = num_classes
                
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
                
                self.cross_attn_speech = CrossModalAttention(hidden_dim)
                self.cross_attn_text = CrossModalAttention(hidden_dim)
                self.cross_attn_image = CrossModalAttention(hidden_dim)
                
                self.attention_fusion = AttentionFusion([hidden_dim, hidden_dim, hidden_dim], hidden_dim)
                
                self.decision_weights = nn.Sequential(
                    nn.Linear(num_classes * 3, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3),
                    nn.Softmax(dim=-1)
                )
                
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
                speech_proj = self.speech_proj(speech_feat).unsqueeze(1)
                text_proj = self.text_proj(text_feat).unsqueeze(1)
                image_proj = self.image_proj(image_feat).unsqueeze(1)
                
                other_speech = torch.cat([text_proj, image_proj], dim=1)
                other_text = torch.cat([speech_proj, image_proj], dim=1)
                other_image = torch.cat([speech_proj, text_proj], dim=1)
                
                speech_enhanced = self.cross_attn_speech(speech_proj, other_speech).squeeze(1)
                text_enhanced = self.cross_attn_text(text_proj, other_text).squeeze(1)
                image_enhanced = self.cross_attn_image(image_proj, other_image).squeeze(1)
                
                fused_features, attention_weights = self.attention_fusion([speech_enhanced, text_enhanced, image_enhanced])
                
                all_preds = torch.cat([speech_pred, text_pred, image_pred], dim=-1)
                decision_weights = self.decision_weights(all_preds)
                
                stacked_preds = torch.stack([speech_pred, text_pred, image_pred], dim=1)
                weighted_preds = (stacked_preds * decision_weights.unsqueeze(-1)).sum(dim=1)
                
                combined = torch.cat([fused_features, weighted_preds], dim=-1)
                logits = self.classifier(combined)
                
                return logits, attention_weights, decision_weights
        
        return MultiModalFusionModel(speech_dim, text_dim, image_dim, num_classes, hidden_dim)

    def fuse_predictions(self, speech_probs, text_probs, image_probs) -> Dict:
        """Fuse predictions using weighted average (fallback)."""
        s = np.array(speech_probs) if speech_probs is not None else np.zeros(len(self.emotions))
        t = np.array(text_probs) if text_probs is not None else np.zeros(len(self.emotions))
        i = np.array(image_probs) if image_probs is not None else np.zeros(len(self.emotions))

        # Weighted average
        weighted = self.weights[0] * s + self.weights[1] * t + self.weights[2] * i
        if weighted.sum() > 0:
            weighted = weighted / weighted.sum()
        idx = int(np.argmax(weighted))
        return {
            'emotion': self.emotions[idx],
            'confidence': float(weighted[idx]),
            'all_probabilities': weighted.tolist(),
        }

    def fuse_with_attention(self, speech_feat, text_feat, image_feat, 
                            speech_pred, text_pred, image_pred) -> Dict:
        """Fuse using the attention-based fusion model."""
        if self.fusion_model is None or self.torch is None:
            return self.fuse_predictions(speech_pred, text_pred, image_pred)
        
        try:
            # Convert to tensors
            speech_feat_t = self.torch.FloatTensor(speech_feat).unsqueeze(0).to(self.device)
            text_feat_t = self.torch.FloatTensor(text_feat).unsqueeze(0).to(self.device)
            image_feat_t = self.torch.FloatTensor(image_feat).unsqueeze(0).to(self.device)
            speech_pred_t = self.torch.FloatTensor(speech_pred).unsqueeze(0).to(self.device)
            text_pred_t = self.torch.FloatTensor(text_pred).unsqueeze(0).to(self.device)
            image_pred_t = self.torch.FloatTensor(image_pred).unsqueeze(0).to(self.device)
            
            with self.torch.no_grad():
                logits, attn_weights, decision_weights = self.fusion_model(
                    speech_feat_t, text_feat_t, image_feat_t,
                    speech_pred_t, text_pred_t, image_pred_t
                )
                predictions = self.torch.nn.functional.softmax(logits, dim=-1)
                preds = predictions[0].cpu().numpy()
            
            idx = int(np.argmax(preds))
            return {
                'emotion': self.emotions[idx],
                'confidence': float(preds[idx]),
                'all_probabilities': preds.tolist(),
                'attention_weights': {
                    'speech': float(attn_weights[0][0]),
                    'text': float(attn_weights[0][1]),
                    'image': float(attn_weights[0][2])
                },
                'decision_weights': {
                    'speech': float(decision_weights[0][0]),
                    'text': float(decision_weights[0][1]),
                    'image': float(decision_weights[0][2])
                }
            }
        except Exception as e:
            print(f"Fusion model error: {e}")
            return self.fuse_predictions(speech_pred, text_pred, image_pred)

    def predict_multimodal(self, audio_path: Optional[str] = None, 
                           text: Optional[str] = None, 
                           image_path: Optional[str] = None):
        """
        Run multimodal emotion prediction.
        Supports any combination of modalities.
        """
        results = {}
        
        # Get predictions from each modality
        if audio_path:
            results['speech'] = self.speech_inference.predict(audio_path)
        if text:
            results['text'] = self.text_inference.predict(text)
        if image_path:
            results['image'] = self.image_inference.predict(image_path)

        # If multiple modalities provided, perform fusion
        if len(results) > 1:
            # Try to use attention-based fusion if all features available
            s_probs = results.get('speech', {}).get('all_probabilities') if 'speech' in results else None
            t_probs = results.get('text', {}).get('all_probabilities') if 'text' in results else None
            i_probs = results.get('image', {}).get('all_probabilities') if 'image' in results else None
            
            # Try to extract features for attention fusion
            if self.fusion_model is not None and audio_path and text and image_path:
                try:
                    s_feat, s_pred = self.speech_inference.extract_features(audio_path)
                    t_feat, t_pred = self.text_inference.extract_features(text)
                    i_feat, i_pred = self.image_inference.extract_features(image_path)
                    
                    if all(x is not None for x in [s_feat, t_feat, i_feat]):
                        results['fusion'] = self.fuse_with_attention(
                            s_feat, t_feat, i_feat, s_pred, t_pred, i_pred
                        )
                    else:
                        results['fusion'] = self.fuse_predictions(s_probs, t_probs, i_probs)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    results['fusion'] = self.fuse_predictions(s_probs, t_probs, i_probs)
            else:
                results['fusion'] = self.fuse_predictions(s_probs, t_probs, i_probs)
        
        return results
