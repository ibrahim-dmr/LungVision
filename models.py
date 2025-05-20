import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

class AdvancedLungXRayModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        
        # EfficientNetV2
        self.efficientnet = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # EfficientNet çıktısı
        output = self.efficientnet(x)
        output = self.dropout(output)
        
        return {
            'output': output,
            'uncertainty': torch.zeros(x.size(0), device=x.device)  # Basit belirsizlik
        }
    
    def predict_with_uncertainty(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs['output'], dim=1)
            
            return {
                'probabilities': probs,
                'uncertainty': outputs['uncertainty']
            }
    
    def detect_ood(self, x, threshold=0.1):
        with torch.no_grad():
            outputs = self.forward(x)
            uncertainty = outputs['uncertainty']
            return uncertainty > threshold
            
    def save(self, path='models/best_model.pth'):
        # Klasörün var olduğundan emin ol
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Modeli kaydet
        torch.save(self.state_dict(), path)
        print(f'Model başarıyla kaydedildi: {path}')
        
    def load(self, path='models/best_model.pth'):
        # Modeli yükle
        self.load_state_dict(torch.load(path))
        print(f'Model başarıyla yüklendi: {path}') 