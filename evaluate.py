import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from models import AdvancedLungXRayModel
from preprocess_data import ImageDataset
import json

class ChestXRayDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def evaluate_model(model_path, test_dir, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = AdvancedLungXRayModel(num_classes=9)  # Update for 9 classes
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = ImageDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get class names
    class_names = sorted(os.listdir(test_dir))
    
    # Initialize lists for predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs['output'], 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=class_names))
    
    # Save evaluation results for visualization
    eval_results = {
        'accuracy': float(accuracy),
        'true_labels': [int(x) for x in all_true_labels],
        'predictions': [int(x) for x in all_predictions],
        'classes': class_names,
        'classification_report': report
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(eval_results, f)
    
    return accuracy, report

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transformation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and loader
    test_dataset = datasets.ImageFolder('data/processed/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Print class distribution
    print("\nClass distribution in test set:")
    class_counts = {}
    for class_name in test_dataset.classes:
        class_dir = os.path.join('data/processed/test', class_name)
        count = len(os.listdir(class_dir))
        class_counts[class_name] = count
        print(f"{class_name}: {count} images")
    
    # Print class mapping
    print("\nClass mapping:")
    for idx, class_name in enumerate(test_dataset.classes):
        print(f"{idx}: {class_name}")
    
    # Load model
    model = AdvancedLungXRayModel(num_classes=9)
    model.load('models/best_model.pth')
    model = model.to(device)
    
    # Evaluate model
    metrics = evaluate_model('models/best_model.pth', 'data/processed/test')
    
    # Print results
    print('\nEvaluation Results:')
    print(f'Overall Accuracy: {metrics[0]:.4f}')
    
    print('\nClassification Report:')
    print(metrics[1])
    
    # Save results to file
    with open('evaluation_results.txt', 'w') as f:
        f.write('Evaluation Results:\n')
        f.write(f'Overall Accuracy: {metrics[0]:.4f}\n\n')
        
        f.write('Classification Report:\n')
        f.write(metrics[1])

if __name__ == "__main__":
    main() 