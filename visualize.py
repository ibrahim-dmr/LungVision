import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from collections import Counter
import torch
from torchvision import transforms
from PIL import Image
import random
from models import AdvancedLungXRayModel

def plot_class_distribution(data_dir):
    """Plot class distribution in the dataset"""
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=plt.cm.tab10.colors[:len(class_counts)])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_accuracy_per_class(y_true, y_pred, classes):
    """Plot accuracy per class"""
    report = classification_report(y_true, y_pred, output_dict=True)
    class_accuracies = []
    for class_name in classes:
        if class_name in report:
            class_accuracies.append(report[class_name]['precision'])
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, class_accuracies)
    plt.title('Accuracy per Class')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('accuracy_per_class.png')
    plt.close()

def plot_sample_images(data_dir, num_samples=5):
    """Plot sample images from each class"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    fig, axes = plt.subplots(len(os.listdir(data_dir)), num_samples, figsize=(15, 3*len(os.listdir(data_dir))))
    if len(os.listdir(data_dir)) == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            sample_images = random.sample(images, min(num_samples, len(images)))
            
            for j, img_name in enumerate(sample_images):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                axes[i, j].imshow(img_np)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(class_name, rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def plot_training_history(history_file):
    """Plot training history"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_pie_chart(data_dir):
    """Plot pie chart of class distribution"""
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    
    plt.figure(figsize=(10, 10))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('Class Distribution (Pie Chart)')
    plt.axis('equal')
    plt.savefig('class_distribution_pie.png')
    plt.close()

def test_random_samples(data_dir, num_samples=5):
    """Test random samples from each class and plot results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedLungXRayModel(num_classes=9)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    sample_images = {}
    
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            selected_images = random.sample(images, min(num_samples, len(images)))
            sample_images[class_name] = selected_images
            correct = 0
            for img_name in selected_images:
                img_path = os.path.join(class_path, img_name)
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                    _, predicted = torch.max(outputs['output'], 1)
                    if predicted.item() == list(os.listdir(data_dir)).index(class_name):
                        correct += 1
            results[class_name] = correct / len(selected_images)
    
    # Plot test results
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results.keys(), results.values())
    plt.title('Random Sample Test Results')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('random_sample_test.png')
    plt.close()
    
    # Plot sample images used in testing
    fig, axes = plt.subplots(len(sample_images), num_samples, figsize=(15, 3*len(sample_images)))
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (class_name, images) in enumerate(sample_images.items()):
        for j, img_name in enumerate(images):
            img_path = os.path.join(data_dir, class_name, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            axes[i, j].imshow(img_np)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(class_name, rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig('random_sample_images.png')
    plt.close()

def main():
    # Plot class distribution
    plot_class_distribution('data/processed/train')
    
    # Plot pie chart
    plot_pie_chart('data/processed/train')
    
    # Plot sample images
    plot_sample_images('data/processed/train')
    
    # Test random samples
    test_random_samples('data/processed/test')
    
    # Load and plot training history
    if os.path.exists('training_history.json'):
        plot_training_history('training_history.json')
    
    # Load evaluation results if available
    if os.path.exists('evaluation_results.json'):
        with open('evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
        
        y_true = eval_results['true_labels']
        y_pred = eval_results['predictions']
        classes = eval_results['classes']
        
        plot_confusion_matrix(y_true, y_pred, classes)
        plot_accuracy_per_class(y_true, y_pred, classes)

if __name__ == '__main__':
    main() 