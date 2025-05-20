import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    img = cv2.imread(image_path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize
    img = cv2.resize(img, target_size)
    # Normalize
    img = img / 255.0
    return img

def create_directory_structure():
    # Create necessary directories for each class
    classes = [
        '00_Normal',
        '01_Pneumonia',
        '02_High_Density',
        '03_Low_Density',
        '04_Obstructive_Diseases',
        '05_Infectious_Diseases',
        '06_Encapsulated_Lesions',
        '07_Mediastinal_Changes',
        '08_Chest_Changes'
    ]
    
    # Remove existing directories if they exist
    if os.path.exists('data/processed'):
        shutil.rmtree('data/processed')
    
    # Create new directories
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(f'data/processed/{split}/{class_name}', exist_ok=True)

def organize_dataset():
    # Path to the archive dataset
    raw_data_path = 'archive (7)'
    
    # Map of directory names to class names
    class_mapping = {
        '00 Anatomia Normal': '00_Normal',
        '01 Processos Inflamatórios Pulmonares (Pneumonia)': '01_Pneumonia',
        '02 Maior Densidade (Derrame Pleural, Consolidação Atelectasica, Hidrotorax, Empiema)': '02_High_Density',
        '03 Menor Densidade (Pneumotorax, Pneumomediastino, Pneumoperitonio)': '03_Low_Density',
        '04 Doenças Pulmonares Obstrutivas (Enfisema, Broncopneumonia, Bronquiectasia, Embolia)': '04_Obstructive_Diseases',
        '05 Doenças Infecciosas Degenerativas (Tuberculose, Sarcoidose, Proteinose, Fibrose)': '05_Infectious_Diseases',
        '06 Lesões Encapsuladas (Abscessos, Nódulos, Cistos, Massas Tumorais, Metastases)': '06_Encapsulated_Lesions',
        '07 Alterações de Mediastino (Pericardite, Malformações Arteriovenosas, Linfonodomegalias)': '07_Mediastinal_Changes',
        '08 Alterações do Tórax (Atelectasias, Malformações, Agenesia, Hipoplasias)': '08_Chest_Changes'
    }
    
    # Process each class
    for source_dir, target_class in class_mapping.items():
        source_path = os.path.join(raw_data_path, source_dir)
        if not os.path.exists(source_path):
            print(f"Warning: Directory not found: {source_path}")
            continue
            
        # Get all image files
        image_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"Warning: No images found in {source_path}")
            continue
            
        print(f"Processing {target_class}: {len(image_files)} images found")
        
        # Split data into train, validation, and test sets
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Copy files to their respective directories
        def copy_files(files, target_split):
            for file in tqdm(files, desc=f"Copying {target_class} to {target_split}"):
                filename = os.path.basename(file)
                dst = os.path.join('data/processed', target_split, target_class, filename)
                shutil.copy2(file, dst)
        
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def preprocess_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Preprocess the dataset and split it into train, validation, and test sets.
    
    Args:
        source_dir (str): Directory containing the original dataset
        target_dir (str): Directory where the processed dataset will be saved
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
    """
    # Create target directories
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_dir in class_dirs:
        # Create class directories in train, val, and test
        os.makedirs(os.path.join(target_dir, 'train', class_dir), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', class_dir), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test', class_dir), exist_ok=True)
        
        # Get all images in the class directory
        images = [f for f in os.listdir(os.path.join(source_dir, class_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(source_dir, class_dir, img)
            dst = os.path.join(target_dir, 'train', class_dir, img)
            shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(source_dir, class_dir, img)
            dst = os.path.join(target_dir, 'val', class_dir, img)
            shutil.copy2(src, dst)
        
        for img in test_images:
            src = os.path.join(source_dir, class_dir, img)
            dst = os.path.join(target_dir, 'test', class_dir, img)
            shutil.copy2(src, dst)

def main():
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Organizing dataset...")
    organize_dataset()
    
    print("Dataset organization completed successfully!")

if __name__ == "__main__":
    main() 