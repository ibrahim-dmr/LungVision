import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

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

def main():
    # Create directories
    os.makedirs('data/processed/train', exist_ok=True)
    os.makedirs('data/processed/test', exist_ok=True)
    os.makedirs('data/processed/val', exist_ok=True)
    
    # Read metadata
    metadata_path = 'data/raw/covid19-pneumonia-normal-chest-xraypa-dataset/metadata.csv'
    df = pd.read_csv(metadata_path)
    
    # Split data into train, validation and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Process and save images
    for split, df_split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"Processing {split} set...")
        for idx, row in tqdm(df_split.iterrows(), total=len(df_split)):
            # Get image path and class
            img_path = os.path.join('data/raw/covid19-pneumonia-normal-chest-xraypa-dataset/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset', 
                                  row['directory'])
            img_class = row['class']
            
            # Create class directory if it doesn't exist
            class_dir = os.path.join('data/processed', split, str(img_class))
            os.makedirs(class_dir, exist_ok=True)
            
            # Process and save image
            try:
                img = preprocess_image(img_path)
                save_path = os.path.join(class_dir, os.path.basename(img_path))
                cv2.imwrite(save_path, (img * 255).astype(np.uint8))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 