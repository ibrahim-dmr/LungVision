import os
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Download dataset
    print("Downloading Montgomery County X-ray Dataset...")
    url = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
    zip_path = "data/raw/montgomery_xray.zip"
    
    try:
        download_file(url, zip_path)
        print("Dataset downloaded successfully!")
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw")
        
        # Clean up
        os.remove(zip_path)
        print("Dataset extracted successfully!")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Please download the dataset manually from:")
        print("https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets")
        return

if __name__ == "__main__":
    main() 