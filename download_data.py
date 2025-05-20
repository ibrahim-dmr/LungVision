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
    print("Downloading COVID-19 Chest X-ray Dataset...")
    url = "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
    zip_path = "data/raw/covid19-chest-xray.zip"
    
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
        print("https://data.mendeley.com/datasets/rscbjbr9sj/1")
        return

if __name__ == "__main__":
    main() 