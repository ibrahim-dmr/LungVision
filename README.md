# Lung X-Ray Classification Project

This project aims to classify lung X-ray images into three categories: COVID-19, Pneumonia, and Normal using deep learning techniques.

## Project Structure

```
lung_X-Ray_Net/
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Processed dataset
├── src/                      # Source code
├── models/                   # Trained models
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lung_X-Ray_Net.git
cd lung_X-Ray_Net
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and preprocess the dataset:
```bash
python download_data.py
python preprocess_data.py
```

## Dataset

The dataset contains chest X-ray images classified into three categories:
- COVID-19
- Pneumonia
- Normal

## Model Architecture

The project uses a pre-trained Swin Transformer model (Swin-T) for image classification.

## Training

To train the model:
```bash
python train.py
```

## Evaluation

To evaluate the model:
```bash
python evaluate.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 