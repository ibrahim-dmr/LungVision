from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np
from models import AdvancedLungXRayModel

app = Flask(__name__)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedLungXRayModel(num_classes=9)  # Updated to 9 classes
model.load('models/best_model.pth')
model = model.to(device)
model.eval()

# Class names
class_names = [
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

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    # Process image
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs['output'], dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        # Calculate uncertainty
        uncertainty = outputs['uncertainty'].item()
    
    # Class name translations
    class_translations = {
        '00_Normal': 'Normal',
        '01_Pneumonia': 'Zatürre',
        '02_High_Density': 'Yüksek Yoğunluk',
        '03_Low_Density': 'Düşük Yoğunluk',
        '04_Obstructive_Diseases': 'Tıkayıcı Hastalıklar',
        '05_Infectious_Diseases': 'Enfeksiyon Hastalıkları',
        '06_Encapsulated_Lesions': 'Kapsüllü Lezyonlar',
        '07_Mediastinal_Changes': 'Mediastinal Değişiklikler',
        '08_Chest_Changes': 'Göğüs Değişiklikleri'
    }
    
    # Get translated class name
    display_class_name = class_translations[class_names[predicted_class]]
    
    return {
        'class': display_class_name,
        'confidence': float(confidence),
        'uncertainty': float(uncertainty),
        'probabilities': {
            class_translations[class_name]: float(prob)
            for class_name, prob in zip(class_names, probabilities)
        }
    }

@app.route('/')
def home():
    return render_template('index.html', classes=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load image and make prediction
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 