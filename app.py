import torch
from flask import Flask, request, jsonify
from PIL import Image
from torch import nn, optim
from torchvision import  transforms, models


classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 102),
    nn.LogSoftmax(dim=1)
)
model = models.resnet18(pretrained=True)
# Charger le modèle sauvegardé
checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))

# Rebuild the model architecture
model.fc = classifier
model.load_state_dict(checkpoint['model_state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Recréer l'optimiseur en fonction des paramètres actuels du modèle
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Définir le modèle en mode évaluation
model.eval()


# Initialiser Flask
app = Flask(__name__)


# Définir les transformations d'image (en fonction de l'entraînement du modèle)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Fonction pour prédire à partir de l'image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # Ouvrir l'image
        img = Image.open(file.stream)
        
        # Appliquer les transformations sur l'image
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Ajouter la dimension du batch
        # Effectuer la prédiction
        with torch.no_grad():  # Désactiver le calcul des gradients
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            idx_to_class = {v: k for k, v in model.class_to_idx.items()}
            predicted_class = cat_to_name[idx_to_class[predicted_idx.item()]]
        # Retourner la classe prédite
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(port=5000)
