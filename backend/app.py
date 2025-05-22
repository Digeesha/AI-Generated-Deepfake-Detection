import sys
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import torchvision.transforms as transforms
from PIL import Image
import torch

# ✅ Add parent directory to sys.path so we can import 'model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import models and prediction
from model.predict import load_model, predict_video
from model.train import DeepFakeDetector  # for loading image model

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Load models (no training!)
model = load_model()  # for videos

image_model = DeepFakeDetector()
image_model.load_state_dict(torch.load("D:/deepfake-app/model/image_model.pth", map_location=torch.device("cpu")))
image_model.eval()

# 🎥 VIDEO prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video.save(temp.name)
        temp_path = temp.name

    result = predict_video(temp_path, model)
    os.remove(temp_path)

    return jsonify({"result": result})

# 🖼️ IMAGE prediction endpoint
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = image_model(input_tensor).item()  # ✅ Use image model here

    label = "REAL" if pred > 0.5 else "FAKE"
    return jsonify({
        "result": {
            "label": label,
            "score": pred
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
