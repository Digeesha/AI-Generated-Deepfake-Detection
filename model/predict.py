import torch
from model.train import DeepFakeDetector
from model.utils import extract_faces

def load_model(model_path="D:/deepfake-app/model/video_model.pth"):
    model = DeepFakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully.")
    return model

def predict_video(video_path, model):
    faces = extract_faces(video_path)
    preds = []
    for face in faces:
        face = face.unsqueeze(0)
        with torch.no_grad():
            pred = model(face)
            preds.append(pred.item())

    avg = sum(preds) / len(preds) if preds else 0
    label = "FAKE" if avg > 0.5 else "REAL"
    return {"label": label, "score": avg}

