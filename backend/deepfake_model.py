import torch
from model_arch import DeepFakeDetector
from video_utils import extract_faces_from_video

def load_model(model_path):
    model = DeepFakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_on_video(video_path, model):
    faces = extract_faces_from_video(video_path)
    predictions = []
    for face in faces:
        face = face.unsqueeze(0)  # Add batch dimension
        pred = model(face)
        predictions.append(pred.item())
    avg_pred = sum(predictions) / len(predictions)
    return "FAKE" if avg_pred > 0.5 else "REAL"
