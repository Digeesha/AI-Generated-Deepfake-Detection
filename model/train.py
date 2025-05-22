# model/train_video.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision import models, transforms

# ✅ Add project root to sys.path so we can import from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import extract_faces

# 🧠 Model: ResNet18 fine-tuned
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 🎥 Video dataset (faces extracted from videos)
class VideoFaceDataset(Dataset):
    def __init__(self, video_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for label, category in enumerate(['real', 'fake']):
            folder = os.path.join(video_dir, category)
            for filename in os.listdir(folder):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(folder, filename), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        faces = extract_faces(video_path)

        # Skip videos with no faces detected
        if not faces:
            return []

        processed_faces = []
        for face in faces:
            if isinstance(face, torch.Tensor):
                face = transforms.functional.resize(face, [224, 224])
                face = self.transform(face)
            else:
                raise TypeError(f"Unexpected face format: {type(face)}")

            processed_faces.append((face, torch.tensor([label], dtype=torch.float32)))

        return processed_faces

# 🚀 Training function
def train_model():
    dataset_path = "D:/deepfake-app/dataset/train"
    raw_dataset = VideoFaceDataset(dataset_path)

    # Flatten dataset (faces only)
    flattened_data = [sample for group in raw_dataset if group for sample in group]
    print(f"Total samples extracted: {len(flattened_data)}")

    loader = DataLoader(flattened_data, batch_size=8, shuffle=True)

    model = DeepFakeDetector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_preds = []
    all_labels = []

    print("🚀 Starting training...")
    for epoch in range(10):
        model.train()
        running_corrects = 0
        total = 0

        for faces, labels in loader:
            faces, labels = faces.to(device), labels.to(device)

            preds = model(faces)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (preds > 0.5).float()
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        epoch_acc = 100 * running_corrects / total
        print(f"📚 Epoch {epoch+1}/10 - Loss: {loss.item():.4f} - Accuracy: {epoch_acc:.2f}%")

    # Save model
    save_path = "D:/deepfake-app/model/video_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    train_model()
