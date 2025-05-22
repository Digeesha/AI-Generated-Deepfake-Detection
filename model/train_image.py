# model/train_image.py

import os
import sys
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ✅ Add project root to sys.path so we can import from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def train_image_model():
    dataset_path = "D:/deepfake-app/dataset/image_train"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])

    dataset = ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DeepFakeDetector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_preds = []
    all_labels = []

    print("🚀 Starting training on image dataset...")
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            epoch_loss += loss.item()

            # Save for ROC
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = epoch_loss / len(loader)
        acc = 100 * correct / total
        print(f"📚 Epoch {epoch+1}/10 - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

    # Save model
    save_path = "D:/deepfake-app/model/image_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Image model saved to {save_path}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # random line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# ✅ Only train if run directly
if __name__ == "__main__":
    train_image_model()
