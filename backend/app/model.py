import torch
import torch.nn as nn

MODEL_PATH = "models/best_deepfake_model_full_dataset.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2, image_size=160):
        super(DeepfakeDetectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        self.flattened_size = 128 * (image_size // 16) ** 2
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model once
model = DeepfakeDetectorCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"[WARNING] Failed to load model: {e}")
