"""
MNIST Inference Script
Run predictions on single images or the test set.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os

# reuse the model definition
from train import MNISTNet, DATA_DIR

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/best_model.pth"

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_model():
    model = MNISTNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict_image(image_path: str) -> int:
    """Predict digit in a single image file."""
    model = load_model()
    img   = Image.open(image_path).convert("L")
    x     = TRANSFORM(img).unsqueeze(0).to(DEVICE)   # add batch dim
    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(1).item()
    print(f"Predicted digit: {pred}")
    return pred


def evaluate_test_set():
    """Run the model on the full MNIST test set and print accuracy."""
    model  = load_model()
    ds     = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds   = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()

    acc = correct / len(ds)
    print(f"Test-set accuracy: {acc:.4f}  ({correct}/{len(ds)})")
    return acc


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        predict_image(sys.argv[1])
    else:
        print("No image path given — evaluating full test set …")
        evaluate_test_set()
