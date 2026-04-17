"""
MNIST Handwritten Digit Classifier
Training script using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json

#Configuration
BATCH_SIZE   = 64
EPOCHS       = 10
LEARNING_RATE = 0.001
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = "./data"
MODEL_DIR    = "./models"

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#model
class MNISTNet(nn.Module):
    """Simple CNN for MNIST digit classification."""

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),                              
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


#data
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),#MNIST mean & std
    ])

    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


#train, eval
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += output.argmax(1).eq(labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            total_loss += criterion(output, labels).item() * images.size(0)
            correct    += output.argmax(1).eq(labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


#main
def main():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_dataloaders()
    model     = MNISTNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} │ "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} │ "
              f"Test  Loss: {te_loss:.4f}  Acc: {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

    # Save final model + history
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pth"))
    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n Training complete! Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
