import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------
# CNN Model
# ------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, num_classes)  # num_classes is passed dynamically
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ------------------------------
# Training function
# ------------------------------
def train_model(quick_test=False):
    """
    quick_test=True -> train for 1 epoch (faster for pytest)
    quick_test=False -> full training
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Transforms and Data
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("dataset/labeled/train", transform=transform)
    val_data = datasets.ImageFolder("dataset/labeled/val", transform=transform)
    test_data = datasets.ImageFolder("dataset/labeled/test", transform=transform)

    num_classes = len(train_data.classes)  # auto-detect number of classes
    print("Classes:", train_data.classes)
    print("Number of classes:", num_classes)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=16)
    test_loader  = DataLoader(test_data, batch_size=16)

    # ------------------------------
    # Model, Loss, Optimizer
    # ------------------------------
    model = SmallCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # ------------------------------
    # Training loop
    # ------------------------------
    history = {"train_loss": [], "validation_loss": [], "validation_acc": []}
    epochs = 1 if quick_test else 3

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)

        history["validation_loss"].append(val_loss / len(val_loader))
        history["validation_acc"].append(correct / total)

    # ------------------------------
    # Test Accuracy
    # ------------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    return {
        "model": model,
        "test_acc": float(test_acc),
        "history": history,
        "loaders": (train_loader, val_loader, test_loader)
    }
