import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CNN Model
class SmallCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(64 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("data/train", transform=transform)
    val_data = datasets.ImageFolder("data/validation", transform=transform)
    test_data = datasets.ImageFolder("data/test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    model = SmallCNN().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_acc": []
    }

    epochs = 3

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        history["train_loss"].append(train_loss)

    # Simple test accuracy

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = (outputs > 0.5).float().cpu()

            correct += (preds.squeeze() == labels).sum().item()

            total += labels.size(0)

    test_acc = correct / total

    return {
        "model": model,
        "test_acc": float(test_acc),
        "history": history,
        "loaders": (train_loader, val_loader, test_loader)
    }
