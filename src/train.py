import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.utils import (
    set_seed, get_device, prepare_dataloaders,
    get_class_distribution, plot_class_distribution
)
from src.model import SimpleCNN

# 1. Configs
set_seed(42)
device = get_device()

DATA_DIR = "data"
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = 224
LR = 0.001
SAVE_PATH = "model.pth"

train_loader, val_loader, class_names = prepare_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
print(f"Classes: {class_names}")

class_counts = get_class_distribution(train_loader, class_names)
plot_class_distribution(class_counts)

model = SimpleCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = train_loss / total
    accuracy = 100. * correct / total
    print(f"\n✅ Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

model.eval()
val_loss, val_correct, val_total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)

val_accuracy = 100. * val_correct / val_total
val_loss /= val_total
print(f"\n🧪 Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

torch.save(model.state_dict(), SAVE_PATH)
print(f"\n💾 Model saved to {SAVE_PATH}")

