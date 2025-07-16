
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform

def prepare_dataloaders(data_dir, batch_size=32, image_size=224, val_split=0.2):
    train_transform, val_transform = get_transforms(image_size)

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

def get_class_distribution(loader, class_names):
    class_counts = Counter()
    for _, labels in loader:
        for label in labels:
            class_counts[class_names[label.item()]] += 1
    return class_counts

def plot_class_distribution(class_counts):
    labels, counts = zip(*class_counts.items())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color='skyblue')
    plt.xticks(rotation=30)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

