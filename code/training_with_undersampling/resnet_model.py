import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
from collections import Counter
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold
from tqdm import tqdm

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),               # Resize to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
])

# Load dataset using ImageFolder
dataset_root = "./mri-images/T1_png_1mm"  # Root directory containing CN, MCI, AD subfolders
dataset = ImageFolder(root=dataset_root, transform=transform)

# Get class distribution
class_counts = Counter(dataset.targets)
print("Class distribution before undersampling:", class_counts)

# Find the minimum class count
min_class_count = min(class_counts.values())

# Perform random undersampling
indices_per_class = {cls: [] for cls in class_counts.keys()}
for idx, (_, label) in enumerate(dataset.samples):
    indices_per_class[label].append(idx)

undersampled_indices = []
for cls, indices in indices_per_class.items():
    undersampled_indices.extend(resample(indices, replace=False, n_samples=min_class_count, random_state=42))

# Create undersampled dataset
undersampled_subset = Subset(dataset, undersampled_indices)

# Verify class distribution after undersampling
undersampled_class_counts = Counter([dataset.targets[i] for i in undersampled_indices])
print("Class distribution after undersampling:", undersampled_class_counts)

# Define model (ResNet-18 with 3-channel input for grayscale images)
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18Classifier, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store accuracies for each fold
fold_train_accuracies = []
fold_val_accuracies = []

# Perform 5-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(kf.split(undersampled_indices)):
    print(f"\nFold {fold + 1}/5")

    # Create train and validation subsets for this fold
    train_subset = Subset(dataset, [undersampled_indices[i] for i in train_indices])
    val_subset = Subset(dataset, [undersampled_indices[i] for i in val_indices])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    # Initialize model, loss, and optimizer
    model = ResNet18Classifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop with validation
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Training phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(loss=running_loss / len(train_loader), acc=correct / total * 100)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(loss=val_loss / len(val_loader), acc=correct / total * 100)

        val_loss /= len(val_loader)
        val_accuracy = correct / total * 100

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Store the last epoch's accuracies for this fold
    fold_train_accuracies.append(train_accuracy)
    fold_val_accuracies.append(val_accuracy)

# Calculate average accuracies across all folds
avg_train_accuracy = np.mean(fold_train_accuracies)
avg_val_accuracy = np.mean(fold_val_accuracies)

print("\nCross-Validation Results:")
print(f"Average Training Accuracy: {avg_train_accuracy:.2f}%")
print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")