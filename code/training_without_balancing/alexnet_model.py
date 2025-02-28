import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import AlexNet_Weights
from collections import Counter
from tqdm import tqdm

# Define dataset class
class NiiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['AD', 'CN', 'MCI']
        self.image_paths = []
        self.labels = []

        # Load images and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        nii_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load NIfTI file
        nii_img = nib.load(nii_path).get_fdata()

        # Get middle axial slice
        mid_slice = nii_img[nii_img.shape[0] // 2, :, :]

        # Normalize image
        mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255.0
        mid_slice = mid_slice.astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(mid_slice)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
])

# Load full dataset
dataset = NiiDataset(root_dir="./grouped_images", transform=transform)

# First split: train (70%) and temp (30%)
train_indices, temp_indices = train_test_split(
    range(len(dataset)),
    test_size=0.3,
    stratify=dataset.labels
)

# Second split: temp into validation (10% of total) and test (20% of total)
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=2/3,  # 2/3 of 30% is 20% of the total dataset
    stratify=[dataset.labels[i] for i in temp_indices]
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Print class distribution in the full dataset
print("Full dataset class distribution:", Counter(dataset.labels))

# Print class distribution in the training set
train_labels = [dataset.labels[i] for i in train_indices]
print("Training set class distribution:", Counter(train_labels))

# Print class distribution in the validation set
val_labels = [dataset.labels[i] for i in val_indices]
print("Validation set class distribution:", Counter(val_labels))

# Print class distribution in the test set
test_labels = [dataset.labels[i] for i in test_indices]
print("Test set class distribution:", Counter(test_labels))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define model (AlexNet with single-channel input)
class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNetClassifier, self).__init__()
        # using updated weights
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # Adjust for single channel
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)  # Adjust output classes

    def forward(self, x):
        return self.model(x)

# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = AlexNetClassifier(num_classes=3).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    # Training phase
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
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

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total * 100

    # Validation phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total * 100

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

print("Training complete!")

# Evaluate on test set
model.eval()
test_loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = correct / total * 100

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")