import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from collections import Counter

# Define dataset class
class NiiDataset(Dataset):
    def __init__(self, root_dir, transform=None, minority_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.minority_transform = minority_transform
        self.classes = ['AD', 'CN', 'MCI']
        self.image_paths = []
        self.labels = []

        # Count occurrences of each class
        self.class_counts = {c: 0 for c in self.classes}

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(label)
                    self.class_counts[class_name] += 1

        # Determine minority classes
        self.minority_classes = [c for c, count in self.class_counts.items() if count < max(self.class_counts.values())]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        nii_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load NIfTI file
        nii_img = nib.load(nii_path)
        img_data = nii_img.get_fdata()

        # Get middle axial slice
        mid_slice = img_data[img_data.shape[0] // 2, :, :]

        # Normalize image
        mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255.0
        mid_slice = mid_slice.astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(mid_slice)

        # Apply different transformations based on class
        if self.minority_transform and self.classes[label] in self.minority_classes:
            img = self.minority_transform(img)
        elif self.transform:
            img = self.transform(img)

        return img, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Stronger augmentations for minority class
minority_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = NiiDataset(root_dir="../../grouped-images", transform=train_transform, minority_transform=minority_transform)

# Stratified split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(np.zeros(len(dataset.labels)), dataset.labels))

train_subset = Subset(dataset, train_idx)
val_subset = Subset(dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

# Define model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),  # Increased dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetClassifier(num_classes=3).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)  # Increased weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total * 100
    scheduler.step(val_loss)  # Step scheduler based on validation loss

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

print("Training complete!")
