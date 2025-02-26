import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

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

# Stratified split into train (80%) and validation (20%)
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=dataset.labels
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define model (ResNet18 with single-channel input)
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetClassifier, self).__init__()
        # self.model = models.resnet18(pretrained=True)
        # using updated weights
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single channel
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output classes

    def forward(self, x):
        return self.model(x)

# Model setup
device="cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetClassifier(num_classes=3).to(device)

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

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

print("Training complete!")
