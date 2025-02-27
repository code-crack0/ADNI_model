import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
import os
from glob import glob

# Define class names and labels
class_labels = {"AD": 0, "CN": 1, "MCI": 2}  # Assign numeric labels

# Load Pretrained VGG16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_labels))  # Adjust output layer
model.to(device)
model.eval()  # Set to inference mode

# Transform for VGG Input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # VGG16 expects 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class NiiDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []

        # Iterate through AD, CN, MCI folders
        for class_name, label in class_labels.items():
            class_path = os.path.join(root_dir, class_name)
            nii_files = glob(os.path.join(class_path, "*.nii"))  # Change to "*.nii.gz" if needed

            self.file_paths.extend(nii_files)
            self.labels.extend([label] * len(nii_files))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nii_path = self.file_paths[idx]
        label = self.labels[idx]

        nii_img = nib.load(nii_path).get_fdata()
        mid_slice = nii_img[nii_img.shape[0] // 2,:, : ]  # Middle axial slice
        mid_slice = np.stack([mid_slice] * 3, axis=-1)  # Convert grayscale to 3-channel

        img_tensor = transform(mid_slice)
        return img_tensor, label

# Load Dataset
dataset_root = "../../grouped-images"
dataset = NiiDataset(dataset_root)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Inference and Accuracy Calculation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
