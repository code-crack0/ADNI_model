import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
import os
from glob import glob
from torchvision.datasets import ImageFolder
# Load the pre-trained AlexNet model with updated weights argument
from torchvision.models import AlexNet_Weights
from sklearn.metrics import classification_report

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the number of classes (3)
model.classifier[6] = nn.Linear(in_features=4096, out_features=3)

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Transform for AlexNet Input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset using ImageFolder
dataset_root = "./mri-images/T1_png_1mm"
dataset = ImageFolder(root=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Inference and Accuracy Calculation
correct = 0
total = 0

# Initialize lists to store true labels and predictions
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Append true labels and predictions to the lists
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
print()

# Generate classification report
report = classification_report(all_labels, all_predictions, target_names=dataset.classes)
print("Classification Report:")
print(report)