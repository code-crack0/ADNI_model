import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained InceptionV3 model with updated weights argument
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the number of classes (3)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Transform for InceptionV3 Input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((299, 299)),               # Resize to 299x299 as required by InceptionV3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for InceptionV3
])

# Load Dataset using ImageFolder
dataset_root = "./mri-images/T1_png_1mm"
dataset = ImageFolder(root=dataset_root, transform=transform)
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