import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained VGG16 model with updated weights argument
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the number of classes (3)
model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=3)

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Transform for VGG16 Input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),               # Resize to 224x224 as required by VGG16
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for VGG16
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