import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import classification_report

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained EfficientNet-B3 model with updated weights argument
model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the number of classes (3)
model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=3)

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Transform for EfficientNet-B3 Input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((300, 300)),               # Resize to 300x300 as required by EfficientNet-B3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for EfficientNet
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