import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
# Class label mapping
class_labels = {0: "AD", 1: "CN", 2: "MCI"}

# -------- 1. InceptionV3 Model Definition --------
# Define the model (InceptionV3 with 3-channel input for grayscale images)
class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV3Classifier, self).__init__()
        # Load InceptionV3 with updated pre-trained weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Disable auxiliary outputs for simplicity
        self.model.aux_logits = False
        self.model.AuxLogits = None
        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# -------- 2. Grad-CAM Utility Class --------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False)

        return cam.squeeze().cpu().detach().numpy()

# -------- 3. Helper Functions --------
def overlay_heatmap(image, heatmap):
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), image.size)

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Convert image to numpy array
    image_np = np.array(image)

    # If the image is grayscale (H, W), convert it to (H, W, 3)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)

    # Ensure types match
    image_np = image_np.astype(np.uint8)
    
    # Blend heatmap with the original image
    blended = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return blended

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

def visualize_explanation(image_path, model, gradcam, save_path=None):
    # Transformations: replicate grayscale channels to match InceptionV3's input requirements (3 channels)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.Resize((299, 299)),               # Resize to InceptionV3 input size
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    image = Image.open(image_path).convert("L")
    image_rgb = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    class_name = class_labels[class_idx]

    cam = gradcam.generate_cam(input_tensor, class_idx)
    heatmap_img = overlay_heatmap(image_rgb, cam)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Grad-CAM Overlay
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_img)
    plt.title(f"Grad-CAM Overlay\nPredicted: {class_name} (Class {class_idx})")
    plt.axis("off")

    # Colorbar-only heatmap
    plt.subplot(1, 3, 3)
    im = plt.imshow(cam, cmap='jet')
    plt.title("Heatmap Intensity")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        Image.fromarray(heatmap_img).save(save_path)

    plt.show()

    # Transformations: replicate grayscale channels to match InceptionV3's input requirements (3 channels)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.Resize((299, 299)),               # Resize to InceptionV3 input size
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    image = Image.open(image_path).convert("L")
    image_rgb = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    cam = gradcam.generate_cam(input_tensor, class_idx)
    heatmap_img = overlay_heatmap(image_rgb, cam)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_img)
    plt.title(f"Grad-CAM Heatmap (Class: {class_idx})")
    plt.axis("off")

    if save_path:
        Image.fromarray(heatmap_img).save(save_path)

    plt.tight_layout()
    plt.show()

# -------- 4. Load Model & Visualize One Image --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InceptionV3Classifier(num_classes=3).to(device)
model.load_state_dict(torch.load("../InceptionV3_model.pth"
                                 , map_location=device))

# Choose last convolutional block in InceptionV3
gradcam = GradCAM(model, model.model.Mixed_7c)

# Pick an example image (adjust path as needed)
sample_image_path = "./mri-images/Test Cases for T1_augmented_hflip/AD - aug_hflip_48_ADNI_067_S_0110_MR_MPR-R__GradWarp__B1_Correction_Br_20070730193829305_S27652_I63046_stripped.png"
visualize_explanation(sample_image_path, model, gradcam, save_path="inception_gradcam_sample_output.png")
