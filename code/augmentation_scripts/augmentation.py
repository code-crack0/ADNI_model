import os
import shutil
import random
import nibabel as nib
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image

# Define paths
original_dir = "./grouped-images"
augmented_dir = "./augmented-images"
os.makedirs(augmented_dir, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    # maybe add rotate 180 degrees
    transforms.RandomHorizontalFlip(p=1.0),
    # too much try less values or remove it all together
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Get class distribution
class_counts = {cls: len(os.listdir(os.path.join(original_dir, cls))) for cls in os.listdir(original_dir)}
max_count = max(class_counts.values())

print("Class distribution before augmentation:", class_counts)

# Create directories for augmented images
for cls in class_counts:
    os.makedirs(os.path.join(augmented_dir, cls), exist_ok=True)

# Function to extract and save middle axial slice
def save_middle_slice(img_path, save_path):
    # Load 3D MRI scan
    mri_image = nib.load(img_path)
    mri_data = mri_image.get_fdata()
    
    # Extract middle axial slice
    mid_slice_idx = mri_data.shape[0] // 2
    mid_slice = mri_data[ mid_slice_idx,:,:]
    
    # Normalize to 0-255 for image processing
    mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255
    mid_slice = mid_slice.astype(np.uint8)
    
    image = Image.fromarray(mid_slice).convert("RGB")
    image.save(save_path)

# Process images
for cls, count in class_counts.items():
    class_dir = os.path.join(original_dir, cls)
    augmented_class_dir = os.path.join(augmented_dir, cls)
    images = os.listdir(class_dir)
    
    # Copy original images (middle slice only)
    for img in images:
        img_path = os.path.join(class_dir, img)
        save_path = os.path.join(augmented_class_dir, img.replace('.nii', '.png'))
        save_middle_slice(img_path, save_path)
    
    # Augment images to balance dataset
    needed = max_count - count
    while needed > 0:
        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)
        
        # Load and process middle slice
        mri_image = nib.load(img_path)
        mri_data = mri_image.get_fdata()
        mid_slice_idx = mri_data.shape[0] // 2
        mid_slice = mri_data[mid_slice_idx,:,:]
        
        # Normalize to 0-255
        mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255
        mid_slice = mid_slice.astype(np.uint8)
        
        image = Image.fromarray(mid_slice).convert("RGB")
        augmented_image = transform(image)
        
        # Save augmented image
        augmented_image_pil = transforms.ToPILImage()(augmented_image)
        new_name = f"aug_{needed}_{img_name.replace('.nii', '.png')}"
        augmented_image_pil.save(os.path.join(augmented_class_dir, new_name))
        needed -= 1

print("Augmentation complete. Balanced dataset created in", augmented_dir)