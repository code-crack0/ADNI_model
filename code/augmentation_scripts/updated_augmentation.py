import os
import shutil
import random
import nibabel as nib
import numpy as np
from skimage import exposure
from collections import Counter
from torchvision import transforms
from PIL import Image

# Define paths
original_dir = "./flirt_registered_T1_1mm_parallel"
augmented_dir = "./T1_augmented_median_axial_slice"
os.makedirs(augmented_dir, exist_ok=True)

# Define transformations with MRI-appropriate parameters
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
vertical_flip = transforms.RandomVerticalFlip(p=1.0)

# Get class distribution
class_counts = {cls: len(os.listdir(os.path.join(original_dir, cls))) for cls in os.listdir(original_dir)}
max_count = max(class_counts.values())

print("Class distribution before augmentation:", class_counts)

# Create directories for augmented images
for cls in class_counts:
    os.makedirs(os.path.join(augmented_dir, cls), exist_ok=True)

def normalize_slice(slice_data):
    """MRI-specific normalization from previous implementation"""
    p1, p99 = np.percentile(slice_data, (1, 99))
    clipped = np.clip(slice_data, p1, p99)
    normalized = exposure.rescale_intensity(clipped, out_range=(0, 255))
    return normalized.astype(np.uint8)

def process_slice(mri_data, rotation=90):
    """Centralized slice processing with MRI-specific handling"""
    # Extract middle axial slice (corrected axis)
    mid_slice_idx = mri_data.shape[2] // 2  # Axial is typically 3rd dimension
    slice_data = mri_data[:, :, mid_slice_idx]
    
    # Apply radiological orientation rotation
    rotated = np.rot90(slice_data, k=rotation//90)
    
    # Normalize using MRI-specific method
    return normalize_slice(rotated)

# Process images with enhanced normalization
def save_middle_slice(img_path, save_path):
    mri_image = nib.load(img_path)
    mri_data = mri_image.get_fdata()
    
    processed_slice = process_slice(mri_data)
    image = Image.fromarray(processed_slice).convert("L")  # Grayscale for MRI
    
    # Save as PNG with original filename
    image.save(save_path)

# Process original images
for cls, count in class_counts.items():
    class_dir = os.path.join(original_dir, cls)
    augmented_class_dir = os.path.join(augmented_dir, cls)
    images = os.listdir(class_dir)
    
    for img in images:
        img_path = os.path.join(class_dir, img)
        save_path = os.path.join(augmented_class_dir, img.replace('.nii.gz', '.png'))
        save_middle_slice(img_path, save_path)

# Augmentation with consistent preprocessing
for cls, count in class_counts.items():
    class_dir = os.path.join(original_dir, cls)
    augmented_class_dir = os.path.join(augmented_dir, cls)
    images = os.listdir(class_dir)
    needed = max_count - count
    augmentation_tracking = {}

    while needed > 0:
        if not augmentation_tracking:
            for img in images:
                augmentation_tracking[img] = [False, False]

        available_images = [img for img, status in augmentation_tracking.items() 
                          if not (status[0] and status[1])]
        
        if not available_images:
            for img in images:
                augmentation_tracking[img] = [False, False]
            available_images = images

        img_name = random.choice(available_images)
        
        # Determine augmentation type
        if not augmentation_tracking[img_name][0] and not augmentation_tracking[img_name][1]:
            use_horizontal = random.choice([True, False])
        elif not augmentation_tracking[img_name][0]:
            use_horizontal = True
        else:
            use_horizontal = False

        # Update tracking
        augmentation_tracking[img_name][0 if use_horizontal else 1] = True

        # Load and process with MRI pipeline
        img_path = os.path.join(class_dir, img_name)
        mri_image = nib.load(img_path)
        mri_data = mri_image.get_fdata()
        
        processed_slice = process_slice(mri_data)
        image = Image.fromarray(processed_slice).convert("L")

        # Apply augmentation
        if use_horizontal:
            augmented_image = horizontal_flip(image)
            aug_type = "hflip"
        else:
            augmented_image = vertical_flip(image)
            aug_type = "vflip"

        # Save augmented image
        new_name = f"aug_{aug_type}_{img_name.replace('.nii.gz', '')}_{needed:04d}.png"
        augmented_image.save(os.path.join(augmented_class_dir, new_name))
        needed -= 1

print(f"Augmentation complete. Balanced dataset created in {augmented_dir}")