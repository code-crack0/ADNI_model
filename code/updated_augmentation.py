import os
import shutil
import random
import nibabel as nib
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image

# Define paths
# grouped-images is the original path of all the nii files
original_dir = "./grouped-images"
# new augmented images will be saved in this directory
augmented_dir = "./augmented-images-v2"
os.makedirs(augmented_dir, exist_ok=True)

# Define transformations
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
vertical_flip = transforms.RandomVerticalFlip(p=1.0)

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
    mid_slice = mri_data[mid_slice_idx,:,:]
    
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
    
    # Track which images have been used for augmentation and which augmentation was applied
    augmentation_tracking = {}  # Format: {image_name: [horizontal_used, vertical_used]}
    
    while needed > 0:
        # Initialize the tracking dict for all images if not already done
        if not augmentation_tracking:
            for img in images:
                augmentation_tracking[img] = [False, False]  # [horizontal_used, vertical_used]
        
        # Find images that haven't had both augmentations applied
        available_images = [img for img, status in augmentation_tracking.items() 
                           if not (status[0] and status[1])]
        
        # If all images have had both augmentations, reset and start over
        if not available_images:
            for img in images:
                augmentation_tracking[img] = [False, False]
            available_images = images
        
        # Select a random image from available ones
        img_name = random.choice(available_images)
        
        # Determine which augmentation to apply
        if not augmentation_tracking[img_name][0] and not augmentation_tracking[img_name][1]:
            # Neither augmentation has been applied, choose randomly
            use_horizontal = random.choice([True, False])
        elif not augmentation_tracking[img_name][0]:
            # Only vertical has been applied, use horizontal
            use_horizontal = True
        elif not augmentation_tracking[img_name][1]:
            # Only horizontal has been applied, use vertical
            use_horizontal = False
        
        # Update tracking
        if use_horizontal:
            augmentation_tracking[img_name][0] = True
        else:
            augmentation_tracking[img_name][1] = True
        
        # Load and process middle slice
        img_path = os.path.join(class_dir, img_name)
        mri_image = nib.load(img_path)
        mri_data = mri_image.get_fdata()
        mid_slice_idx = mri_data.shape[0] // 2
        mid_slice = mri_data[mid_slice_idx,:,:]
        
        # Normalize to 0-255
        mid_slice = (mid_slice - np.min(mid_slice)) / (np.max(mid_slice) - np.min(mid_slice)) * 255
        mid_slice = mid_slice.astype(np.uint8)
        
        image = Image.fromarray(mid_slice).convert("RGB")
        
        # Apply the selected augmentation
        if use_horizontal:
            augmented_image = horizontal_flip(image)
            aug_type = "hflip"
        else:
            augmented_image = vertical_flip(image)
            aug_type = "vflip"
        
        # Save augmented image
        new_name = f"aug_{aug_type}_{needed}_{img_name.replace('.nii', '.png')}"
        augmented_image.save(os.path.join(augmented_class_dir, new_name))
        needed -= 1

print("Augmentation complete. Balanced dataset created in", augmented_dir)
