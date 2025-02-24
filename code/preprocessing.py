import os
import shutil

# Define the source directory (where the images are nested)
source_directory = "../T1_model"  # Change this to your actual root folder

# Define the destination directory (where all images will be copied)
destination_directory = "../consolidated-images"  # Change this to your desired folder

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Supported image file extensions
image_extensions = {".nii"}  # Add more if needed

# Counter for total images copied
total_images_copied = 0

# Walk through all subdirectories and copy images
for root, _, files in os.walk(source_directory):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_directory, file)
            
            # Handle duplicate file names by appending a number
            base, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(destination_directory, f"{base}_{counter}{ext}")
                counter += 1
            
            shutil.move(source_path, destination_path)
            total_images_copied += 1
            print(f"Copied: {source_path} -> {destination_path}")

print(f"All images have been consolidated! Total images copied: {total_images_copied}")
