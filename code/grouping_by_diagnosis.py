import os
import shutil
import pandas as pd

# Define paths
source_directory = "../consolidated-images"  # Change this to your actual root folder
destination_directory = "../grouped-images"  # Change this to your desired folder
csv_file_path = "../metadata.csv"  # Change to your actual CSV file path

# Ensure the destination directories exist
group_folders = {"AD": os.path.join(destination_directory, "AD"),
                 "CN": os.path.join(destination_directory, "CN"),
                 "MCI": os.path.join(destination_directory, "MCI")}
for folder in group_folders.values():
    os.makedirs(folder, exist_ok=True)

# Read CSV and create a mapping of image IDs to groups
df = pd.read_csv(csv_file_path)
image_group_map = dict(zip(df["Image Data ID"].astype(str), df["Group"].astype(str)))

print(image_group_map)
# Supported image file extensions
image_extensions = {".nii"}  # Add more if needed

total_images_copied = {"AD": 0, "CN": 0, "MCI": 0}
# Walk through all subdirectories and copy images
for root, _, files in os.walk(source_directory):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # Extract the image ID (everything after the last underscore)
            image_id = file.rsplit("_", 1)[-1].split(".")[0]  # Extract the numeric ID after last underscore
            
            # Check if the image ID is in the CSV mapping
            if image_id in image_group_map:
                group = image_group_map[image_id]
                if group in group_folders:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(group_folders[group], file)
                    
                    # Handle duplicate file names by appending a number
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_path):
                        destination_path = os.path.join(group_folders[group], f"{base}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(source_path, destination_path)
                    total_images_copied[group] += 1
                    print(f"Copied: {source_path} -> {destination_path}")

print(f"Images copied: AD={total_images_copied['AD']}, CN={total_images_copied['CN']}, MCI={total_images_copied['MCI']}")
