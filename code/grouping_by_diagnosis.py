import os
import shutil

# Define the root directory where 'ad', 'cn', and 'mci' folders are located
root_directory = "grouped_images"  # Change this to your actual path

# Traverse each category folder (ad, cn, mci)
for category in ["ad", "cn", "mci"]:
    category_path = os.path.join(root_directory, category)

    if not os.path.exists(category_path):
        continue  # Skip if category folder doesn't exist

    # Walk through all subdirectories (bottom-up to get the deepest file first)
    for root, _, files in os.walk(category_path, topdown=False):
        for file in files:
            if file.endswith(".nii"):  # Only process .nii files
                source_path = os.path.join(root, file)
                destination_path = os.path.join(category_path, file)
                shutil.move(source_path, destination_path)  # Move file

# Count the number of .nii files in each category folder
file_counts = {category: len([f for f in os.listdir(os.path.join(root_directory, category)) if f.endswith(".nii")])
               for category in ["ad", "cn", "mci"]}

# Print final summary
print("\nSummary of Moved Files:")
for category, count in file_counts.items():
    print(f"{category.upper()}: {count} .nii files")

print("\nAll .nii files have been moved and counted!")
