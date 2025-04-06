import os
import random
import shutil

def select_and_copy_images(src_dir, dest_dir, category, num_images):
    # Paths for source and destination directories
    source_path = os.path.join(src_dir, category)
    destination_path = os.path.join(dest_dir, category)

    # Create destination directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # List all PNG files in the source directory
    images = [f for f in os.listdir(source_path) if f.lower().endswith('.png')]

    # Check if there are enough images to select from
    if len(images) < num_images:
        raise ValueError(f"Not enough images in {source_path}. Found {len(images)} images.")

    # Randomly select the specified number of images without replacement
    selected_images = random.sample(images, num_images)

    # Copy selected images to the destination directory
    for img in selected_images:
        shutil.copy(os.path.join(source_path, img), os.path.join(destination_path, img))

if __name__ == "__main__":
    # Define your directories here:
    SRC_DIR = "./png_output_1mm"
    DEST_DIR = "MCI-images"

    # Number of images to select from each category
    NUM_IMAGES = 1000

    # Categories to process (only CN and MCI as per your request)
    # categories_to_process = ["CN", "MCI"]
    select_and_copy_images(SRC_DIR, DEST_DIR, "MCI", NUM_IMAGES)

print("All done!")
