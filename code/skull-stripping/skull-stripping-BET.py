import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image

# Path to your dataset root directory
# dataset_dir = Path("./grouped-images-nii")
dataset_dir = Path("./mri-images/grouped-images-test-nii")


# Output directory for skull-stripped middle axial slices
output_dir = Path("./SkullStripped_MiddleAxial_PNG")
output_dir.mkdir(parents=True, exist_ok=True)

# Classes to process
classes = ["AD", "MCI", "CN"]

# BET parameters (adjust '-f' as needed; typical range: 0.3-0.5)
bet_fractional_intensity = 0.2

for cls in classes:
    input_class_dir = dataset_dir / cls
    output_class_dir = output_dir / cls
    output_class_dir.mkdir(exist_ok=True)

    nii_files = list(input_class_dir.glob("*.nii"))

    for nii_file in nii_files:
        subject_id = nii_file.stem  # e.g., "subject1"
        stripped_nii_file = input_class_dir / f"{subject_id}_brain.nii.gz"  # Temporary file path

        # BET command: skull stripping with mask generation (-m)
        bet_cmd = [
            "bet",
            str(nii_file),
            str(stripped_nii_file),
            "-f",
            str(bet_fractional_intensity),
            "-R"
        ]

        print(f"Processing {cls}/{subject_id}...")
        subprocess.run(bet_cmd, check=True)

        # Load the skull-stripped NIfTI file
        stripped_img = nib.load(str(stripped_nii_file))
        stripped_data = stripped_img.get_fdata()

        # Extract the middle axial slice (corrected indexing)
        mid_slice_idx = stripped_data.shape[0] // 2  # Middle slice index along axial plane
        middle_slice = stripped_data[mid_slice_idx, :, : ]  # Correct indexing for axial slices

        # Normalize the pixel values to [0, 255] for saving as PNG
        normalized_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice))
        normalized_slice = (normalized_slice * 255).astype(np.uint8)

        # Save the middle axial slice as a PNG file in the output directory
        png_output_path = output_class_dir / f"{subject_id}_middle_axial.png"
        Image.fromarray(normalized_slice).save(png_output_path)

        print(f"Saved middle axial slice for {cls}/{subject_id}.")

print("Skull stripping and middle axial slice extraction completed successfully.")
