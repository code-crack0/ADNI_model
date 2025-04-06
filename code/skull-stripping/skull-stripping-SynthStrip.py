import os
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
import subprocess

# Paths to dataset and output directories
dataset_dir = Path("./mri-images/grouped-images-test-nii")  # Input directory containing AD, CN, MCI subfolders
output_dir = Path("./SkullStripped_MiddleAxial_PNG_SS_delete")  # Output directory for PNGs
output_dir.mkdir(parents=True, exist_ok=True)

# Classes to process (subfolders in dataset_dir)
classes = ["AD", "MCI", "CN"]

# Path to the SynthStrip model
synthstrip_model_path = "/mnt/c/Users/sbm76/Downloads/synthstrip.1.pt" 

# Function to run SynthStrip on a NIfTI file
def run_synthstrip(input_file, output_file):
    """
    Run SynthStrip on a NIfTI file to perform skull stripping.
    
    Parameters:
    - input_file (str): Path to input NIfTI file.
    - output_file (str): Path to save the skull-stripped NIfTI file.
    """
    cmd = [
        "nipreps-synthstrip",  # Command-line tool for SynthStrip (ensure it's installed)
        "-i", str(input_file),
        "-o", str(output_file),
        "--model", synthstrip_model_path
        # "--gpu"
    ]
    print(f"Running SynthStrip on {input_file}...")
    subprocess.run(cmd, check=True)
    print(f"Saved skull-stripped file to {output_file}")

# Function to extract the middle axial slice and save as PNG
def extract_middle_axial_slice(input_file, output_png_path):
    """
    Extract the middle axial slice from a skull-stripped NIfTI file and save it as a PNG.
    
    Parameters:
    - input_file (str): Path to the skull-stripped NIfTI file.
    - output_png_path (str): Path to save the middle axial slice as a PNG.
    """
    # Load the NIfTI file
    nii_img = nib.load(input_file)
    brain_data = nii_img.get_fdata()

    # Extract the middle axial slice (assuming z-axis is the first dimension)
    mid_slice_idx = brain_data.shape[0] // 2  # Middle slice index along z-axis
    middle_slice = brain_data[mid_slice_idx, :, :]

    # Normalize pixel values to [0, 255] for saving as PNG
    normalized_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice))
    normalized_slice = (normalized_slice * 255).astype(np.uint8)

    # Save as PNG using PIL
    Image.fromarray(normalized_slice).save(output_png_path)
    print(f"Saved middle axial slice as PNG: {output_png_path}")

# Process each class folder (AD, MCI, CN)
for cls in classes:
    input_class_dir = dataset_dir / cls  # Input directory for current class
    output_class_dir = output_dir / cls  # Output directory for current class
    output_class_dir.mkdir(exist_ok=True)

    nii_files = list(input_class_dir.glob("*.nii"))  # List of all .nii files in the class folder

    for nii_file in nii_files:
        subject_id = nii_file.stem  # Extract subject ID from filename

        # Paths for intermediate skull-stripped file and final PNG output
        stripped_nii_path = input_class_dir / f"{subject_id}_stripped.nii.gz"
        png_output_path = output_class_dir / f"{subject_id}_middle_axial.png"

        # Run SynthStrip to perform skull stripping
        run_synthstrip(nii_file, stripped_nii_path)

        # Extract middle axial slice and save as PNG
        extract_middle_axial_slice(stripped_nii_path, png_output_path)

print("Processing completed successfully!")
