"""
FLIRT Registration Script for Skull-Stripped NIfTI Files
Processes: input_dir/AD|CN|MCI/*.nii -> output_dir/AD|CN|MCI/*.nii.gz
"""

import subprocess
from pathlib import Path

# Configuration

#  # Input directory containing AD, CN, MCI subfolders
# INPUT_DIR = Path("/mnt/c/Users/sbm76/Documents/Projects/senior-design-machine-learning/ADNI")
# # Output directory for registered files
# OUTPUT_DIR = Path("/mnt/c/Users/sbm76/Documents/Projects/senior-design-machine-learning/flirt_registered")
# FSL_TEMPLATE = Path("/home/saeed/fsl/data/standard/MNI152_T1_2mm_brain")

# Configuration
INPUT_DIR = Path("./ADNI")  # Input directory containing AD, CN, MCI subfolders
OUTPUT_DIR = Path("./flirt_registered_T1_1mm") # Output directory for registered files

FSL_TEMPLATE = Path("/home/saeed/fsl/data/standard/MNI152_T1_1mm_brain")  # Path to MNI template in WSL
CLASSES = ["AD", "CN", "MCI"]  # Subfolders to process

def process_file(input_path, output_path):
    """Register a single NIfTI file to the MNI template using FLIRT."""
    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run FLIRT registration command
        subprocess.run(
            [
                "flirt",
                "-in", str(input_path),
                "-ref", str(FSL_TEMPLATE),
                "-out", str(output_path),
                "-dof", "12",  # 12 degrees of freedom for affine transformation
                "-v"           # Verbose output for debugging
            ],
            check=True
        )
        print(f"Registered: {input_path.name} -> {output_path.name}")

    except subprocess.CalledProcessError as e:
        print(f"FLIRT error processing {input_path.name}: {e}")
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")

def process_directory():
    """Process all NIfTI files in the input directory and replicate the structure in the output directory."""
    # Create root output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each class folder (AD, CN, MCI)
    for cls in CLASSES:
        input_cls_dir = INPUT_DIR / cls
        output_cls_dir = OUTPUT_DIR / cls
        output_cls_dir.mkdir(exist_ok=True)

        # Process all NIfTI files in the class folder
        for nii_file in input_cls_dir.glob("*.nii*"):
            output_file = output_cls_dir / nii_file.name
            process_file(nii_file, output_file)

# Run processing directly when the script is executed
print(f"Starting FLIRT registration using 1mm MNI template")
process_directory()
print(f"Completed! Registered files saved to {OUTPUT_DIR}")