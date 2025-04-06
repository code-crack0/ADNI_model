"""
Simplified FSL Reorientation Script
Processes: input_dir/AD|CN|MCI/*.nii -> output_dir/AD|CN|MCI/*.nii.gz
"""

import subprocess
from pathlib import Path

# Configuration
INPUT_DIR = Path("./mri-images/skullstripped-test-nii")  # Input directory containing AD, CN, MCI subfolders
OUTPUT_DIR = Path("./nifti_reoriented")  # Output directory for processed files
CLASSES = ["AD"]  # Subfolders to process

# CLASSES = ["AD", "CN", "MCI"]  # Subfolders to process

def process_file(input_path, output_path):
    """Reorient a single NIfTI file using FSL."""
    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run FSL reorientation command
        subprocess.run(
            ["fslreorient2std", str(input_path), str(output_path)],
            check=True
        )
        
        print(f"Processed: {input_path.name} -> {output_path.name}")
    
    except subprocess.CalledProcessError as e:
        print(f"FSL error processing {input_path.name}: {e}")
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

# Run the processing directly when the script is executed
print(f"Starting processing from {INPUT_DIR}")
process_directory()
print(f"Finished processing. Output saved to {OUTPUT_DIR}")