"""
Parallel FLIRT Registration Script (4 Cores)
Processes: input_dir/AD|CN|MCI/*.nii -> output_dir/AD|CN|MCI/*.nii.gz
"""

import subprocess
from pathlib import Path
from multiprocessing import Pool

# Configuration
INPUT_DIR = Path("./nifti_reoriented")  # Input directory containing AD, CN, MCI subfolders
OUTPUT_DIR = Path("./flirt_registered_T1_1mm_parallel")  # Output directory for registered files
FSL_TEMPLATE = Path("/home/saeed/fsl/data/standard/MNI152_T1_1mm_brain")  # MNI template
CLASSES = ["MCI"]  # Subfolders to process
# CLASSES = ["AD", "CN", "MCI"]  # Subfolders to process
NUM_CORES = 6  # Number of cores to use for parallel processing

def process_file(args):
    """Register a single NIfTI file to the MNI template using FLIRT."""
    input_path, output_path = args
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
                "-dof", "12",
                "-v"
            ],
            check=True
        )
        print(f"Registered: {input_path.name} -> {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FLIRT error processing {input_path.name}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False

def process_directory():
    """Process all NIfTI files in parallel"""
    # Create root output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create list of all input/output pairs
    file_pairs = []
    for cls in CLASSES:
        input_cls_dir = INPUT_DIR / cls
        output_cls_dir = OUTPUT_DIR / cls
        output_cls_dir.mkdir(exist_ok=True)

        for nii_file in input_cls_dir.glob("*.nii*"):
            output_file = output_cls_dir / nii_file.name
            file_pairs.append((nii_file, output_file))

    # Process files in parallel using 4 cores
    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(process_file, file_pairs)

    # Print summary
    success_count = sum(results)
    print(f"\nProcessing complete! Success rate: {success_count}/{len(file_pairs)}")

# Run processing
if __name__ == "__main__":
    print(f"Starting FLIRT registration using 1mm MNI template (6 cores)")
    process_directory()
    print(f"Registered files saved to: {OUTPUT_DIR}")