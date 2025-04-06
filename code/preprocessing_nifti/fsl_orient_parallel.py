"""
Parallel FSL Reorientation Script
Processes files concurrently using 6 physical cores
"""

import subprocess
from pathlib import Path
from multiprocessing import Pool
import psutil  # Requires `pip install psutil`

# Configuration
INPUT_DIR = Path("./skull-stripped-images-nii-unzipped")
OUTPUT_DIR = Path("./nifti_reoriented")
CLASSES = ["MCI"]  # ["AD", "CN", "MCI"]

# Detect physical cores (not logical processors)
NUM_WORKERS = psutil.cpu_count(logical=False)  # Returns number of physical cores (6 for your CPU)

def process_file(args):
    """Reorient a single file with error handling"""
    input_path, output_path = args
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["fslreorient2std", str(input_path), str(output_path)],
            check=True
        )
        return f"Processed: {input_path.name}"
    except Exception as e:
        return f"Error processing {input_path.name}: {str(e)}"

def main():
    # Create list of all files to process
    file_pairs = []
    for cls in CLASSES:
        input_cls_dir = INPUT_DIR / cls
        output_cls_dir = OUTPUT_DIR / cls
        
        for nii_file in input_cls_dir.glob("*.nii*"):
            file_pairs.append((
                nii_file,
                output_cls_dir / nii_file.name
            ))

    # Create output directories first
    for _, output_path in file_pairs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process files in parallel using physical cores
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_file, file_pairs)

    # Print results
    for result in results:
        print(result)

print(f"Starting parallel processing using {NUM_WORKERS} physical cores")
main()
print(f"Finished processing. Output saved to {OUTPUT_DIR}")