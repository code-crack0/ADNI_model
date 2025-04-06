import shutil
from pathlib import Path

def find_and_copy_missing_files(input_1: Path, input_2: Path, output_dir: Path):
    # Get list of filenames in both directories
    files_1 = {f.name for f in input_1.glob('*') if f.is_file()}
    files_2 = {f.name for f in input_2.glob('*') if f.is_file()}
    
    # Find files in input_1 that are missing in input_2
    missing_files = files_1 - files_2
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy missing files
    copied_files = []
    for filename in missing_files:
        src = input_1 / filename
        dest = output_dir / filename
        
        if src.exists():
            shutil.copy2(str(src), str(dest))  # Changed from move to copy2
            copied_files.append(filename)
    
    # Print summary
    print(f"Found {len(missing_files)} missing files")
    print(f"Successfully copied {len(copied_files)} files to {output_dir}")
    
    if len(missing_files) != len(copied_files):
        print("\nFiles that couldn't be copied:")
        for f in (missing_files - set(copied_files)):
            print(f" - {f}")

# Example usage
input_1 = Path("./nifti_reoriented/CN")
input_2 = Path("./flirt_registered_T1_1mm_parallel/CN")
output_dir = Path("./output_missing_files_CN")

find_and_copy_missing_files(input_1, input_2, output_dir)