import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Load a sample .nii image
nii_path = "./mri-images/flirt_registered_T1_1mm_parallel/AD/ADNI_002_S_1018_MR_MPR-R__GradWarp__B1_Correction_Br_20070913150426276_S35096_I73021_stripped.nii.gz"  # Update with actual file path
nii_img = nib.load(nii_path).get_fdata()

# Extract the middle axial slice
print(nii_img.shape)
mid_slice = nii_img[nii_img.shape[0] // 2, :, :]

# sagittal, coronal, and axial

# Rotate the slice 90 degrees to the left
rotated_slice = np.rot90(mid_slice)

# Display using matplotlib
plt.imshow(rotated_slice, cmap="gray")  # Use grayscale colormap
plt.title("Middle Axial Slice (Rotated 90° Left)")
plt.axis("off")  # Hide axis for better visualization
plt.show()