import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Load a sample .nii image
nii_path = "./grouped-images/AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__B1_Correction_Br_20081015125601708_S55373_I120990.nii"  # Update with actual file path
nii_img = nib.load(nii_path).get_fdata()

# Extract the middle axial slice
mid_slice = nii_img[nii_img.shape[0] // 2, :, :]

# Display using matplotlib
plt.imshow(mid_slice, cmap="gray")  # Use grayscale colormap
plt.title("Middle Axial Slice")
plt.axis("off")  # Hide axis for better visualization
plt.show()