import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Set the path to your NIfTI file here
# file_path = "./grouped-images/AD/ADNI_002_S_0619_MR_MPR-R__GradWarp__B1_Correction_Br_20081015125601708_S55373_I120990.nii"
# file_path = "./grouped-images/AD/ADNI_005_S_0221_MR_MPR-R__GradWarp__B1_Correction_Br_20080410142515399_S28460_I102059.nii"
file_path = "./grouped-images/AD/ADNI_002_S_0938_MR_MPR-R__GradWarp__B1_Correction_Br_20070713122900520_S29621_I60044.nii"

# Load NIfTI file
print(f"Loading NIfTI file from {file_path}")
nii_img = nib.load(file_path)
img_data = nii_img.get_fdata()
print(f"Image data shape: {img_data.shape}")

# Get slices
# mid_axial = nii_img[nii_img.shape[0] // 2, :, :]
# mid_coronal = nii_img[nii_img.shape[1] // 2, :, :]
# mid_sagittal = nii_img[nii_img.shape[2] // 2, :, :]

mid_axial = img_data[ img_data.shape[0] // 2, :, :]
mid_coronal = img_data[ :, img_data.shape[0] // 2, :]
mid_sagittal = img_data[:, :, img_data.shape[2] // 2]

# mid_axial = img_data[ :, :, img_data.shape[0] // 2]
# mid_coronal = img_data[ :, :, img_data.shape[1] // 2]
# mid_sagittal = img_data[:, :, img_data.shape[2] // 2]

# mid_axial = img_data[:, :, img_data.shape[2] // 2]
# mid_coronal = img_data[:, img_data.shape[1] // 2, :]
# mid_sagittal = img_data[img_data.shape[0] // 2, :, :]

# Plot slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Axial slice
axes[0].imshow(mid_axial, cmap="gray", origin="lower")
axes[0].set_title("Axial Slice")

# Coronal slice
axes[1].imshow(np.rot90(np.rot90(mid_coronal)), cmap="gray", origin="lower")
axes[1].set_title("Coronal Slice")

# Sagittal slice
axes[2].imshow(np.rot90(np.rot90(mid_sagittal)), cmap="gray", origin="lower")
axes[2].set_title("Sagittal Slice")

plt.show()