import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Set the path to your NIfTI file here
file_path = "./flirt_registered/AD/ADNI_002_S_1018_MR_MPR-R__GradWarp__B1_Correction_Br_20080113114819719_S43492_I87209_stripped.nii.gz"
file_path_2 = "./flirt_registered_T1_1mm/AD/ADNI_002_S_1018_MR_MPR-R__GradWarp__B1_Correction_Br_20080113114819719_S43492_I87209_stripped.nii.gz"


# Load NIfTI file
print(f"Loading NIfTI file from {file_path}")
nii_img = nib.load(file_path)
img_data = nii_img.get_fdata()
print(f"Image data shape: {img_data.shape}")
# the shape of the image data is (256, 256, 166) where
# 256 is number of slices in the axial, (I think)
# 256 is the number of slices in the coronal plane (I think)
# 166 is the number of slices in sagittal plane

print(f"Loading NIfTI file from {file_path_2}")
nii_img_2 = nib.load(file_path_2)
img_data_2 = nii_img_2.get_fdata()
print(f"Image data shape: {img_data_2.shape}")
print(nii_img_2.header)
# Get slices

mid_axial = img_data[ img_data.shape[0] // 2, :, :]
mid_coronal = img_data[ :, img_data.shape[1] // 2, :]
mid_sagittal = img_data[:, :, img_data.shape[2] // 2]

mid_axial_1 = img_data_2[ img_data_2.shape[0] // 2, :, :]
mid_coronal_1 = img_data_2[ :, img_data_2.shape[1] // 2, :]
mid_sagittal_1 = img_data_2[:, :, img_data_2.shape[2] // 2]


# Plot slices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# File 1 slices
axes[0, 0].imshow(mid_axial, cmap="gray", origin="lower")
axes[0, 0].set_title("File 1: Axial Slice")

axes[0, 1].imshow(np.rot90(mid_coronal), cmap="gray", origin="lower")
axes[0, 1].set_title("File 1: Coronal Slice")

axes[0, 2].imshow(np.rot90(mid_sagittal), cmap="gray", origin="lower")
axes[0, 2].set_title("File 1: Sagittal Slice")

# File 2 slices
axes[1, 0].imshow(mid_axial_1, cmap="gray", origin="lower")
axes[1, 0].set_title("File 2: Axial Slice")

axes[1, 1].imshow(np.rot90(mid_coronal_1), cmap="gray", origin="lower")
axes[1, 1].set_title("File 2: Coronal Slice")

axes[1, 2].imshow(np.rot90(mid_sagittal_1), cmap="gray", origin="lower")
axes[1, 2].set_title("File 2: Sagittal Slice")

plt.tight_layout()
plt.show()