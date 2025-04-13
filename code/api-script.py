import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
from skimage import exposure
from PIL import Image
from io import BytesIO
from torchvision.models import inception_v3, Inception_V3_Weights

# Constants
class_labels = ['AD', 'CN', 'MCI']
UPLOAD_DIR = Path("./uploaded_nifti")
REORIENTED_DIR = Path("./reoriented")
REGISTERED_DIR = Path("./registered")
MNI_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain"  # Update if needed

# Initialize app
app = FastAPI(title="NIfTI InceptionV3 Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (InceptionV3 with 3-channel input for grayscale images)
class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV3Classifier, self).__init__()
        # Load InceptionV3 with updated pre-trained weights
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Disable auxiliary outputs for simplicity
        self.model.aux_logits = False
        self.model.AuxLogits = None
        # Modify the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@app.on_event("startup")
def load_model():
    global model
    model = InceptionV3Classifier(num_classes=3).to(device)
    model.load_state_dict(torch.load("../InceptionV3_model.pth", map_location=device, weights_only=False))
    model.eval()

@app.get("/")
def home():
    return {"message": "NIfTI InceptionV3 classifier is running"}

def normalize_slice(slice_data):
    p1, p99 = np.percentile(slice_data, (1, 99))
    clipped = np.clip(slice_data, p1, p99)
    normalized = exposure.rescale_intensity(clipped, out_range=(0, 255))
    return normalized.astype(np.uint8)

def extract_middle_slice(nifti_path):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if len(data.shape) != 3:
        raise ValueError("Only 3D NIfTI images are supported.")

    middle_idx = data.shape[-1] // 2
    # sagittal, coronal, and axial
    slice_data = np.rot90(data[:, :, middle_idx])
    normalized = normalize_slice(slice_data)

    pil_image = Image.fromarray(normalized).convert("L")
    return pil_image

def run_fsl_reorient(input_path, output_path):
    subprocess.run(["fslreorient2std", str(input_path), str(output_path)], check=True)

def run_flirt(input_path, output_path):
    subprocess.run([
        "flirt", "-in", str(input_path),
        "-ref", MNI_TEMPLATE,
        "-out", str(output_path),
        "-dof", "12"
    ], check=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["application/nii", "application/nii.gz", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    try:
        # Save uploaded file
        UPLOAD_DIR.mkdir(exist_ok=True)
        filename = file.filename.replace(".nii.gz", "").replace(".nii", "")
        uploaded_path = UPLOAD_DIR / f"{filename}.nii.gz"
        with open(uploaded_path, "wb") as f:
            f.write(await file.read())

        # Reorient
        REORIENTED_DIR.mkdir(exist_ok=True)
        reoriented_path = REORIENTED_DIR / uploaded_path.name
        run_fsl_reorient(uploaded_path, reoriented_path)

        # FLIRT registration
        REGISTERED_DIR.mkdir(exist_ok=True)
        registered_path = REGISTERED_DIR / uploaded_path.name
        run_flirt(reoriented_path, registered_path)

        # Extract slice
        pil_image = extract_middle_slice(registered_path)

        # Transformations: replicate grayscale channels to match InceptionV3's input requirements (3 channels)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
            transforms.Resize((299, 299)),               # Resize to InceptionV3 input size
            transforms.ToTensor(),                       # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
        ])

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output[0], dim=0)
            confidence, pred_class_idx = torch.max(probs, dim=0)

        predicted_label = class_labels[pred_class_idx.item()]

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "probability": float(confidence)
        })

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FSL error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")