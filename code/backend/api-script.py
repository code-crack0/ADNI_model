# import os
# import torch
# import torch.nn as nn
# from torchvision import models
# import torchvision.transforms as transforms
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import subprocess
# import nibabel as nib
# import numpy as np
# from skimage import exposure
# from PIL import Image
# from io import BytesIO
# from torchvision.models import inception_v3, Inception_V3_Weights

# # Constants
# class_labels = ['AD', 'CN', 'MCI']
# UPLOAD_DIR = Path("./uploaded_nifti")
# REORIENTED_DIR = Path("./reoriented")
# REGISTERED_DIR = Path("./registered")
# # MNI_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain"  # Update if needed
# MNI_TEMPLATE = "/home/saeed/fsl/data/standard/MNI152_T1_1mm_brain"

# # Initialize app
# app = FastAPI(title="NIfTI InceptionV3 Classifier API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the model (InceptionV3 with 3-channel input for grayscale images)
# class InceptionV3Classifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super(InceptionV3Classifier, self).__init__()
#         # Load InceptionV3 with updated pre-trained weights
#         self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
#         # Disable auxiliary outputs for simplicity
#         self.model.aux_logits = False
#         self.model.AuxLogits = None
#         # Modify the final fully connected layer to match the number of classes
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# @app.on_event("startup")
# def load_model():
#     global model
#     model = InceptionV3Classifier(num_classes=3).to(device)
#     model.load_state_dict(torch.load("../InceptionV3_model.pth", map_location=device, weights_only=False))
#     model.eval()

# @app.get("/")
# def home():
#     return {"message": "NIfTI InceptionV3 classifier is running"}

# def normalize_slice(slice_data):
#     p1, p99 = np.percentile(slice_data, (1, 99))
#     clipped = np.clip(slice_data, p1, p99)
#     normalized = exposure.rescale_intensity(clipped, out_range=(0, 255))
#     return normalized.astype(np.uint8)

# def extract_middle_slice(nifti_path):
#     img = nib.load(str(nifti_path))
#     data = img.get_fdata()

#     if len(data.shape) != 3:
#         raise ValueError("Only 3D NIfTI images are supported.")

#     middle_idx = data.shape[-1] // 2
#     # sagittal, coronal, and axial
#     slice_data = np.rot90(data[:, :, middle_idx])
#     normalized = normalize_slice(slice_data)

#     pil_image = Image.fromarray(normalized).convert("L")
#     return pil_image

# def run_fsl_reorient(input_path, output_path):
#     # Convert to string and ensure paths are properly quoted/escaped
#     input_path_str = str(input_path)
#     output_path_str = str(output_path)
    
#     # Print paths for debugging
#     print(f"Running fslreorient2std on: {input_path_str}")
#     print(f"Output to: {output_path_str}")
    
#     try:
#         subprocess.run(["fslreorient2std", input_path_str, output_path_str], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"FSL command failed: {e}")
#         # Check if file exists
#         if not os.path.exists(input_path_str):
#             print(f"Input file does not exist: {input_path_str}")
#         raise

# def run_flirt(input_path, output_path):
#     # Convert to string and ensure paths are properly quoted/escaped
#     input_path_str = str(input_path)
#     output_path_str = str(output_path)
    
#     # Print paths for debugging
#     print(f"Running flirt on: {input_path_str}")
#     print(f"Output to: {output_path_str}")
    
#     try:
#         subprocess.run([
#             "flirt", "-in", input_path_str,
#             "-ref", MNI_TEMPLATE,
#             "-out", output_path_str,
#             "-dof", "12",
#             "-v"
#         ], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"FSL command failed: {e}")
#         # Check if file exists
#         if not os.path.exists(input_path_str):
#             print(f"Input file does not exist: {input_path_str}")
#         raise

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if file.content_type not in ["application/nii", "application/nii.gz", "application/octet-stream"]:
#         raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

#     try:
#         # Save uploaded file
#         UPLOAD_DIR.mkdir(exist_ok=True)
        
#         # Sanitize filename - more thorough character replacement
#         original_filename = file.filename
#         sanitized_filename = ''.join(c if c.isalnum() or c in '_.' else '_' for c in original_filename)

#         # Remove file extension and add it back consistently
#         base_filename = sanitized_filename.replace(".nii.gz", "").replace(".nii", "")
#         uploaded_path = UPLOAD_DIR / f"{base_filename}.nii.gz"

#         print(f"Original filename: {original_filename}")
#         print(f"Sanitized filename: {sanitized_filename}")
#         print(f"Final path: {uploaded_path}")
        
#         with open(uploaded_path, "wb") as f:
#             f.write(await file.read())

#         # Reorient
#         REORIENTED_DIR.mkdir(exist_ok=True)
#         reoriented_path = REORIENTED_DIR / uploaded_path.name
#         run_fsl_reorient(uploaded_path, reoriented_path)

#         # FLIRT registration
#         REGISTERED_DIR.mkdir(exist_ok=True)
#         registered_path = REGISTERED_DIR / uploaded_path.name
#         run_flirt(reoriented_path, registered_path)

#         # Extract slice
#         pil_image = extract_middle_slice(registered_path)

#         # Transformations: replicate grayscale channels to match InceptionV3's input requirements (3 channels)
#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
#             transforms.Resize((299, 299)),               # Resize to InceptionV3 input size
#             transforms.ToTensor(),                       # Convert to tensor
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
#                          std=[0.229, 0.224, 0.225])   # ImageNet std
#         ])

#         input_tensor = transform(pil_image).unsqueeze(0).to(device)

#         # Predict
#         with torch.no_grad():
#             output = model(input_tensor)
#             probs = torch.softmax(output[0], dim=0)
#             confidence, pred_class_idx = torch.max(probs, dim=0)

#         predicted_label = class_labels[pred_class_idx.item()]

#         return JSONResponse(content={
#             "predicted_class": predicted_label,
#             "probability": float(confidence)
#         })

#     except subprocess.CalledProcessError as e:
#         raise HTTPException(status_code=500, detail=f"FSL error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    


import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import subprocess
import nibabel as nib
from skimage import exposure
from PIL import Image
from io import BytesIO
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib.patheffects as path_effects

# Constants
class_labels = ['AD', 'CN', 'MCI']
UPLOAD_DIR = Path("./uploaded_nifti")
STRIPPED_DIR = Path("./skull_stripped")  # New directory for skull-stripped files
REORIENTED_DIR = Path("./reoriented")
REGISTERED_DIR = Path("./registered")
HEATMAP_DIR = Path("./heatmaps")  # New directory for storing heatmaps
MNI_TEMPLATE = "/home/saeed/fsl/data/standard/MNI152_T1_1mm_brain"
synthstrip_model_path = "/mnt/c/Users/sbm76/Downloads/synthstrip.1.pt"  # Update with actual model path

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

# # GradCAM implementation
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         def forward_hook(module, input, output):
#             self.activations = output
            
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0]
        
#         # Register hooks on the target layer
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)
    
#     def generate_cam(self, input_image, target_class):
#         # Forward pass
#         output = self.model(input_image)
        
#         # Clear previous gradients
#         self.model.zero_grad()
        
#         # Target for backprop
#         one_hot = torch.zeros_like(output)
#         one_hot[0, target_class] = 1
        
#         # Backward pass
#         output.backward(gradient=one_hot, retain_graph=True)
        
#         # Get weights
#         gradients = self.gradients.detach().cpu().data.numpy()[0]
#         activations = self.activations.detach().cpu().data.numpy()[0]
        
#         # Global average pooling
#         weights = np.mean(gradients, axis=(1, 2))
        
#         # Weighted combination of activation maps
#         cam = np.zeros(activations.shape[1:], dtype=np.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i, :, :]
        
#         # Apply ReLU to focus on positive contributions
#         cam = np.maximum(cam, 0)
        
#         # Normalize CAM
#         if np.max(cam) > 0:
#             cam = cam / np.max(cam)
        
#         # Resize to input image size
#         cam = cv2.resize(cam, (299, 299))
        
#         return cam

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on the target layer
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        # Replace the deprecated register_backward_hook with register_full_backward_hook
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.detach().cpu().data.numpy()[0]
        activations = self.activations.detach().cpu().data.numpy()[0]
        
        # Global average pooling
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize CAM
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Resize to input image size
        cam = cv2.resize(cam, (299, 299))
        
        return cam
        
    def __del__(self):
        # Remove hooks when the object is deleted
        self.forward_handle.remove()
        self.backward_handle.remove()

# Function to overlay heatmap on original image
def overlay_heatmap(original_image, cam):
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    
    # Convert PIL image to numpy array
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Convert to float if needed
    if original_image.dtype != np.float32:
        original_image = original_image.astype(np.float32) / 255.0
    
    # Ensure original image has 3 channels (RGB)
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image, original_image, original_image], axis=2)
    
    # Overlay: heatmap * alpha + original * (1-alpha)
    alpha = 0.4
    overlaid = original_image * (1 - alpha) + heatmap * alpha
    overlaid = np.clip(overlaid, 0, 1)
    
    # Convert back to uint8
    overlaid = (overlaid * 255).astype(np.uint8)
    
    return Image.fromarray(overlaid)

@app.on_event("startup")
def load_model():
    global model, gradcam
    model = InceptionV3Classifier(num_classes=3).to(device)
    model.load_state_dict(torch.load("../../InceptionV3_model.pth", map_location=device, weights_only=False))
    model.eval()
    
    # Initialize GradCAM with the last convolutional layer
    target_layer = model.model.Mixed_7c
    gradcam = GradCAM(model, target_layer)

@app.get("/")
def home():
    return {"message": "NIfTI InceptionV3 classifier with GradCAM is running"}

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

def run_synthstrip(input_path, output_path):
    """
    Run SynthStrip on a NIfTI file to perform skull stripping.
    """
    input_path_str = str(input_path)
    output_path_str = str(output_path)
    
    print(f"Running SynthStrip on {input_path_str}...")
    
    cmd = [
        "nipreps-synthstrip",  # Command-line tool for SynthStrip (ensure it's installed)
        "-i", input_path_str,
        "-o", output_path_str,
        "--model", synthstrip_model_path
        # "--gpu"  # Uncomment to use GPU acceleration
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Saved skull-stripped file to {output_path_str}")
    except subprocess.CalledProcessError as e:
        print(f"SynthStrip command failed: {e}")
        if not os.path.exists(input_path_str):
            print(f"Input file does not exist: {input_path_str}")
        raise

def run_fsl_reorient(input_path, output_path):
    # Convert to string and ensure paths are properly quoted/escaped
    input_path_str = str(input_path)
    output_path_str = str(output_path)
    
    # Print paths for debugging
    print(f"Running fslreorient2std on: {input_path_str}")
    print(f"Output to: {output_path_str}")
    
    try:
        subprocess.run(["fslreorient2std", input_path_str, output_path_str], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FSL command failed: {e}")
        # Check if file exists
        if not os.path.exists(input_path_str):
            print(f"Input file does not exist: {input_path_str}")
        raise

def run_flirt(input_path, output_path):
    # Convert to string and ensure paths are properly quoted/escaped
    input_path_str = str(input_path)
    output_path_str = str(output_path)
    
    # Print paths for debugging
    print(f"Running flirt on: {input_path_str}")
    print(f"Output to: {output_path_str}")
    
    try:
        subprocess.run([
            "flirt", "-in", input_path_str,
            "-ref", MNI_TEMPLATE,
            "-out", output_path_str,
            "-dof", "12",
            "-v"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FSL command failed: {e}")
        # Check if file exists
        if not os.path.exists(input_path_str):
            print(f"Input file does not exist: {input_path_str}")
        raise

def overlay_heatmap_with_labels(original_image, cam, predicted_class, probability):
    """
    Create a clean heatmap visualization without bottom text.
    """
    # Convert PIL image to numpy array if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Create figure
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)
    
    # Convert to RGB if grayscale
    if len(original_image.shape) == 2:
        original_image_rgb = np.stack([original_image, original_image, original_image], axis=2)
    else:
        original_image_rgb = original_image
        
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    
    # Convert to float
    if original_image_rgb.dtype != np.float32:
        original_image_rgb = original_image_rgb.astype(np.float32) / 255.0
    
    # Overlay with transparency
    alpha = 0.4
    overlaid = original_image_rgb * (1 - alpha) + heatmap * alpha
    overlaid = np.clip(overlaid, 0, 1)
    
    # Display overlay
    mappable = ax.imshow(overlaid)
    
    # Add colorbar
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
    
    # Simple title with prediction
    ax.set_title(f"Predicted: {predicted_class} ({probability:.1%})", 
                fontsize=14, pad=10)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout - no bottom text
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt_image = Image.open(buf)
    plt.close()
    
    return plt_image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["application/nii", "application/nii.gz", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    try:
        # Save uploaded file
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        # Sanitize filename - more thorough character replacement
        original_filename = file.filename
        sanitized_filename = ''.join(c if c.isalnum() or c in '_.' else '_' for c in original_filename)

        # Remove file extension and add it back consistently
        base_filename = sanitized_filename.replace(".nii.gz", "").replace(".nii", "")
        uploaded_path = UPLOAD_DIR / f"{base_filename}.nii"

        print(f"Original filename: {original_filename}")
        print(f"Sanitized filename: {sanitized_filename}")
        print(f"Final path: {uploaded_path}")
        
        with open(uploaded_path, "wb") as f:
            f.write(await file.read())

        # Create all necessary directories
        STRIPPED_DIR.mkdir(exist_ok=True)
        REORIENTED_DIR.mkdir(exist_ok=True)
        REGISTERED_DIR.mkdir(exist_ok=True)
        HEATMAP_DIR.mkdir(exist_ok=True)
        
        # Step 1: Skull Stripping
        stripped_path = STRIPPED_DIR / uploaded_path.name
        run_synthstrip(uploaded_path, stripped_path)

        # Step 2: Reorient the skull-stripped file
        reoriented_path = REORIENTED_DIR / uploaded_path.name
        run_fsl_reorient(stripped_path, reoriented_path)

        # Step 3: FLIRT registration
        registered_path = REGISTERED_DIR / (uploaded_path.name + ".gz")
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
        
        # # Generate GradCAM visualization
        # cam = gradcam.generate_cam(input_tensor, pred_class_idx.item())
        # heatmap_img = overlay_heatmap(pil_image.resize((299, 299)), cam)
        
        # # Save heatmap image
        # heatmap_path = HEATMAP_DIR / f"{base_filename}_heatmap.png"
        # heatmap_img.save(heatmap_path)
        
        # # Convert heatmap to base64 for response
        # buffered = BytesIO()
        # heatmap_img.save(buffered, format="PNG")
        # heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Generate GradCAM visualization
        cam = gradcam.generate_cam(input_tensor, pred_class_idx.item())
        
        # Use the new function with labels
        heatmap_img = overlay_heatmap_with_labels(
            pil_image.resize((299, 299)), 
            cam,
            predicted_label,
            float(confidence)
        )
        
        # Save heatmap image
        heatmap_path = HEATMAP_DIR / f"{base_filename}_heatmap.png"
        heatmap_img.save(heatmap_path)
        
        # Convert heatmap to base64 for response
        buffered = BytesIO()
        heatmap_img.save(buffered, format="PNG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "probability": float(confidence),
            "heatmap": heatmap_base64
        })

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str):
    """
    Endpoint to retrieve a saved heatmap image.
    """
    heatmap_path = HEATMAP_DIR / filename
    
    if not heatmap_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    try:
        return Response(content=open(heatmap_path, "rb").read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving heatmap: {str(e)}")