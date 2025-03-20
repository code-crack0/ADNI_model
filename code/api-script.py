import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Class labels (must match your dataset)
class_labels = ['AD', 'CN', 'MCI']

# Initialize FastAPI app
app = FastAPI(title="InceptionV3 Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global variables for the model and device
model = None
device = None

class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV3Classifier, self).__init__()
        
        # Load pretrained InceptionV3 model
        weights = models.Inception_V3_Weights.DEFAULT
        self.model = models.inception_v3(weights=weights)
        
        # Adjust classifier for our number of classes (3)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Disable auxiliary outputs for simplicity
        self.model.aux_logits = False
        self.model.AuxLogits = None

    def forward(self, x):
        return self.model(x)

@app.on_event("startup")
def load_model():
    """
    Load the model once during application startup.
    """
    global model, device
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model architecture and load weights
    model = InceptionV3Classifier(num_classes=3).to(device)
    model.load_state_dict(torch.load('inception_v3_model.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode

@app.get("/")
def home():
    """
    Health check endpoint.
    """
    return {"message": "Welcome to the InceptionV3 Classifier API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"Received file with content_type={file.content_type}, filename={file.filename}")

    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail=f"Please upload a PNG image. Received content type: {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("L")

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs[0], dim=0)
            confidence, predicted_class_idx = torch.max(probabilities, dim=0)

        predicted_label = class_labels[predicted_class_idx.item()]

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "confidence": float(confidence)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
