from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import torch
import json

from model_loader import load_model
from predict import predict_image

# ==============================
# Initialize FastAPI App
# ==============================
app = FastAPI(title="Fundus Disease Classification API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Load Disease Mapping
# ==============================
with open("disease_mapping.json", "r") as f:
    disease_mapping = json.load(f)

# If JSON is dict → convert to list
if isinstance(disease_mapping, dict):
    class_names = list(disease_mapping.values())
else:
    class_names = disease_mapping

# ==============================
# Load Model Once at Startup
# ==============================
MODEL_PATH = "best_convnextv2_fundus.pth"

model, device = load_model(
    model_path=MODEL_PATH,
    num_classes=len(class_names)
)

print("✅ Model Loaded Successfully!")

# ==============================
# Prediction Endpoint
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = BytesIO(await file.read())

        result = predict_image(
            model=model,
            device=device,
            image_bytes=image_bytes,
            class_names=class_names
        )

        return {
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ==============================
# Health Check Route
# ==============================
@app.get("/")
def home():
    return {"message": "Fundus Disease Classification API Running"}
