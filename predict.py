import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np

# =========================
# Image Transform Pipeline
# =========================
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Prediction Function
# =========================
def predict_image(model, device, image_bytes, class_names, confidence_threshold=0.5):

    try:
        # 1️⃣ Validate and open image
        image = Image.open(image_bytes).convert("RGB")

    except UnidentifiedImageError:
        raise ValueError("Invalid image file. Please upload a valid fundus image.")

    # 2️⃣ Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3️⃣ Inference (AMP enabled if CUDA)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(image_tensor)
        else:
            outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # 4️⃣ Convert to CPU numpy
    confidence_value = confidence.item()
    predicted_index = predicted_class.item()

    # 5️⃣ Top-3 Predictions
    top3_conf, top3_idx = torch.topk(probabilities, 3)
    top3_results = []

    for i in range(3):
        top3_results.append({
            "disease": class_names[top3_idx[0][i].item()],
            "confidence": round(top3_conf[0][i].item() * 100, 2)
        })

    # 6️⃣ Low confidence detection
    if confidence_value < confidence_threshold:
        final_prediction = "Uncertain - Please consult specialist"
    else:
        final_prediction = class_names[predicted_index]

    # 7️⃣ Final structured response
    return {
        "prediction": final_prediction,
        "confidence": round(confidence_value * 100, 2),
        "top3_predictions": top3_results
    }
