import os
from flask import Flask, request, jsonify, send_file
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F
import io

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(app.root_path, "model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preload model and processor during initialization
print("Loading fine-tuned ViT model and processor...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
print("Model loaded on:", DEVICE)

@app.route("/")
def index():
    return send_file(os.path.join(app.root_path, "index.html"))

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in key 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Prepare input tensors matching the visual processor configuration
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).squeeze(0)

            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item()

        predicted_label = model.config.id2label[pred_idx]

        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence,
            "probabilities": {
                model.config.id2label[0]: probabilities[0].item(),
                model.config.id2label[1]: probabilities[1].item()
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    # Suitable for dev environments; use production WSGI server (like Gunicorn) in production
    app.run(host="0.0.0.0", port=5000)
