<h1 align="center">Pigment</h1>

<p align="center">
  <em>Beyond the surface. Into the pattern.</em><br>
  An AI-assisted vitiligo screening prototype built on a fine-tuned Vision Transformer.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-292322?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ViT--B%2F16-B77570?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-API-292322?style=flat-square&logo=flask&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-prototype-E8D0CB?style=flat-square&labelColor=292322">
</p>

> **Informational only.** Pigment is a university prototype designed for informational screening, **not medical diagnosis**. It cannot confirm or rule out vitiligo or any other condition. Please consult a qualified dermatologist about any skin concern.

<!-- Add a screenshot: ![Pigment home page](docs/screenshot.png) -->

---

## Overview

Vitiligo is a skin condition marked by the loss of pigment, which appears as lighter patches on different parts of the body. Its appearance varies widely with skin tone, lighting, image quality and background, which makes it hard to recognise consistently from photographs.
<img width="937" height="435" alt="image" src="https://github.com/user-attachments/assets/a7f65013-9456-4349-a594-3ebcc84b6756" />

Pigment explores that problem with a **Vision Transformer (ViT-B/16)** fine-tuned on the [Kaggle Vitiligo Dataset](https://www.kaggle.com/datasets/zienabesam/vitiligo-dataset). A photo goes in, and the model returns the class it finds most likely together with a confidence value for each class.

The project has two parts: a small **Flask API** that runs the model, and a single-file **web app** with an editorial design for uploading or capturing a photo and reading the result.

## Features

- **Photo analysis:** upload or drag and drop an image and get a prediction with per-class confidence.
- **Live camera capture:** take a still straight from the browser. The preview stays on your device until you choose to analyze it.
- **Contrast preview:** an optional, view-only enhancement to make pigment differences easier to see. The original image is what is sent to the model.
- **Clear limits:** every result is shown as informational, with the limitations stated on the page.
- **Accessible and responsive:** semantic HTML, keyboard navigation, visible focus states, and a layout that adapts to mobile. Motion respects the reduced-motion setting.

## How it works

1. **Upload or capture.** The browser sends the image to `POST /predict`.
2. **Preprocessing.** The image is converted to RGB, then resized and normalized by the `ViTImageProcessor` for `google/vit-base-patch16-224` (224×224).
3. **Inference.** The fine-tuned `ViTForImageClassification` model produces logits, and softmax turns them into probabilities.
4. **Informational result.** The API returns the predicted label, its confidence and the probability for each class.
<img width="957" height="442" alt="image" src="https://github.com/user-attachments/assets/671ac3c4-d0b1-4e0c-bdfd-3aaff4b4adf8" />
<img width="948" height="414" alt="image" src="https://github.com/user-attachments/assets/f5a41e11-78e8-45cf-97ad-97946e8ac3d1" />

## Project structure

```
.
├── app.py              # Flask API: loads the model and serves /predict
├── best_vit_model/     # Fine-tuned ViT checkpoint (not included, see below)
├── static/
│   └── index.html      # The web app (single file, images embedded)
└── README.md
```

## Getting started

### Requirements

- Python 3.9 or newer
- A fine-tuned ViT checkpoint saved with `save_pretrained()` in `./best_vit_model`
- A GPU is optional. The app uses CUDA when it is available and falls back to CPU.

### Install

```bash
git clone https://github.com/<your-username>/pigment-vitiligo-vit.git
cd pigment-vitiligo-vit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install flask torch transformers pillow
```

### Add the model

Place your fine-tuned checkpoint in `best_vit_model/`. Model weights are large, so they are not committed to this repository. Add `best_vit_model/` to `.gitignore`, or host the weights separately and link them here.
<img width="885" height="435" alt="image" src="https://github.com/user-attachments/assets/a64849c0-3a72-4b53-8e75-0bf7734ec3b0" />

### Serve the web app from Flask

Add this route to `app.py` so the page and the API share one origin (no CORS setup needed):

```python
@app.route("/")
def home():
    return app.send_static_file("index.html")
```

### Run

```bash
python app.py
```

Open <http://localhost:5000>. Camera capture requires `localhost` or HTTPS.

## API

### `POST /predict`

Send a multipart form with the image in the `image` field.

```bash
curl -X POST http://localhost:5000/predict -F "image=@photo.jpg"
```

Example response (label names come from your model's `id2label` config):

```json
{
  "status": "success",
  "prediction": "<label>",
  "confidence": 0.0,
  "probabilities": {
    "<label_0>": 0.0,
    "<label_1>": 0.0
  }
}
```

| Status | Meaning |
| --- | --- |
| `200` | Prediction returned |
| `400` | No `image` field, or an empty filename |
| `500` | The image could not be processed |

## Evaluation

The model is evaluated on a held-out test split using standard classification metrics. Recall matters most for a screening task, because a missed case costs more than a false alarm.

| Metric | Result |
| --- | --- |
| Accuracy | _to be added_ |
| Precision | _to be added_ |
| Recall | _to be added_ |
| F1 score | _to be added_ |

Add your confusion matrix and the training notebook or script here. This README makes no accuracy or clinical-validation claims until those numbers are published.

## Limitations

- Lighting, skin tone, image quality and background can all change the result.
- The model only separates the classes it was trained on, so other skin conditions can be misclassified.
- The training data is a single public dataset and may not represent every person or setting.
- A prediction is not a diagnosis.
<img width="934" height="426" alt="image" src="https://github.com/user-attachments/assets/61873d69-209a-4f71-b2bb-e47c112d9714" />

## Roadmap

- [ ] Publish evaluation metrics and the confusion matrix
- [ ] Add the training script or notebook
- [ ] Add attention-based visual explanations
- [ ] Add automated tests for the API
- [ ] Add a Dockerfile

## Acknowledgments

- [Vision Transformer](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) via Hugging Face Transformers
- The [Vitiligo Dataset](https://www.kaggle.com/datasets/zienabesam/vitiligo-dataset) on Kaggle

## License

Add a license before publishing, for example MIT. Check the dataset's own terms before sharing trained weights.
