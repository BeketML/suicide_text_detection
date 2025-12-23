from pathlib import Path
import logging

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
Inference and API service for the fine‑tuned XLM‑RoBERTa suicide text classifier.

- Loads the model and tokenizer from the local `results` directory
  (where `config.json`, `model.safetensors`, `tokenizer_config.json`, etc. are saved).
- Exposes a FastAPI endpoint `/predict` for programmatic access.
- Can also be imported and used directly via the `predict` function.
"""

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

# Local path to the fine-tuned model directory.
# Project structure:
#   <project_root>/results/              <-- model & tokenizer files here
#   <project_root>/suicide_final_project/final/inference.py  (this file)
# So we go two levels up from this file to project root, then into "results".
MODEL_PATH = str((Path(__file__).resolve().parents[2] / "results").resolve())
logger.info(f"Loading model from: {MODEL_PATH}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Select device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Using GPU: {gpu_name}")
else:
    device = torch.device("cpu")
    logger.info("Using CPU (no CUDA device detected)")

model.to(device)
model.eval()

# Mapping from class id to human-readable label (must match training)
ID2LABEL = {
    0: "non_suicide",
    1: "suicide",
}


def predict(text: str) -> dict:
    """
    Run inference on a single input text.

    Args:
        text: Input string to classify.

    Returns:
        dict with keys:
          - 'label_id': predicted class index (int)
          - 'label': human-readable label (str)
          - 'confidence': probability of the predicted class (float)
          - 'device': device used for inference ('cpu' or 'cuda')
    """
    text = str(text).strip()
    if not text:
        raise ValueError("Input text is empty")

    # Tokenize input text
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Move tensors to the same device as the model
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Get predicted class and its probability
    pred_id = int(torch.argmax(probs, dim=-1).item())
    confidence = float(probs[0, pred_id].item())
    label = ID2LABEL.get(pred_id, str(pred_id))

    logger.info(
        "Prediction made",
        extra={
            "label_id": pred_id,
            "label": label,
            "confidence": confidence,
            "text_len": len(text),
            "device": str(device),
        },
    )

    return {
        "label_id": pred_id,
        "label": label,
        "confidence": confidence,
        "device": str(device),
    }


# -----------------------------------------------------------------------------
# FastAPI app definition
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Suicide Risk Classifier API",
    description="FastAPI endpoint for the XLM‑RoBERTa suicide text classifier.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label_id: int
    label: str
    confidence: float
    device: str


@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """HTTP endpoint for making predictions."""
    result = predict(request.text)
    return PredictResponse(**result)


if __name__ == "__main__":
    # Local CLI usage for quick manual testing
    sample_text = input("Enter text to classify: ")
    try:
        result = predict(sample_text)
        print(
            f"Predicted label: {result['label']} "
            f"(id={result['label_id']}, confidence={result['confidence']:.4f}, device={result['device']})"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        print(f"Error: {e}")