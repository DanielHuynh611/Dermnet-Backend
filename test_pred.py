from pathlib import Path
import os
import json

from dotenv import load_dotenv
from PIL import Image

from app.services.predictor import TwoStageDermPredictor


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_IMAGE_PATH = PROJECT_ROOT / "cholinergic-uriticaria-2.jpg"
HEAD_CHECKPOINT_PATH = PROJECT_ROOT / "derm_foundation_mlp_head.pt"
CLASS_NAMES_PATH = PROJECT_ROOT / "class_names.json"


# -----------------------------
# Basic file checks
# -----------------------------

assert TEST_IMAGE_PATH.exists(), f"Test image not found: {TEST_IMAGE_PATH}"
assert HEAD_CHECKPOINT_PATH.exists(), f"MLP checkpoint not found: {HEAD_CHECKPOINT_PATH}"
assert CLASS_NAMES_PATH.exists(), f"class_names.json not found: {CLASS_NAMES_PATH}"


# -----------------------------
# Load .env
# -----------------------------

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("Warning: HF_TOKEN not found.")
    print("This may still work if the Derm Foundation model is already cached locally.")
else:
    print("HF_TOKEN found.")


# -----------------------------
# Check class_names.json
# -----------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))


# -----------------------------
# Optional image sanity check
# -----------------------------

image = Image.open(TEST_IMAGE_PATH).convert("RGB")
print("Test image:", TEST_IMAGE_PATH.name)
print("Image size:", image.size)


# -----------------------------
# Build predictor
# -----------------------------

predictor = TwoStageDermPredictor(
    derm_model_id="google/derm-foundation",
    head_checkpoint_path=str(HEAD_CHECKPOINT_PATH),
    hf_token=HF_TOKEN,
    local_files_only=False,
    image_size=448,
    device_name="auto",
)

print("Predictor loaded.")
print("Device:", predictor.device)


# -----------------------------
# Run prediction
# -----------------------------

image_bytes = TEST_IMAGE_PATH.read_bytes()

result = predictor.predict(image_bytes)


# -----------------------------
# Print result
# -----------------------------

print("\nPrediction result")
print("-----------------")
print("Predicted index:", result["predicted_index"])
print("Predicted class:", result["predicted_class"])
print("Confidence:", result["confidence"])


print("\nTop 5 probabilities")
print("-------------------")

top_probs = sorted(
    result["probabilities"],
    key=lambda x: x["probability"],
    reverse=True,
)[:5]

for item in top_probs:
    print(
        f'{item["index"]:02d} | '
        f'{item["class_name"]} | '
        f'{item["probability"]:.4f}'
    )