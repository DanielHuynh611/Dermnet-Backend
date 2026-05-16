---
title: Basic Docker SDK Space
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

# Derm Foundation FastAPI Two-Stage Classifier

This project deploys a two-stage inference pipeline:

```text
image -> Google Derm Foundation SavedModel -> embedding -> PyTorch MLP head -> class probabilities
```

The preprocessing follows the notebook pipeline:

```text
RGB -> resize 448x448 -> PNG bytes -> tf.train.Example with key image/encoded
```

## Project structure

```text
derm_fastapi_project/
  app/
    main.py                         # FastAPI app and endpoints
    config.py                       # Environment settings
    schemas.py                      # API response models
    services/
      preprocessing.py              # image -> serialized tf.train.Example
      derm_backbone.py              # Derm Foundation wrapper
      predictor.py                  # two-stage sequential forward pass
    models/
      mlp_head.py                   # load model_state_dict from .pt checkpoint
  scripts/
    test_request.py                 # local API test client
  requirements.txt
  class_names.json                  # replace with your real class order
  .env.example
```

## Setup

Create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Put your PyTorch checkpoint in the project root:

```text
derm_foundation_mlp_head.pt
```

The checkpoint should contain:

```python
{
    "model_state_dict": mlp_head.state_dict(),
    # optional but recommended:
    "class_names": [...]
}
```

If the checkpoint does not contain `class_names`, edit `class_names.json` so the order exactly matches your training label order.

## Hugging Face token

Do not put your token in the source code.

Use an environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

You must already have access to `google/derm-foundation` on Hugging Face.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test in browser:

```text
http://127.0.0.1:8000/docs
```

Test with Python:

```bash
python scripts/test_request.py path/to/image.jpg
```

## API endpoint

### POST `/predict`

Input: multipart image upload named `file`.

Output:

```json
{
  "predicted_index": 0,
  "predicted_class": "class_0",
  "confidence": 0.91,
  "probabilities": [
    {"index": 0, "class_name": "class_0", "probability": 0.91},
    {"index": 1, "class_name": "class_1", "probability": 0.02}
  ]
}
```

## Important note about the MLP head

`app/models/mlp_head.py` reconstructs the MLP from Linear layer tensors in `model_state_dict`.
It assumes Linear layers with ReLU between hidden layers. This is usually fine for a simple MLP head.
If your original head used a different activation, BatchNorm, or a more complex custom architecture, replace `InferredMLPHead` with the exact same class used during training.
