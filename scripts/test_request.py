import sys
import requests


if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/test_request.py path/to/image.jpg")

image_path = sys.argv[1]
url = "http://127.0.0.1:8000/predict"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.status_code)
print(response.json())
