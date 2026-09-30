import requests
import json
import os

# Auto-find first test image
test_dir = "../dataset/test/images"
files_list = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

if not files_list:
    print("No test images found!")
    exit()

image_path = os.path.join(test_dir, files_list[0])
print(f"Testing with: {image_path}\n")

with open(image_path, "rb") as f:
    files = {"file": f}
    r = requests.post("http://localhost:8000/predict/", files=files)

if r.status_code == 200:
    print(json.dumps(r.json(), indent=2))
else:
    print(f"Error {r.status_code}: {r.text}")