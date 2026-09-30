import torch
import torchvision
import cv2
import fastapi
import numpy as np

print("--- SETUP SUCCESSFUL ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"FastAPI Version: {fastapi.__version__}")

if torch.cuda.is_available():
    print("GPU: YES! Your computer has an NVIDIA GPU and is ready for fast AI training.")
else:
    print("GPU: NO. You are using CPU. This is fine, but we will use Google Colab for training later to save time.")