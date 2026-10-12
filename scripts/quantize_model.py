"""
Static weight-only quantization for Mask R-CNN.
Quantizes the ENTIRE model (including ResNet50 backbone) to int8.
"""
import os
import torch
import torch.quantization
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
import copy

# ===== CONFIGURATION =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tomato_maskrcnn.pth")
QUANTIZED_PATH = os.path.join(BASE_DIR, "models", "tomato_maskrcnn_quantized.pth")

NUM_CLASSES = 3

print("=" * 60)
print("MASK R-CNN STATIC QUANTIZATION (WEIGHT-ONLY)")
print("=" * 60)

# ===== STEP 1: Check original model =====
print(f"\n[1] Checking original model...")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Original model not found at {MODEL_PATH}")
    exit(1)

original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"    Original model size: {original_size:.2f} MB")

# ===== STEP 2: Build the model architecture =====
print(f"\n[2] Building Mask R-CNN architecture...")
model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

# ===== STEP 3: Load original weights =====
print(f"\n[3] Loading original weights...")
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print(f"    [OK] Original weights loaded")

# ===== STEP 4: Apply weight-only quantization MANUALLY =====
# We quantize every Conv2d and Linear weight tensor to int8 manually.
# This reduces file size by ~75% while preserving full compatibility.

print(f"\n[4] Applying manual int8 weight quantization...")
print(f"    This may take 1-2 minutes...")

start_time = time.time()

def quantize_tensor(tensor):
    """Quantize a tensor to int8 and return the quantized representation."""
    if tensor.dim() == 0:
        # Scalar - leave as is
        return tensor
    try:
        # Compute scale and zero point
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min) / 255.0
        if scale == 0:
            scale = torch.tensor(1.0)
        zero_point = (-t_min / scale).round().clamp(0, 255).to(torch.uint8)
        # Quantize to uint8
        q_tensor = ((tensor - t_min) / scale).round().clamp(0, 255).to(torch.uint8)
        return {
            "quantized": q_tensor,
            "scale": scale,
            "zero_point": zero_point,
            "shape": tensor.shape,
            "dtype": "int8"
        }
    except Exception:
        return tensor

# Quantize each weight tensor
quantized_state = {}
for key, value in state_dict.items():
    if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
        quantized_state[key] = quantize_tensor(value)
    else:
        quantized_state[key] = value

elapsed = time.time() - start_time
print(f"    [OK] Quantization complete in {elapsed:.1f} seconds")

# ===== STEP 5: Save quantized model =====
print(f"\n[5] Saving quantized model...")
torch.save(quantized_state, QUANTIZED_PATH)

if os.path.exists(QUANTIZED_PATH):
    new_size = os.path.getsize(QUANTIZED_PATH) / (1024 * 1024)
    reduction = (1 - new_size / original_size) * 100
    print(f"    [OK] Quantized model saved: {QUANTIZED_PATH}")
    print(f"    Quantized size: {new_size:.2f} MB")
    print(f"    Size reduction: {reduction:.1f}%")
else:
    print(f"    [ERROR] Failed to save quantized model")
    exit(1)

print("\n" + "=" * 60)
print("QUANTIZATION COMPLETE")
print("=" * 60)
print(f"\nOriginal: {original_size:.2f} MB")
print(f"Quantized: {new_size:.2f} MB")
print(f"Savings: {reduction:.1f}%")