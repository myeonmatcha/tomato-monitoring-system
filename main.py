from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
import io
import numpy as np
import os
import datetime
import cv2

# ===== CONFIGURATION =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tomato_maskrcnn.pth")
NUM_CLASSES = 3

CLASS_NAMES = {1: "Ripe", 2: "Unripe"}

# ===== CONFIDENCE THRESHOLD =====
# Lowered to 0.25 to detect tomatoes in complex real-world images
CONFIDENCE_THRESHOLD = 0.25

STAGES_8 = [
    "Immature Green",
    "Mature Green",
    "Breaker",
    "Turning",
    "Pink",
    "Light Red",
    "Red (Ripe)",
    "Over Ripe",
]

RIPENESS_PRIORITY = {
    "Immature Green": 1,
    "Mature Green": 2,
    "Breaker": 3,
    "Turning": 4,
    "Pink": 5,
    "Light Red": 6,
    "Red (Ripe)": 7,
    "Over Ripe": 8,
}

app = FastAPI(title="Tomato Monitoring System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODEL =====
print("Loading Mask R-CNN model...")
model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print("[OK] Trained model weights loaded successfully!")
else:
    print(f"[WARN] No trained model found at {MODEL_PATH}.")

model.eval()


# ===== HSV REFINEMENT (3-class -> 8-stage) =====
def refine_to_8_stage(model_class, mean_hsv):
    """
    Refine a 3-class model prediction into one of 8 ripeness stages
    using HSV color analysis of the detected tomato mask.

    HSV ranges (OpenCV):
      Hue (H): 0-179  (0=red, 30=yellow, 60=green, 90=cyan, 120=blue, 150=magenta)
      Saturation (S): 0-255 (0=grey, 255=fully saturated)
      Value (V): 0-255 (0=black, 255=full brightness)
    """
    h, s, v = mean_hsv

    if model_class == "Unripe":
        # Green/yellow tomatoes - subdivide by saturation
        if s > 180:
            return "Immature Green"
        elif s > 130:
            return "Mature Green"
        else:
            return "Breaker"

    elif model_class == "Ripe":
        # Red/orange tomatoes - subdivide by hue and brightness
        if h < 10:
            # Red hue zone: distinguish Ripe vs Over Ripe
            # Over Ripe = VERY dark (v < 110) AND dull (s < 180)
            if v < 110 and s < 180:
                return "Over Ripe"
            elif v < 130:
                return "Light Red"
            else:
                return "Red (Ripe)"
        elif 10 <= h < 18:
            return "Light Red"
        elif 18 <= h < 25:
            return "Pink"
        else:
            return "Turning"

    return "Unknown"


def extract_mean_hsv(image_np, mask_np):
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    if mask_np.shape != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    mask_bool = mask_np > 0.5
    if mask_bool.sum() == 0:
        return (90, 100, 100)

    pixels = hsv_image[mask_bool]
    return (
        float(pixels[:, 0].mean()),
        float(pixels[:, 1].mean()),
        float(pixels[:, 2].mean()),
    )


# ===== QUALITY GRADING =====
def compute_circularity(mask_np):
    if mask_np.shape != (256, 256):
        mask_np = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_NEAREST)

    mask_binary = (mask_np > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.5

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0.5

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)


def compute_color_uniformity(image_np, mask_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    if mask_np.shape != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    mask_bool = mask_np > 0.5
    if mask_bool.sum() < 100:
        return 0.5

    pixels = hsv[mask_bool]

    h_std = np.std(pixels[:, 0]) / 90.0
    s_std = np.std(pixels[:, 1]) / 128.0
    v_std = np.std(pixels[:, 2]) / 128.0

    avg_std = (h_std + s_std + v_std) / 3.0
    uniformity = max(0.0, 1.0 - avg_std)
    return uniformity


def compute_size_score(mask_np, image_np):
    if mask_np.shape != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    mask_area = (mask_np > 0.5).sum()
    image_area = image_np.shape[0] * image_np.shape[1]

    if image_area == 0:
        return 0.5

    relative_size = mask_area / image_area
    size_score = min(relative_size / 0.10, 1.0)
    return size_score


def compute_quality_grade(image_np, mask_np, stage_8):
    circularity = compute_circularity(mask_np)
    uniformity = compute_color_uniformity(image_np, mask_np)
    size_score = compute_size_score(mask_np, image_np)

    score = (circularity * 0.35) + (uniformity * 0.45) + (size_score * 0.20)

    if stage_8 == "Over Ripe":
        score -= 0.20
    elif stage_8 == "Immature Green":
        score -= 0.10

    score = max(0.0, min(score, 1.0))

    if score >= 0.85:
        grade = "A"
        description = "Excellent Quality — Suitable for export and premium market"
        recommendation = "Best for fresh consumption, export, or premium retail display."
    elif score >= 0.70:
        grade = "B"
        description = "Good Quality — Suitable for local market"
        recommendation = "Ideal for local markets, supermarkets, or direct consumer sale."
    elif score >= 0.50:
        grade = "C"
        description = "Moderate Quality — Suitable for processing"
        recommendation = "Best used for sauces, ketchup, canning, or juice processing."
    else:
        grade = "D"
        description = "Poor Quality — Not suitable for consumption"
        recommendation = "Reject or discard. May be used for composting or animal feed."

    features = {
        "circularity": round(circularity, 3),
        "uniformity": round(uniformity, 3),
        "size_score": round(size_score, 3),
        "overall_score": round(score, 3),
        "description": description,
        "recommendation": recommendation,
    }

    return grade, features


# ===== HARVEST LOGIC =====
def determine_harvest_readiness(stage_8):
    ready_stages = ["Red (Ripe)", "Light Red"]
    soon_stages = ["Pink", "Turning"]

    if stage_8 == "Over Ripe":
        return "Past Harvest", -1
    elif stage_8 in ready_stages:
        return "Ready", 0
    elif stage_8 in soon_stages:
        return "Ready Soon", 2
    elif stage_8 == "Breaker":
        return "Not Ready", 5
    else:
        return "Not Ready", 7


def compute_harvest_date(days_to_harvest):
    if days_to_harvest < 0:
        return "Past - Discard or Process Soon"
    today = datetime.date.today()
    harvest_date = today + datetime.timedelta(days=days_to_harvest)
    return harvest_date.strftime("%B %d, %Y")


def build_summary_message(counts):
    total = sum(counts.values())
    parts = []
    for stage in STAGES_8:
        if counts.get(stage, 0) > 0:
            parts.append(f"{counts[stage]} {stage}")
    if total == 1:
        return f"1 tomato detected: {parts[0]}"
    return f"{total} tomatoes detected: {', '.join(parts)}"


# ===== API ENDPOINTS =====
@app.get("/")
def home():
    return {"message": "Tomato Monitoring System API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image)

        with torch.no_grad():
            prediction = model([img_tensor])

        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        masks = prediction[0]['masks'].cpu().numpy()

        all_detections = []
        counts = {}

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
                class_id = int(labels[i])
                model_class = CLASS_NAMES.get(class_id, "Unknown")

                mask_np = masks[i, 0]
                mean_hsv = extract_mean_hsv(image_np, mask_np)
                stage_8 = refine_to_8_stage(model_class, mean_hsv)

                quality, quality_features = compute_quality_grade(image_np, mask_np, stage_8)
                readiness, days_to_harvest = determine_harvest_readiness(stage_8)
                harvest_date = compute_harvest_date(days_to_harvest)

                all_detections.append({
                    "ripeness_stage": stage_8,
                    "model_class": model_class,
                    "confidence": float(scores[i]),
                    "mean_hsv": {
                        "hue": round(mean_hsv[0], 1),
                        "saturation": round(mean_hsv[1], 1),
                        "value": round(mean_hsv[2], 1),
                    },
                    "quality_grade": quality,
                    "quality_description": quality_features.get("description", ""),
                    "quality_recommendation": quality_features.get("recommendation", ""),
                    "quality_features": quality_features,
                    "harvest_readiness": readiness,
                    "estimated_harvest_day": harvest_date,
                })

                counts[stage_8] = counts.get(stage_8, 0) + 1

        if not all_detections:
            return {
                "status": "no_detection",
                "message": "No tomatoes detected. Please try another image.",
                "result": None,
                "all_detections": [],
                "summary": None,
                "stages": STAGES_8,
            }

        best = max(all_detections, key=lambda x: (
            RIPENESS_PRIORITY.get(x["ripeness_stage"], 0),
            x["confidence"]
        ))

        active_stage_index = (
            STAGES_8.index(best["ripeness_stage"])
            if best["ripeness_stage"] in STAGES_8 else 0
        )

        summary_message = build_summary_message(counts)

        return {
            "status": "success",
            "result": best,
            "active_stage_index": active_stage_index,
            "all_detections": all_detections,
            "summary": {
                "total_detected": len(all_detections),
                "counts": counts,
                "message": summary_message,
            },
            "stages": STAGES_8,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)