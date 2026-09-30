import os
import json
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
from PIL import Image
from collections import defaultdict
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "tomato_maskrcnn.pth")
TEST_IMAGES = os.path.join(BASE_DIR, "dataset", "test", "images")
TEST_ANN = os.path.join(BASE_DIR, "dataset", "test", "annotations", "test_annotations.json")

NUM_CLASSES = 3
CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = {
    1: "Ripe",
    2: "Unripe",
}

# Original dataset category -> 2-class label
CATEGORY_MAPPING = {
    1: 1,  # fully_ripened -> Ripe
    2: 2,  # green         -> Unripe
    3: 1,  # half_ripened  -> Ripe
}

print("Loading model...")
model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
print("[OK] Model loaded.")

with open(TEST_ANN, "r") as f:
    coco = json.load(f)

image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

gt_by_image = defaultdict(list)
for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    if cat_id in CATEGORY_MAPPING:
        gt_by_image[ann["image_id"]].append(CATEGORY_MAPPING[cat_id])

transform = T.Compose([T.ToTensor()])

total_gt = 0
total_correct = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

print("\nEvaluating on test set...")
print("=" * 60)

image_ids = list(image_id_to_file.keys())

for idx, img_id in enumerate(image_ids):
    img_file = image_id_to_file[img_id]
    img_path = os.path.join(TEST_IMAGES, img_file)
    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])

    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    if len(pred_scores) == 0:
        continue

    best_idx = np.argmax(pred_scores)
    if pred_scores[best_idx] < CONFIDENCE_THRESHOLD:
        continue

    predicted_class_name = CLASS_NAMES.get(int(pred_labels[best_idx]), "Unknown")

    gt_classes = gt_by_image[img_id]
    if not gt_classes:
        continue

    actual_class_id = max(set(gt_classes), key=gt_classes.count)
    actual_class_name = CLASS_NAMES.get(actual_class_id, "Unknown")

    total_gt += 1
    class_total[actual_class_name] += 1
    confusion[actual_class_name][predicted_class_name] += 1

    if predicted_class_name == actual_class_name:
        total_correct += 1
        class_correct[actual_class_name] += 1

    if (idx + 1) % 20 == 0:
        print(f"Processed {idx + 1}/{len(image_ids)} images...")

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

accuracy = total_correct / total_gt if total_gt > 0 else 0
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%  ({total_correct}/{total_gt})")

print("\nPer-Class Metrics:")
print("-" * 60)
print(f"{'Class':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-" * 60)

for class_name in ["Ripe", "Unripe"]:
    correct = class_correct[class_name]
    total = class_total[class_name]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"{class_name:<20} {correct:<10} {total:<10} {acc:.2f}%")

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print("\nRows = Actual, Columns = Predicted\n")

headers = ["Ripe", "Unripe"]
print(f"{'':<18}", end="")
for h in headers:
    print(f"{h:<18}", end="")
print()

for actual in headers:
    print(f"{actual:<18}", end="")
    for predicted in headers:
        count = confusion[actual][predicted]
        print(f"{count:<18}", end="")
    print()

results = {
    "overall_accuracy": accuracy,
    "total_images": total_gt,
    "correct_predictions": total_correct,
    "per_class": {
        cls: {
            "correct": class_correct[cls],
            "total": class_total[cls],
            "accuracy": class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        } for cls in headers
    }
}

output_file = os.path.join(BASE_DIR, "backend", "evaluation_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to: {output_file}")
print("\nDone!")