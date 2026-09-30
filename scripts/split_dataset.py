import os
import shutil
import random
import json

# ===== CONFIGURATION =====
# This goes up two levels from /backend/scripts to find the root Tomato_Monitoring_System
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations", "train_annotations.json")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2

# ===== STEP 1: Get all image filenames =====
all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Total images found in images folder: {len(all_images)}")

if len(all_images) == 0:
    print("ERROR: No images found. Did you move the files into dataset/images?")
    exit()

# Shuffle for randomness
random.seed(42)
random.shuffle(all_images)

# ===== STEP 2: Calculate split counts =====
train_count = int(len(all_images) * TRAIN_RATIO)
val_count = int(len(all_images) * VAL_RATIO)

train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

print(f"Train: {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Test: {len(test_images)}")

# ===== STEP 3: Copy files to their new homes =====
def copy_files(file_list, split_name):
    img_target = os.path.join(DATASET_DIR, split_name, "images")
    ann_target = os.path.join(DATASET_DIR, split_name, "annotations")
    
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(ann_target, exist_ok=True)
    
    for img_file in file_list:
        # Copy the image
        shutil.copy(
            os.path.join(IMAGES_DIR, img_file),
            os.path.join(img_target, img_file)
        )
    
    print(f"✅ {split_name} folder populated with {len(file_list)} images.")

copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

# ===== STEP 4: Load and Split Annotations =====
print("\nSplitting annotations...")

with open(ANNOTATIONS_FILE, "r") as f:
    coco_data = json.load(f)

def get_image_ids(image_files):
    ids = []
    for img in image_files:
        for img_data in coco_data["images"]:
            if img_data["file_name"] == img:
                ids.append(img_data["id"])
                break
    return ids

train_ids = get_image_ids(train_images)
val_ids = get_image_ids(val_images)
test_ids = get_image_ids(test_images)

def filter_coco(coco_data, image_ids):
    new_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": [img for img in coco_data["images"] if img["id"] in image_ids],
        "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in image_ids]
    }
    return new_data

for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    filtered = filter_coco(coco_data, ids)
    output_path = os.path.join(DATASET_DIR, split_name, "annotations", f"{split_name}_annotations.json")
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"✅ {split_name}_annotations.json saved with {len(filtered['images'])} images and {len(filtered['annotations'])} annotations.")

print("\n🎉 Dataset splitting complete!")