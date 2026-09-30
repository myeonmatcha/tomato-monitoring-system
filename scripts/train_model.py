import os
import json
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import time

# ===== CONFIGURATION =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_ANN = os.path.join(DATASET_DIR, "train", "annotations", "train_annotations.json")
VAL_IMAGES = os.path.join(DATASET_DIR, "val", "images")
VAL_ANN = os.path.join(DATASET_DIR, "val", "annotations", "val_annotations.json")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "backend", "models", "tomato_maskrcnn.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ===== CUSTOM DATASET CLASS =====
class TomatoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        with open(annotation_file, "r") as f:
            self.coco = json.load(f)
        
        # Map image IDs to filenames
        self.image_ids = [img["id"] for img in self.coco["images"]]
        self.image_id_to_file = {img["id"]: img["file_name"] for img in self.coco["images"]}
        
        # Build category map (background is 0)
        self.cat_id_to_label = {cat["id"]: i+1 for i, cat in enumerate(self.coco["categories"])}
        self.num_classes = len(self.coco["categories"]) + 1 # +1 for background
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        print(f"Dataset loaded: {len(self.image_ids)} images, {self.num_classes} classes (including background)")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_file = self.image_id_to_file[img_id]
        img_path = os.path.join(self.image_dir, img_file)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = T.ToTensor()(img)
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Bounding box [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            
            # Category label
            labels.append(self.cat_id_to_label[ann["category_id"]])
            
            # Segmentation mask (polygon format)
            seg = ann["segmentation"]
            if isinstance(seg, list) and len(seg) > 0:
                # Convert polygon to binary mask using PIL
                from PIL import ImageDraw
                mask_img = Image.new('L', (img.width, img.height), 0)
                draw = ImageDraw.Draw(mask_img)
                for poly in seg:
                    # poly is a flat list [x1, y1, x2, y2, ...]
                    poly_points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    draw.polygon(poly_points, outline=1, fill=1)
                mask = np.array(mask_img, dtype=np.uint8)
            else:
                mask = np.zeros((img.height, img.width), dtype=np.uint8)
            
            masks.append(mask)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = areas
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([img_id])
        
        return img_tensor, target

# ===== COLLATE FUNCTION =====
def collate_fn(batch):
    return tuple(zip(*batch))

# ===== LOAD MODEL =====
def get_model(num_classes):
    # Load pre-trained Mask R-CNN with ResNet50 backbone
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# ===== TRAINING FUNCTION =====
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if i % 10 == 0:
            print(f"Epoch {epoch} | Batch {i}/{len(data_loader)} | Loss: {losses.item():.4f}")
    
    return total_loss / len(data_loader)

# ===== MAIN TRAINING LOOP =====
if __name__ == "__main__":
    print("=" * 50)
    print("TOMATO MASK R-CNN TRAINING")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading training dataset...")
    train_dataset = TomatoDataset(TRAIN_IMAGES, TRAIN_ANN)
    print("\nLoading validation dataset...")
    val_dataset = TomatoDataset(VAL_IMAGES, VAL_ANN)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    # Load model
    print("\nLoading pre-trained Mask R-CNN model...")
    model = get_model(num_classes=train_dataset.num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Train for 5 epochs (we can do more later if needed)
    NUM_EPOCHS = 5
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 50)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        # Train
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Epoch {epoch}/{NUM_EPOCHS} completed in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}")
        print("-" * 50)
    
    # Save the model
    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print("\n🎉 TRAINING COMPLETE!")
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Class mapping: {train_dataset.cat_id_to_label}")