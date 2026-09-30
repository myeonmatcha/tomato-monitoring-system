import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import requests
from io import BytesIO

print("Loading pre-trained Mask R-CNN model...")
# Load the pre-trained model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval() # Set to evaluation mode

# Download a sample image of tomatoes from the internet
url = "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
print(f"Downloading test image...")
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# Preprocess the image (convert to tensor)
transform = T.Compose([T.ToTensor()])
img_tensor = transform(image)

print("Running AI inference... (This might take a few seconds)")
# Run the model
with torch.no_grad():
    prediction = model([img_tensor])

print("Drawing results...")
# Visualize the results
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Draw the bounding boxes
for i in range(len(prediction[0]['scores'])):
    score = prediction[0]['scores'][i].item()
    if score > 0.7: # Only show predictions with >70% confidence
        box = prediction[0]['boxes'][i].cpu().numpy()
        label = prediction[0]['labels'][i].item()
        
        # Draw box
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"Class {label} ({score:.2f})", color='red', fontsize=12, weight='bold')

plt.axis('off')
plt.savefig("ai_test_result.jpg")
print("Test Complete! Image saved as ai_test_result.jpg")

print("Test Complete! A window should have popped up showing the detected objects.")