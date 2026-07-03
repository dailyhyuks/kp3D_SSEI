"""Simple test to check MiDaS transform behavior."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image

print("\n" + "="*60)
print("MiDaS Transform Test")
print("="*60 + "\n")

# Load sample image
sample_path = Path(__file__).parent.parent / "samples" / "boat_painting_2.png"
img_pil = Image.open(sample_path).convert("RGB")
img_np = np.array(img_pil)
print(f"Original image shape: {img_np.shape}")

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
small_transform = midas_transforms.small_transform

# Apply transform
input_tensor = small_transform(img_np)
print(f"After transform shape: {input_tensor.shape}")
print(f"Transform adds batch dimension: {len(input_tensor.shape) == 4}")

# Test with model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
model.eval()

with torch.no_grad():
    # MiDaS transform already adds batch dimension, don't add another
    prediction = model(input_tensor)
    print(f"Model output shape: {prediction.shape}")

print("\n[OK] Transform test completed successfully!")
