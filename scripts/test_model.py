import sys
sys.path.insert(0, '/workspace/GeoScene3D/dust3r')
sys.path.append('src')

import torch
from omegaconf import OmegaConf
from models.geocar import GeoCAR

# Load config
config = OmegaConf.load('configs/base_config.yaml')

# Create model
# Create model
model = GeoCAR(config.model)
model.eval()  # Set to evaluation mode

# Test forward pass
B = 1
img_t1 = torch.randn(B, 3, 224, 224)
img_t2 = torch.randn(B, 3, 224, 224)
question = "What changed?"

with torch.no_grad():
    output = model(img_t1, img_t2, question)

print("Output:", output['text'])
print("Confidence:", output['confidence'])
print("Model works!")
