import sys
sys.path.append('src')

import torch
from omegaconf import OmegaConf
from models.geocar import GeoCAR

# Load config
config = OmegaConf.load('configs/base_config.yaml')

# Create model
model = GeoCAR(config.model)

# Test forward pass
B = 1
img_t1 = torch.randn(B, 3, 224, 224)
img_t2 = torch.randn(B, 3, 224, 224)
question = "What changed?"

output = model(img_t1, img_t2, question)


print("Output:", output['text'])
print("Confidence:", output['confidence'])
print("Model works!")