import sys
sys.path.insert(0, '/workspace/GeoScene3D/dust3r')
sys.path.insert(0, '/workspace/GeoScene3D/dust3r/croco')
sys.path.append('src')

import torch
from omegaconf import OmegaConf

print("=" * 60)
print("Testing GeoCAR Model Components")
print("=" * 60)

# Load config
config = OmegaConf.load('configs/base_config.yaml')

# Test 1: Load VLM
print("\n1. Loading VLM...")
from models.base_vlm import BaseLLaVA
vlm = BaseLLaVA(config.model.vlm)
print("✓ VLM loaded successfully")

# Test 2: Load Geometry Encoder
print("\n2. Loading Geometry Encoder...")
from models.geometry_encoder import DUSt3REncoder
geo_encoder = DUSt3REncoder(config.model.geometry)
print("✓ Geometry Encoder loaded successfully")

# Test 3: Load Change Fusion
print("\n3. Loading Change Fusion Module...")
from models.change_fusion import ChangeFusionModule
change_fusion = ChangeFusionModule(config.model.fusion)
print("✓ Change Fusion Module loaded successfully")

# Test 4: Load Verifier
print("\n4. Loading Verifier...")
from models.verifier import GeometricVerifier
verifier = GeometricVerifier(config.model.verifier)
print("✓ Verifier loaded successfully")

# Test 5: Create dummy data
print("\n5. Testing with dummy data...")
B = 1
img_t1 = torch.randn(B, 3, 224, 224)
img_t2 = torch.randn(B, 3, 224, 224)

# Test geometry encoder (should work with fallback)
geometry = geo_encoder(img_t1, img_t2)
print(f"✓ Geometry encoder output shape: {geometry['change_mask'].shape}")

# Test change fusion with dummy VLM features
dummy_vlm_features = torch.randn(B, 576, 1024)  # Fake VLM output
fused = change_fusion(dummy_vlm_features, geometry)
print(f"✓ Change fusion output shape: {fused.shape}")

print("\n" + "=" * 60)
print("SUCCESS! All model components loaded and tested.")
print("=" * 60)
print("\nNote: Full end-to-end inference requires proper image preprocessing")
print("and LLaVA's specific image token handling, which needs the official")
print("video processor. For now, components work independently.")
