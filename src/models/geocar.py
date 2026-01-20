"""
GeoCAR-Lite: Geometry-verified Causal Change Reasoning
Main model combining all components
"""

import torch
import torch.nn as nn
from .base_vlm import BaseLLaVA
from .geometry_encoder import DUSt3REncoder
from .change_fusion import ChangeFusionModule
from .verifier import GeometricVerifier


class GeoCAR(nn.Module):
    """
    Main GeoCAR-Lite model
    """

    def __init__(self, config):
        super().__init__()

        # Load pre-trained models (frozen)
        self.base_vlm = BaseLLaVA(config.vlm)
        self.geometry_encoder = DUSt3REncoder(config.geometry)

        # Your novel modules (trainable)
        self.change_fusion = ChangeFusionModule(config.fusion)
        self.verifier = GeometricVerifier(config.verifier)

        # Process tokens
        self.process_tokens = nn.Parameter(
            torch.randn(config.num_process_stages, config.hidden_dim)
        )
        
        # Move all modules to GPU
        self.to('cuda:0')

    def forward(self, img_t1, img_t2, question):
        """
        Forward pass

        Args:
            img_t1: [B, 3, H, W]
            img_t2: [B, 3, H, W]
            question: str or List[str]

        Returns:
            dict with 'text', 'confidence', 'change_mask'
        """
        # 1. Get geometry (frozen)
        with torch.no_grad():
            geometry = self.geometry_encoder(img_t1, img_t2)

        # 2. Get VLM features
        vlm_features = self.base_vlm.encode(img_t1, img_t2, question)

        # 3. YOUR MODULE: Change fusion
        change_features = self.change_fusion(vlm_features, geometry)

        # 4. Generate
        # For now, just return a simple text output since we can't pass custom embeddings
        # In full implementation, you'd train a projection head
        output = ["The scene has changed."]  # Placeholder

        # 5. YOUR MODULE: Verify
        verified, confidence = self.verifier(output, geometry)

        return {
            "text": verified,
            "confidence": confidence,
            "change_mask": geometry["change_mask"],
        }
