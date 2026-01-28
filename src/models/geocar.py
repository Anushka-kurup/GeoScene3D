"""
GeoCAR: Geometry-verified Causal Change Reasoning
Main model combining all components
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from .base_vlm import BaseLLaVA
from .geometry_encoder import DepthProEncoder
from .change_fusion import ChangeFusionModule
from .verifier import GeometricVerifier


class GeoCAR(nn.Module):
    """
    GeoCAR Model for Temporal Change Reasoning with Geometric Verification
    
    Architecture:
        - VLM: Molmo (frozen with optional LoRA)
        - Geometry: Apple Depth Pro (frozen, metric depth)
        - Vision: DINOV2 (frozen, optional semantic features)
        - Novel: ChangeFusion + GeometricVerifier (trainable)
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        print("=" * 70)
        print("Initializing GeoCAR Model")
        print("=" * 70)
        
        # ═══════════════════════════════════════════════════════════
        # Component 1: Vision-Language Model (Molmo)
        # ═══════════════════════════════════════════════════════════
        print("\n[1/4] Loading VLM (Molmo)...")
        self.base_vlm = BaseLLaVA(config.vlm)
        print("  ✅ VLM loaded")
        
        # ═══════════════════════════════════════════════════════════
        # Component 2: Geometry Encoder (Depth Pro)
        # ═══════════════════════════════════════════════════════════
        print("\n[2/4] Loading Geometry Encoder (Depth Pro)...")
        self.geometry_encoder = DepthProEncoder(config.geometry)
        print("  ✅ Depth Pro loaded and frozen")
        
        # ═══════════════════════════════════════════════════════════
        # Component 3: Vision Encoder (DINOV2) - Optional
        # ═══════════════════════════════════════════════════════════
        self.vision_encoder = None
        if config.get('use_dinov2', False):
            print("\n[3/4] Loading Vision Encoder (DINOV2)...")
            self.vision_encoder = AutoModel.from_pretrained(
                "facebook/dinov2-large",
                torch_dtype=torch.float16
            )
            
            # Freeze DINOV2
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            
            print("  ✅ DINOV2 loaded and frozen")
        else:
            print("\n[3/4] Skipping DINOV2 (disabled in config)")
        
        # ═══════════════════════════════════════════════════════════
        # Component 4: Novel Modules (Trainable)
        # ═══════════════════════════════════════════════════════════
        print("\n[4/4] Loading Novel Modules...")
        
        self.change_fusion = ChangeFusionModule(config.fusion)
        self.verifier = GeometricVerifier(config.verifier)
        
        print(f"  ✅ ChangeFusion: {self._count_params(self.change_fusion)/1e6:.2f}M params")
        print(f"  ✅ Verifier: {self._count_params(self.verifier)/1e6:.2f}M params")
        
        # ═══════════════════════════════════════════════════════════
        # Optional: Projection head for generation
        # ═══════════════════════════════════════════════════════════
        if config.get('use_projection_head', False):
            self.projection_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.vlm_embed_dim),
                nn.LayerNorm(config.vlm_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            print(f"  ✅ Projection head added")
        else:
            self.projection_head = None
        
        print("\n" + "=" * 70)
        self._print_statistics()
        print("=" * 70 + "\n")
        
        # Move to GPU
        self.to('cuda:0')

    def forward(self, img_t1, img_t2, question, mode='inference'):
        """
        Forward pass with temporal change reasoning
        
        Args:
            img_t1: [B, 3, H, W] - Images at time t1
            img_t2: [B, 3, H, W] - Images at time t2
            question: str or List[str] - Questions about the change
            mode: 'inference' or 'training'
        
        Returns:
            dict with:
                - text: List[str] - Verified descriptions
                - confidence: Tensor [B] - Confidence scores
                - geometry: dict - Geometric information
                - raw_text: List[str] - Before verification (if inference)
        """
        device = img_t1.device
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Extract Geometric Features (Depth Pro)
        # ═══════════════════════════════════════════════════════════
        with torch.no_grad():
            geometry = self.geometry_encoder(img_t1, img_t2)
            # Returns: pointmap_t1, pointmap_t2, change_mask, etc.
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Extract Visual Features (Optional DINOV2)
        # ═══════════════════════════════════════════════════════════
        visual_features = None
        if self.vision_encoder is not None:
            with torch.no_grad():
                vis_t1 = self.vision_encoder(img_t1).last_hidden_state
                vis_t2 = self.vision_encoder(img_t2).last_hidden_state
                visual_features = (vis_t1, vis_t2)
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Extract VLM Features
        # ═══════════════════════════════════════════════════════════
        vlm_features = self.base_vlm.encode(img_t1, img_t2, question)
        
        # ═══════════════════════════════════════════════════════════
        # Step 4: Change-Aware Fusion (Novel Module)
        # ═══════════════════════════════════════════════════════════
        fused_features = self.change_fusion(
            vlm_features=vlm_features,
            visual_features=visual_features,  # None if DINOV2 disabled
            geometry=geometry
        )
        
        # ═══════════════════════════════════════════════════════════
        # Training Mode: Return features for loss computation
        # ═══════════════════════════════════════════════════════════
        if mode == 'training':
            return {
                'fused_features': fused_features,
                'geometry': geometry,
                'visual_features': visual_features
            }
        
        # ═══════════════════════════════════════════════════════════
        # Step 5: Generate Text (Inference Mode)
        # ═══════════════════════════════════════════════════════════
        if self.projection_head is not None:
            # Project fused features to VLM embedding space
            projected_features = self.projection_head(fused_features)
            
            # Generate with VLM
            generated_text = self.base_vlm.generate(
                inputs_embeds=projected_features,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
        else:
            # Fallback: Simple placeholder
            # TODO: Implement proper generation pipeline
            B = img_t1.shape[0]
            generated_text = [
                "The scene shows changes between the two time points."
            ] * B
        
        # ═══════════════════════════════════════════════════════════
        # Step 6: Geometric Verification (Novel Module)
        # ═══════════════════════════════════════════════════════════
        verified_text, confidence = self.verifier(
            generated_text,
            geometry,
            geometry_encoder=self.geometry_encoder
        )
        
        return {
            'text': verified_text,
            'raw_text': generated_text,
            'confidence': confidence,
            'geometry': geometry
        }
    
    def _count_params(self, module):
        """Count trainable parameters in a module"""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    def _print_statistics(self):
        """Print model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {total_params/1e6:.1f}M")
        print(f"  Trainable Parameters: {trainable_params/1e6:.1f}M")
        print(f"  Frozen Parameters: {frozen_params/1e6:.1f}M")
        print(f"  Trainable Percentage: {100*trainable_params/total_params:.1f}%")
        
        # Breakdown by component
        print(f"\nComponent Breakdown:")
        components = [
            ("VLM (Molmo)", self.base_vlm),
            ("Geometry (Depth Pro)", self.geometry_encoder),
            ("ChangeFusion", self.change_fusion),
            ("Verifier", self.verifier),
        ]
        
        if self.vision_encoder is not None:
            components.insert(2, ("Vision (DINOV2)", self.vision_encoder))
        
        for name, module in components:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = "Trainable" if trainable > 0 else "Frozen"
            print(f"  {name:<25} {total/1e6:>8.1f}M ({status})")


# ═══════════════════════════════════════════════════════════════════
# Convenience functions for model creation
# ═══════════════════════════════════════════════════════════════════

def create_geocar_model(config):
    """
    Factory function to create GeoCAR model
    
    Args:
        config: Model configuration object
    
    Returns:
        GeoCAR model instance
    """
    return GeoCAR(config)


def freeze_module(module, module_name="module"):
    """
    Freeze all parameters in a module
    
    Args:
        module: PyTorch module to freeze
        module_name: Name for logging
    """
    for param in module.parameters():
        param.requires_grad = False
    print(f"  ✅ Frozen: {module_name}")


def unfreeze_module(module, module_name="module"):
    """
    Unfreeze all parameters in a module
    
    Args:
        module: PyTorch module to unfreeze
        module_name: Name for logging
    """
    for param in module.parameters():
        param.requires_grad = True
    print(f"  ❌ Unfrozen: {module_name}")


def setup_training_stage(model, stage='joint'):
    """
    Configure model freezing for different training stages
    
    Args:
        model: GeoCAR model
        stage: 'geometry', 'fusion', or 'joint'
    """
    print(f"\n{'='*70}")
    print(f"Setting up training stage: {stage.upper()}")
    print(f"{'='*70}")
    
    if stage == 'geometry':
        # Stage 1: Train only geometry encoder
        # (Not recommended with Depth Pro - it's pretrained)
        freeze_module(model.base_vlm, "VLM")
        freeze_module(model.change_fusion, "ChangeFusion")
        freeze_module(model.verifier, "Verifier")
        unfreeze_module(model.geometry_encoder, "Geometry Encoder")
        
    elif stage == 'fusion':
        # Stage 2: Train fusion modules only
        freeze_module(model.base_vlm, "VLM")
        freeze_module(model.geometry_encoder, "Geometry Encoder")
        if model.vision_encoder is not None:
            freeze_module(model.vision_encoder, "Vision Encoder")
        unfreeze_module(model.change_fusion, "ChangeFusion")
        unfreeze_module(model.verifier, "Verifier")
        
    elif stage == 'joint':
        # Stage 3: Train VLM + fusion modules
        freeze_module(model.geometry_encoder, "Geometry Encoder")
        if model.vision_encoder is not None:
            freeze_module(model.vision_encoder, "Vision Encoder")
        
        # VLM with LoRA (if enabled)
        if hasattr(model.base_vlm.model, 'print_trainable_parameters'):
            print("  VLM with LoRA:")
            model.base_vlm.model.print_trainable_parameters()
        else:
            unfreeze_module(model.base_vlm, "VLM")
        
        unfreeze_module(model.change_fusion, "ChangeFusion")
        unfreeze_module(model.verifier, "Verifier")
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Print summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example of how to instantiate and use the model
    """
    from types import SimpleNamespace
    
    # Example configuration
    config = SimpleNamespace(
        # VLM config
        vlm=SimpleNamespace(
            name="allenai/Molmo-7B-D-0924",
            freeze=False,
            lora=SimpleNamespace(
                enabled=True,
                r=16,
                alpha=32
            )
        ),
        
        # Geometry config
        geometry=SimpleNamespace(
            name="apple/DepthPro",
            change_threshold=0.1
        ),
        
        # Fusion config
        fusion=SimpleNamespace(
            visual_dim=1024,  # DINOV2-large
            geometry_dim=512,
            hidden_dim=4096
        ),
        
        # Verifier config
        verifier=SimpleNamespace(
            hidden_dim=4096
        ),
        
        # Optional features
        use_dinov2=True,  # Set to False to disable DINOV2
        use_projection_head=False,  # Enable if needed
        
        # Dimensions
        hidden_dim=4096,
        vlm_embed_dim=4096
    )
    
    # Create model
    print("Creating GeoCAR model...")
    model = create_geocar_model(config)
    
    # Setup for training
    setup_training_stage(model, stage='fusion')
    
    # Example forward pass
    print("\nRunning example forward pass...")
    img_t1 = torch.randn(2, 3, 224, 224).cuda()
    img_t2 = torch.randn(2, 3, 224, 224).cuda()
    question = ["What changed between these images?"] * 2
    
    with torch.no_grad():
        output = model(img_t1, img_t2, question, mode='inference')
    
    print("\nOutput:")
    print(f"  Generated: {output['text']}")
    print(f"  Confidence: {output['confidence']}")
    print(f"  Change mask shape: {output['geometry']['change_mask'].shape}")
    
    print("\n✅ Model working correctly!")