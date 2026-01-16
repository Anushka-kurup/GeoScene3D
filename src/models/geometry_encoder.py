"""
Geometry encoder wrapper for DUSt3R
Extracts 3D point clouds and change masks
"""

import torch
import torch.nn as nn
import sys
import os

# Add dust3r to path if needed
dust3r_path = os.path.expanduser('~/GeoScene3D/dust3r')
if os.path.exists(dust3r_path) and dust3r_path not in sys.path:
    sys.path.insert(0, dust3r_path)

try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.image_pairs import make_pairs
    DUST3R_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DUSt3R: {e}")
    print("Falling back to simple geometry encoder")
    DUST3R_AVAILABLE = False


if DUST3R_AVAILABLE:
    class DUSt3REncoder(nn.Module):
        """
        Wrapper around DUSt3R for 3D reconstruction
        """
        def __init__(self, config):
            super().__init__()
            self.config = config
            
            print(f"Loading DUSt3R model: {config.name}")
            
            # Load pre-trained DUSt3R
            model_name = config.name
            
            # Check if it's a local path or HuggingFace model
            if os.path.exists(model_name):
                print(f"Loading from local checkpoint: {model_name}")
                self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            else:
                # Try HuggingFace
                print(f"Loading from HuggingFace: {model_name}")
                self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            
            self.model.eval()
            
            # Always freeze
            for param in self.model.parameters():
                param.requires_grad = False
            
            print("DUSt3R loaded and frozen")
        
        @torch.no_grad()
        def forward(self, img_t1, img_t2):
            """
            Extract 3D geometry from image pair
            
            Args:
                img_t1: [B, 3, H, W] tensors
                img_t2: [B, 3, H, W] tensors
            
            Returns:
                dict with pointmaps and change masks
            """
            device = next(self.model.parameters()).device
            B = img_t1.shape[0]
            
            # Process each sample in batch
            all_results = []
            
            for b in range(B):
                # Get single images
                img1 = img_t1[b]  # [3, H, W]
                img2 = img_t2[b]  # [3, H, W]
                
                # Convert to PIL for DUSt3R
                from PIL import Image
                import torchvision.transforms as T
                
                to_pil = T.ToPILImage()
                img1_pil = to_pil(img1.cpu())
                img2_pil = to_pil(img2.cpu())
                
                # Load images in DUSt3R format
                from dust3r.utils.image import load_images
                images = load_images([img1_pil, img2_pil], size=512)
                
                # Create pairs
                pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                
                # Run inference
                output = inference(pairs, self.model, device, batch_size=1)
                
                # Extract point clouds
                pred1 = output['pred1']
                pred2 = output['pred2']
                
                # Get first pair (since we symmetrize, take first)
                pointmap_t1 = pred1['pts3d'][0]  # [H, W, 3]
                pointmap_t2 = pred2['pts3d'][0]  # [H, W, 3]
                confidence_t1 = pred1['conf'][0]  # [H, W]
                confidence_t2 = pred2['conf'][0]  # [H, W]
                
                all_results.append({
                    'pointmap_t1': pointmap_t1,
                    'pointmap_t2': pointmap_t2,
                    'confidence_t1': confidence_t1,
                    'confidence_t2': confidence_t2
                })
            
            # Stack batch results
            pointmap_t1 = torch.stack([r['pointmap_t1'] for r in all_results])
            pointmap_t2 = torch.stack([r['pointmap_t2'] for r in all_results])
            confidence_t1 = torch.stack([r['confidence_t1'] for r in all_results])
            confidence_t2 = torch.stack([r['confidence_t2'] for r in all_results])
            
            # Compute change mask
            change_magnitude = torch.norm(
                pointmap_t2 - pointmap_t1, 
                dim=-1
            )  # [B, H, W]
            
            # Threshold to get binary mask
            change_mask = (change_magnitude > self.config.change_threshold).float()
            
            # Weight by confidence
            avg_confidence = (confidence_t1 + confidence_t2) / 2
            change_mask = change_mask * avg_confidence
            
            return {
                'pointmap_t1': pointmap_t1,
                'pointmap_t2': pointmap_t2,
                'confidence_t1': confidence_t1,
                'confidence_t2': confidence_t2,
                'change_mask': change_mask,
                'change_magnitude': change_magnitude
            }
        
        def compute_metrics(self, geometry):
            """
            Compute geometric metrics from point clouds
            """
            pointmap_t1 = geometry['pointmap_t1']
            pointmap_t2 = geometry['pointmap_t2']
            change_mask = geometry['change_mask']
            
            # Get changed points
            changed_t1 = pointmap_t1[change_mask > 0.5]
            changed_t2 = pointmap_t2[change_mask > 0.5]
            
            if len(changed_t1) == 0:
                return {
                    'height_change': 0.0,
                    'volume_change': 0.0,
                    'area_change': 0.0
                }
            
            # Height change (max Z difference)
            height_change = (
                changed_t2[:, 2].max() - changed_t1[:, 2].max()
            ).item()
            
            # Approximate volume
            volume_change = geometry['change_magnitude'][change_mask > 0.5].sum().item() * 0.01
            
            # Area
            area_change = (change_mask > 0.5).sum().item() * 0.01
            
            return {
                'height_change': float(height_change),
                'volume_change': float(volume_change),
                'area_change': float(area_change)
            }

else:
    # Fallback to simple encoder
    class DUSt3REncoder(nn.Module):
        """Simple fallback encoder"""
        def __init__(self, config):
            super().__init__()
            self.config = config
            print("WARNING: Using SimpleGeometryEncoder fallback")
        
        @torch.no_grad()
        def forward(self, img_t1, img_t2):
            B, C, H, W = img_t1.shape
            device = img_t1.device
            
            # Simple mock geometry
            diff = torch.abs(img_t2 - img_t1).mean(dim=1)
            depth_t1 = img_t1.mean(dim=1)
            depth_t2 = img_t2.mean(dim=1)
            
            # Mock points
            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            
            pointmap_t1 = torch.stack([
                x.float().unsqueeze(0).expand(B, -1, -1) / W,
                y.float().unsqueeze(0).expand(B, -1, -1) / H,
                depth_t1
            ], dim=-1)
            
            pointmap_t2 = torch.stack([
                x.float().unsqueeze(0).expand(B, -1, -1) / W,
                y.float().unsqueeze(0).expand(B, -1, -1) / H,
                depth_t2
            ], dim=-1)
            
            change_mask = (diff > 0.1).float()
            
            return {
                'pointmap_t1': pointmap_t1,
                'pointmap_t2': pointmap_t2,
                'confidence_t1': torch.ones_like(depth_t1),
                'confidence_t2': torch.ones_like(depth_t2),
                'change_mask': change_mask,
                'change_magnitude': diff
            }
        
        def compute_metrics(self, geometry):
            return {
                'height_change': 0.5,
                'volume_change': 100.0,
                'area_change': 50.0
            }