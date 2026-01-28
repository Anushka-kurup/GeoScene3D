"""
Geometry encoder wrapper for Apple Depth Pro
Extracts metric depth maps and change masks
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T


class DepthProEncoder(nn.Module):
    """
    Wrapper around Apple Depth Pro for metric depth estimation
    
    Replaces DUSt3R with a faster, simpler alternative that still
    provides metric depth in meters for geometric verification.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print("Loading Apple Depth Pro...")
        
        try:
            import depth_pro
            
            # Load model and preprocessing
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model.eval()
            self.model = self.model.to('cuda:0')
            
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            print("✅ Depth Pro loaded and frozen")
            print(f"   Change threshold: {config.change_threshold}m")
            
        except ImportError:
            raise ImportError(
                "depth_pro not found. Install with:\n"
                "pip install git+https://github.com/apple/ml-depth-pro.git"
            )
    
    @torch.no_grad()
    def forward(self, img_t1, img_t2):
        """
        Extract metric depth for both images
        
        Args:
            img_t1: [B, 3, H, W] - Images at time t1
            img_t2: [B, 3, H, W] - Images at time t2
        
        Returns:
            dict with:
                - pointmap_t1: [B, H, W, 3] - 3D points (x, y, depth)
                - pointmap_t2: [B, H, W, 3] - 3D points (x, y, depth)
                - confidence_t1: [B, H, W] - Confidence scores
                - confidence_t2: [B, H, W] - Confidence scores
                - change_mask: [B, H, W] - Binary change mask
                - change_magnitude: [B, H, W] - Change magnitude in meters
        """
        B = img_t1.shape[0]
        device = img_t1.device
        
        depth_t1_list = []
        depth_t2_list = []
        focal_t1_list = []
        focal_t2_list = []
        
        # Process each image in batch
        for b in range(B):
            # Convert to PIL
            img1_pil = self._to_pil(img_t1[b])
            img2_pil = self._to_pil(img_t2[b])
            
            # Run Depth Pro inference
            pred1 = self.model.infer(self.transform(img1_pil))
            pred2 = self.model.infer(self.transform(img2_pil))
            
            # Extract depth (in meters!)
            depth_t1_list.append(pred1['depth'])
            depth_t2_list.append(pred2['depth'])
            
            # Extract focal length (optional, for completeness)
            focal_t1_list.append(pred1.get('focallength_px', None))
            focal_t2_list.append(pred2.get('focallength_px', None))
        
        # Stack into batch tensors
        depth_t1 = torch.stack(depth_t1_list).to(device)  # [B, H, W]
        depth_t2 = torch.stack(depth_t2_list).to(device)  # [B, H, W]
        
        H, W = depth_t1.shape[1:3]
        
        # Create 3D point maps (x, y, depth) format
        # This matches DUSt3R output format for compatibility
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Expand to batch
        x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
        y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
        
        # Stack: [x, y, depth]
        pointmap_t1 = torch.stack([
            x_coords,
            y_coords,
            depth_t1  # Metric depth in meters!
        ], dim=-1)  # [B, H, W, 3]
        
        pointmap_t2 = torch.stack([
            x_coords,
            y_coords,
            depth_t2
        ], dim=-1)
        
        # Compute change magnitude (in meters)
        change_magnitude = torch.abs(depth_t2 - depth_t1)
        
        # Create binary change mask
        # Threshold is configurable (default: 0.1m = 10cm)
        change_threshold = getattr(self.config, 'change_threshold', 0.1)
        change_mask = (change_magnitude > change_threshold).float()
        
        # Create confidence maps (Depth Pro doesn't provide these,
        # so we use uniform confidence)
        confidence_t1 = torch.ones_like(depth_t1)
        confidence_t2 = torch.ones_like(depth_t2)
        
        return {
            'pointmap_t1': pointmap_t1,
            'pointmap_t2': pointmap_t2,
            'confidence_t1': confidence_t1,
            'confidence_t2': confidence_t2,
            'change_mask': change_mask,
            'change_magnitude': change_magnitude,
            # Additional metadata
            'depth_t1': depth_t1,
            'depth_t2': depth_t2,
            'focal_length_t1': focal_t1_list,
            'focal_length_t2': focal_t2_list,
        }
    
    def compute_metrics(self, geometry):
        """
        Compute geometric metrics from depth maps
        
        Args:
            geometry: dict from forward()
        
        Returns:
            dict with height_change, volume_change, area_change
        """
        pointmap_t1 = geometry['pointmap_t1']  # [B, H, W, 3]
        pointmap_t2 = geometry['pointmap_t2']
        change_mask = geometry['change_mask']  # [B, H, W]
        
        # Get changed regions
        changed_t1 = pointmap_t1[change_mask > 0.5]
        changed_t2 = pointmap_t2[change_mask > 0.5]
        
        if len(changed_t1) == 0:
            return {
                'height_change': 0.0,
                'volume_change': 0.0,
                'area_change': 0.0
            }
        
        # Height change (max depth difference in meters)
        height_change = (changed_t2[:, 2].max() - changed_t1[:, 2].max()).item()
        
        # Approximate volume (sum of depth changes × pixel area)
        # Note: This is a rough approximation
        volume_change = geometry['change_magnitude'][change_mask > 0.5].sum().item() * 0.01
        
        # Area of change (number of changed pixels × pixel area)
        area_change = (change_mask > 0.5).sum().item() * 0.01
        
        return {
            'height_change': float(height_change),
            'volume_change': float(volume_change),
            'area_change': float(area_change)
        }
    
    def _to_pil(self, img_tensor):
        """
        Convert tensor to PIL Image
        
        Args:
            img_tensor: [3, H, W] tensor
        
        Returns:
            PIL Image
        """
        if isinstance(img_tensor, Image.Image):
            return img_tensor
        
        to_pil = T.ToPILImage()
        return to_pil(img_tensor.cpu())