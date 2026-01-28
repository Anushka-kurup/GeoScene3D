"""
Change-aware fusion module
YOUR NOVEL CONTRIBUTION #1

Fuses VLM features with geometric change information,
optionally incorporating semantic visual features from DINOV2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeFusionModule(nn.Module):
    """
    Fuses visual features with geometric change information
    
    Key innovations:
    1. Change-aware attention (focuses on changed regions)
    2. Geometric delta encoding (models change, not states)
    3. Multi-modal fusion (VLM + geometry + optional vision)
    4. Temporal representation learning
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.visual_dim = getattr(config, 'visual_dim', 1024)  # DINOV2-large
        self.geometry_dim = getattr(config, 'geometry_dim', 512)
        self.hidden_dim = config.hidden_dim  # 4096 for LLM
        
        # ═══════════════════════════════════════════════════
        # Step 1: Project 3D points to feature space
        # ═══════════════════════════════════════════════════
        self.geo_projection = nn.Sequential(
            nn.Linear(3, 128),  # 3D point (x, y, depth) → feature
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.geometry_dim),
            nn.ReLU(),
            nn.LayerNorm(self.geometry_dim),
        )
        
        # ═══════════════════════════════════════════════════
        # Step 2: Change-aware cross attention
        # ═══════════════════════════════════════════════════
        self.change_attention = nn.MultiheadAttention(
            embed_dim=self.geometry_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # ═══════════════════════════════════════════════════
        # Step 3: Visual feature projection (for DINOV2)
        # ═══════════════════════════════════════════════════
        self.visual_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ═══════════════════════════════════════════════════
        # Step 4: Multi-modal fusion
        # ═══════════════════════════════════════════════════
        # Calculate input dimension based on what's provided
        # Base: VLM + geometry
        # Optional: + visual features from DINOV2
        self.base_fusion_dim = self.hidden_dim + self.geometry_dim
        self.visual_fusion_dim = self.base_fusion_dim + (self.hidden_dim // 2)
        
        # Two fusion paths depending on visual features availability
        self.fusion_mlp_base = nn.Sequential(
            nn.Linear(self.base_fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        self.fusion_mlp_with_visual = nn.Sequential(
            nn.Linear(self.visual_fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # ═══════════════════════════════════════════════════
        # Step 5: Learnable change type embeddings
        # ═══════════════════════════════════════════════════
        self.change_type_embed = nn.Embedding(
            num_embeddings=4,  # additive, subtractive, transformative, none
            embedding_dim=self.geometry_dim,
        )
        
        # ═══════════════════════════════════════════════════
        # Step 6: Output projection
        # ═══════════════════════════════════════════════════
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        print(f"ChangeFusionModule initialized: {self.count_parameters() / 1e6:.2f}M params")
    
    def forward(self, vlm_features, geometry, visual_features=None):
        """
        Fuse VLM features with geometric change information
        
        Args:
            vlm_features: [B, N, hidden_dim] from VLM encoder
            geometry: dict with pointmaps and masks
            visual_features: Optional tuple (vis_t1, vis_t2) from DINOV2
                           Each [B, N_patches, visual_dim]
        
        Returns:
            fused_features: [B, N_out, hidden_dim]
        """
        B, N, D = vlm_features.shape
        device = vlm_features.device
        
        # ═══════════════════════════════════════════════════
        # Step 1: Extract and project geometry
        # ═══════════════════════════════════════════════════
        pointmap_t1 = geometry['pointmap_t1']  # [B, H, W, 3]
        pointmap_t2 = geometry['pointmap_t2']
        change_mask = geometry['change_mask']  # [B, H, W]
        
        H, W = pointmap_t1.shape[1:3]
        
        # Downsample to reduce memory (e.g., 224x224 -> 28x28)
        pool_size = 28
        
        points_t1_pooled = F.adaptive_avg_pool2d(
            pointmap_t1.permute(0, 3, 1, 2),  # [B, 3, H, W]
            (pool_size, pool_size)
        ).permute(0, 2, 3, 1)  # [B, pool_size, pool_size, 3]
        
        points_t2_pooled = F.adaptive_avg_pool2d(
            pointmap_t2.permute(0, 3, 1, 2),
            (pool_size, pool_size)
        ).permute(0, 2, 3, 1)
        
        change_mask_pooled = F.adaptive_avg_pool2d(
            change_mask.unsqueeze(1),  # [B, 1, H, W]
            (pool_size, pool_size)
        ).squeeze(1)  # [B, pool_size, pool_size]
        
        # Flatten spatial dimensions
        points_t1 = points_t1_pooled.reshape(B, pool_size * pool_size, 3)
        points_t2 = points_t2_pooled.reshape(B, pool_size * pool_size, 3)
        change_mask_flat = change_mask_pooled.reshape(B, pool_size * pool_size)
        
        # Project to feature space
        geo_feat_t1 = self.geo_projection(points_t1)  # [B, pool_size^2, geo_dim]
        geo_feat_t2 = self.geo_projection(points_t2)
        
        # ═══════════════════════════════════════════════════
        # Step 2: Change-aware attention
        # ═══════════════════════════════════════════════════
        change_weights = change_mask_flat.unsqueeze(-1)  # [B, pool_size^2, 1]
        
        # Query: current state, Key/Value: previous state
        geo_change, attn_weights = self.change_attention(
            query=geo_feat_t2,
            key=geo_feat_t1,
            value=geo_feat_t1
        )
        
        # Weight by change mask (focus on changed regions)
        geo_change_weighted = geo_change * change_weights
        
        # ═══════════════════════════════════════════════════
        # Step 3: Compute geometric delta
        # ═══════════════════════════════════════════════════
        geo_delta = geo_feat_t2 - geo_feat_t1  # [B, pool_size^2, geo_dim]
        
        # Combine weighted attention and delta
        geo_combined = geo_change_weighted + 0.5 * geo_delta
        
        # ═══════════════════════════════════════════════════
        # Step 4: Pool geometry to match VLM token count
        # ═══════════════════════════════════════════════════
        geo_pooled = F.adaptive_avg_pool1d(
            geo_combined.transpose(1, 2),  # [B, geo_dim, pool_size^2]
            output_size=N,
        ).transpose(1, 2)  # [B, N, geo_dim]
        
        # ═══════════════════════════════════════════════════
        # Step 5: Detect change type and add embedding
        # ═══════════════════════════════════════════════════
        change_types = self.detect_change_type(geometry)  # [B]
        change_embeds = self.change_type_embed(change_types)  # [B, geo_dim]
        
        # Add to pooled features
        geo_pooled = geo_pooled + change_embeds.unsqueeze(1)
        
        # ═══════════════════════════════════════════════════
        # Step 6: Process visual features (if provided)
        # ═══════════════════════════════════════════════════
        if visual_features is not None:
            vis_t1, vis_t2 = visual_features
            
            # Project visual features
            vis_t1_proj = self.visual_projection(vis_t1)  # [B, N_patches, hidden//2]
            vis_t2_proj = self.visual_projection(vis_t2)
            
            # Temporal difference encoding
            vis_change = vis_t2_proj - vis_t1_proj
            
            # Pool to match VLM token count if needed
            if vis_t1_proj.shape[1] != N:
                vis_combined = torch.cat([vis_t1_proj, vis_t2_proj, vis_change], dim=1)
                vis_pooled = F.adaptive_avg_pool1d(
                    vis_combined.transpose(1, 2),
                    output_size=N
                ).transpose(1, 2)
            else:
                vis_pooled = torch.cat([vis_t1_proj, vis_t2_proj, vis_change], dim=1)
                # Average over temporal dimension
                vis_pooled = vis_pooled.view(B, 3, N, -1).mean(dim=1)  # [B, N, hidden//2]
            
            # Concatenate: VLM + geometry + visual
            combined = torch.cat([vlm_features, geo_pooled, vis_pooled], dim=-1)
            fused = self.fusion_mlp_with_visual(combined)
        else:
            # Concatenate: VLM + geometry only
            combined = torch.cat([vlm_features, geo_pooled], dim=-1)
            fused = self.fusion_mlp_base(combined)
        
        # ═══════════════════════════════════════════════════
        # Step 7: Output projection
        # ═══════════════════════════════════════════════════
        output = self.output_projection(fused)  # [B, N, hidden_dim]
        
        return output
    
    def detect_change_type(self, geometry):
        """
        Detect type of change (additive, subtractive, transformative, none)
        
        Args:
            geometry: dict with change information
        
        Returns:
            change_types: [B] tensor of integers 0-3
        """
        B = geometry['pointmap_t1'].shape[0]
        device = geometry['pointmap_t1'].device
        
        # Simple heuristic based on height change
        change_types = []
        
        for b in range(B):
            # Get single sample
            mask_b = geometry['change_mask'][b]  # [H, W]
            
            if mask_b.sum() == 0:
                change_types.append(3)  # No change
                continue
            
            # Get depth changes in changed regions
            depth_t1 = geometry['pointmap_t1'][b, :, :, 2]  # [H, W]
            depth_t2 = geometry['pointmap_t2'][b, :, :, 2]
            
            changed_depth_t1 = depth_t1[mask_b > 0.5]
            changed_depth_t2 = depth_t2[mask_b > 0.5]
            
            # Compute average depth change
            depth_diff = (changed_depth_t2.mean() - changed_depth_t1.mean()).item()
            
            # Classify based on threshold
            if depth_diff > 0.5:  # More than 50cm increase
                change_types.append(0)  # Additive (construction)
            elif depth_diff < -0.5:  # More than 50cm decrease
                change_types.append(1)  # Subtractive (demolition)
            else:
                change_types.append(2)  # Transformative (modification)
        
        return torch.tensor(change_types, device=device)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)