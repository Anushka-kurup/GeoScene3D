"""
Change-aware fusion module
YOUR NOVEL CONTRIBUTION #1
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
    3. Multi-scale fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        visual_dim = config.visual_dim  # 1024 from CLIP
        geometry_dim = config.geometry_dim  # 512 for geometry features
        hidden_dim = config.hidden_dim  # 4096 for LLM

        # ═══════════════════════════════════════════════════
        # Step 1: Project geometry to feature space
        # ═══════════════════════════════════════════════════
        self.geo_projection = nn.Sequential(
            nn.Linear(3, 128),  # 3D point → feature
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, geometry_dim),
            nn.ReLU(),
            nn.LayerNorm(geometry_dim),
        )

        # ═══════════════════════════════════════════════════
        # Step 2: Change-aware cross attention
        # ═══════════════════════════════════════════════════
        self.change_attention = nn.MultiheadAttention(
            embed_dim=geometry_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # ═══════════════════════════════════════════════════
        # Step 3: Fusion with visual features
        # ═══════════════════════════════════════════════════
        self.fusion_mlp = nn.Sequential(
            nn.Linear(visual_dim + geometry_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ═══════════════════════════════════════════════════
        # Optional: Learnable change type embeddings
        # ═══════════════════════════════════════════════════
        self.change_type_embed = nn.Embedding(
            num_embeddings=4,  # additive, subtractive, transformative, none
            embedding_dim=geometry_dim,
        )
        
        # ═══════════════════════════════════════════════════
        # Projection head: hidden_dim -> visual_dim for generation
        # ═══════════════════════════════════════════════════
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, visual_dim),
        )

        print(
            f"ChangeFusionModule initialized: {self.count_parameters() / 1e6:.2f}M params"
        )

    def forward(self, vlm_features, geometry):
        """
        Fuse VLM features with geometric change information

        Args:
            vlm_features: [B, N, visual_dim] from VLM encoder
            geometry: dict with pointmaps and masks

        Returns:
            fused_features: [B, 2N, hidden_dim]
        """
        B, N, D = vlm_features.shape
        # device = vlm_features.device

        # ═══════════════════════════════════════════════════
        # Step 1: Extract and project geometry
        # ═══════════════════════════════════════════════════
        pointmap_t1 = geometry["pointmap_t1"]  # [B, H, W, 3]
        pointmap_t2 = geometry["pointmap_t2"]
        change_mask = geometry["change_mask"]  # [B, H, W]

        H, W = pointmap_t1.shape[1:3]

        # Downsample to reduce memory (224x224 -> 28x28)
        # Use adaptive pooling to reduce spatial dimensions
        import torch.nn.functional as F
        pool_size = 28  # Reduce from 224x224 to 28x28
        
        # Reshape for pooling: [B, H, W, 3] -> [B, 3, H, W]
        points_t1_pooled = F.adaptive_avg_pool2d(
            pointmap_t1.permute(0, 3, 1, 2), 
            (pool_size, pool_size)
        ).permute(0, 2, 3, 1)  # Back to [B, pool_size, pool_size, 3]
        
        points_t2_pooled = F.adaptive_avg_pool2d(
            pointmap_t2.permute(0, 3, 1, 2), 
            (pool_size, pool_size)
        ).permute(0, 2, 3, 1)
        
        change_mask_pooled = F.adaptive_avg_pool2d(
            change_mask.unsqueeze(1), 
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

        # Compute attention weighted by change mask
        change_weights = change_mask_flat.unsqueeze(-1)  # [B, H*W, 1]

        # Query: current state, Key/Value: previous state
        geo_change, attn_weights = self.change_attention(
            query=geo_feat_t2, key=geo_feat_t1, value=geo_feat_t1
        )

        # Weight by change mask (focus on changed regions)
        geo_change_weighted = geo_change * change_weights

        # ═══════════════════════════════════════════════════
        # Step 3: Compute geometric delta
        # ═══════════════════════════════════════════════════
        geo_delta = geo_feat_t2 - geo_feat_t1  # [B, H*W, geo_dim]

        # Combine weighted attention and delta
        geo_combined = geo_change_weighted + 0.5 * geo_delta

        # ═══════════════════════════════════════════════════
        # Step 4: Pool to match visual token count
        # ═══════════════════════════════════════════════════

        # Use adaptive pooling
        geo_pooled = F.adaptive_avg_pool1d(
            geo_combined.transpose(1, 2),  # [B, geo_dim, H*W]
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
        # Step 6: Fuse with visual features
        # ═══════════════════════════════════════════════════

        # Concatenate visual and geometric
        combined = torch.cat(
            [vlm_features, geo_pooled], dim=-1
        )  # [B, N, visual_dim + geo_dim]

        # Fuse through MLP
        fused = self.fusion_mlp(combined)  # [B, N, hidden_dim]

        # ═══════════════════════════════════════════════════
        # Step 7: Create temporal representation
        # ═══════════════════════════════════════════════════

        # Split into t1 and t2 (simple duplication for now)
        # In practice, you might want separate processing
        fused_t1 = fused
        fused_t2 = fused  # Could add temporal offset here

        # Concatenate temporal
        output = torch.cat([fused_t1, fused_t2], dim=1)  # [B, 2N, hidden_dim]

        return output

    def detect_change_type(self, geometry):
        """
        Detect type of change (additive, subtractive, etc.)

        Args:
            geometry: dict with change information

        Returns:
            change_types: [B] tensor of integers 0-3
        """
        B = geometry["pointmap_t1"].shape[0]

        # Simple heuristic based on height change
        metrics = []
        for b in range(B):
            geo_b = {k: v[b : b + 1] for k, v in geometry.items()}

            # Get height change
            changed_t1 = geo_b["pointmap_t1"][geo_b["change_mask"] > 0.5]
            changed_t2 = geo_b["pointmap_t2"][geo_b["change_mask"] > 0.5]

            if len(changed_t1) == 0:
                metrics.append(3)  # No change
            else:
                height_diff = (changed_t2[:, 2].mean() - changed_t1[:, 2].mean()).item()

                if height_diff > 0.5:
                    metrics.append(0)  # Additive
                elif height_diff < -0.5:
                    metrics.append(1)  # Subtractive
                else:
                    metrics.append(2)  # Transformative

        return torch.tensor(metrics, device=geometry["pointmap_t1"].device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
