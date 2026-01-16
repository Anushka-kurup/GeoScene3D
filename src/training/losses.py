"""
Loss functions for GeoCAR training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoCarLoss(nn.Module):
    """
    Combined loss for GeoCAR training
    
    Components:
    1. Language modeling loss (standard VLM)
    2. Verification loss (confidence prediction)
    3. Change attention loss (focus on changed regions)
    4. Process ordering loss (temporal consistency)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.lm_weight = config.get('lm_weight', 1.0)
        self.verification_weight = config.get('verification_weight', 0.2)
        self.change_weight = config.get('change_weight', 0.1)
        self.process_weight = config.get('process_weight', 0.1)
    
    def forward(self, outputs, targets):
        """
        Compute combined loss
        
        Args:
            outputs: dict from model forward()
            targets: dict with ground truth
        
        Returns:
            loss: scalar tensor
            loss_dict: dict with individual losses
        """
        losses = {}
        
        # 1. Language modeling loss (from VLM)
        if outputs['loss'] is not None:
            losses['lm_loss'] = outputs['loss']
        else:
            losses['lm_loss'] = torch.tensor(0.0, device=outputs['change_mask'].device)
        
        # 2. Verification loss
        if outputs['confidence'] is not None and 'gt_plausibility' in targets:
            losses['verification_loss'] = F.binary_cross_entropy(
                outputs['confidence'],
                targets['gt_plausibility']
            )
        else:
            losses['verification_loss'] = torch.tensor(0.0, device=outputs['change_mask'].device)
        
        # 3. Change attention loss
        if 'process_attention' in outputs and 'change_mask' in outputs:
            losses['change_loss'] = self.compute_change_attention_loss(
                outputs['process_attention'],
                outputs['change_mask']
            )
        else:
            losses['change_loss'] = torch.tensor(0.0, device=outputs['change_mask'].device)
        
        # 4. Process ordering loss
        if 'process_stages' in targets:
            losses['process_loss'] = self.compute_process_ordering_loss(
                outputs.get('process_attention'),
                targets['process_stages']
            )
        else:
            losses['process_loss'] = torch.tensor(0.0, device=outputs['change_mask'].device)
        
        # Combined loss
        total_loss = (
            self.lm_weight * losses['lm_loss'] +
            self.verification_weight * losses['verification_loss'] +
            self.change_weight * losses['change_loss'] +
            self.process_weight * losses['process_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses
    
    def compute_change_attention_loss(self, attention_weights, change_mask):
        """
        Encourage attention to focus on changed regions
        
        Args:
            attention_weights: [B, num_stages, N]
            change_mask: [B, H, W]
        
        Returns:
            loss: scalar
        """
        B, num_stages, N = attention_weights.shape
        H, W = change_mask.shape[1:]
        
        # Resize change mask to match attention
        change_mask_flat = F.adaptive_avg_pool2d(
            change_mask.unsqueeze(1),
            output_size=(int(N**0.5), int(N**0.5))
        ).flatten(2)  # [B, 1, N]
        
        # Normalize
        change_mask_norm = change_mask_flat / (change_mask_flat.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Average attention across stages
        avg_attention = attention_weights.mean(dim=1, keepdim=True)  # [B, 1, N]
        
        # KL divergence between attention and change mask
        loss = F.kl_div(
            (avg_attention + 1e-6).log(),
            change_mask_norm + 1e-6,
            reduction='batchmean'
        )
        
        return loss
    
    def compute_process_ordering_loss(self, attention_weights, process_stages):
        """
        Encourage temporal ordering of process stages
        
        Args:
            attention_weights: [B, num_stages, N]
            process_stages: List[List[str]] - list of stages per sample
        
        Returns:
            loss: scalar
        """
        if attention_weights is None:
            return torch.tensor(0.0)
        
        # Simple ordering loss: later stages should have higher attention
        # on t2 features vs t1 features
        
        # This is a placeholder - in practice, you'd implement
        # more sophisticated temporal ordering constraints
        
        return torch.tensor(0.0, device=attention_weights.device)


def compute_gt_plausibility(descriptions, constraints):
    """
    Compute ground truth plausibility scores
    
    Args:
        descriptions: List[str]
        constraints: List[dict]
    
    Returns:
        plausibility: [B] tensor
    """
    # In practice, you'd have human annotations
    # For now, assume all GT is plausible
    B = len(descriptions)
    return torch.ones(B)