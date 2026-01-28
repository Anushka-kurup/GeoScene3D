"""
Geometric verifier module
YOUR NOVEL CONTRIBUTION #2

Verifies VLM outputs against metric geometric constraints from Depth Pro
"""

import torch
import torch.nn as nn
import re


class GeometricVerifier(nn.Module):
    """
    Verifies VLM outputs against geometric constraints
    
    Key innovations:
    1. Extracts quantitative claims from text using regex
    2. Checks against metric geometric ground truth (from Depth Pro)
    3. Computes physical plausibility scores
    4. Supports iterative refinement (future work)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # ═══════════════════════════════════════════════════
        # Claim extraction network (optional learned component)
        # ═══════════════════════════════════════════════════
        self.use_learned_extraction = getattr(config, 'use_learned_extraction', False)
        
        if self.use_learned_extraction:
            self.claim_extractor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    batch_first=True,
                ),
                num_layers=2,
            )
        
        # ═══════════════════════════════════════════════════
        # Plausibility scorer
        # ═══════════════════════════════════════════════════
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, self.hidden_dim),  # +64 for constraint features
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        # ═══════════════════════════════════════════════════
        # Domain-specific change rate knowledge
        # ═══════════════════════════════════════════════════
        self.change_rates = {
            'construction': {'min': 0.1, 'max': 5.0},  # meters/month
            'vegetation': {'min': 0.01, 'max': 0.5},
            'erosion': {'min': 0.001, 'max': 1.0},
            'demolition': {'min': 1.0, 'max': 10.0},
        }
        
        param_count = self.count_parameters()
        print(f"GeometricVerifier initialized: {param_count / 1e6:.2f}M params")
    
    def forward(self, generated_text, geometry, geometry_encoder=None, max_iterations=3):
        """
        Verify and potentially refine generated text
        
        Args:
            generated_text: List[str] - Generated descriptions
            geometry: dict from Depth Pro encoder
            geometry_encoder: Optional DepthProEncoder instance (for computing metrics)
            max_iterations: int - Max refinement iterations (not yet implemented)
        
        Returns:
            verified_text: List[str] - Verified descriptions
            confidence_scores: Tensor [B] - Confidence scores per sample
        """
        B = len(generated_text)
        device = geometry['pointmap_t1'].device
        
        verified_texts = []
        confidence_scores = []
        
        for b in range(B):
            text = generated_text[b]
            
            # Get geometry for this sample
            geo_b = {k: v[b:b+1] if isinstance(v, torch.Tensor) else v 
                     for k, v in geometry.items()}
            
            # Compute geometric constraints
            if geometry_encoder is not None:
                constraints = geometry_encoder.compute_metrics(geo_b)
            else:
                constraints = self._compute_constraints_from_geometry(geo_b)
            
            # Verify this text
            verified, confidence = self._verify_single(
                text, constraints, max_iterations
            )
            
            verified_texts.append(verified)
            confidence_scores.append(confidence)
        
        confidence_tensor = torch.tensor(confidence_scores, device=device)
        
        return verified_texts, confidence_tensor
    
    def _verify_single(self, text, constraints, max_iterations):
        """
        Verify a single text output against constraints
        
        Args:
            text: str - Generated text
            constraints: dict - Geometric measurements (height_change, etc.)
            max_iterations: int - Max refinement iterations
        
        Returns:
            verified_text: str
            confidence: float [0, 1]
        """
        current_text = text
        
        for iteration in range(max_iterations):
            # Extract quantitative claims from text
            claims = self.extract_claims(current_text)
            
            # Check claims against geometric constraints
            verification_results = self.check_claims(claims, constraints)
            
            # Compute physical plausibility
            plausibility = self.compute_plausibility(claims, constraints)
            
            # If verification passes and plausibility is high, we're done
            if verification_results['all_pass'] and plausibility > 0.85:
                return current_text, plausibility
            
            # TODO: Implement iterative refinement
            # Would need to:
            # 1. Generate feedback from violations
            # 2. Call VLM to regenerate with constraints
            # 3. Re-verify
            # For now, just return with confidence score
            break
        
        # Compute final confidence as minimum of verification score and plausibility
        final_confidence = min(verification_results['score'], plausibility)
        
        return current_text, final_confidence
    
    def extract_claims(self, text):
        """
        Extract quantitative claims from text using regex
        
        Looks for:
        - Height/depth mentions: "5 meters tall", "grew 3m", etc.
        - Volume mentions: "100 cubic meters", "100 m³"
        - Area mentions: "50 square meters", "50 m²"
        
        Args:
            text: str
        
        Returns:
            claims: dict with extracted values
                e.g., {'height': 5.0, 'volume': 100.0}
        """
        claims = {}
        text_lower = text.lower()
        
        # ═══════════════════════════════════════════════════
        # Height patterns
        # ═══════════════════════════════════════════════════
        height_patterns = [
            r'(\d+\.?\d*)\s*(?:meters?|m)\s*(?:tall|high|height)',
            r'(?:grew|increased|rose|higher)\s*(?:by\s*)?(\d+\.?\d*)\s*(?:meters?|m)',
            r'(?:depth|height).*?(\d+\.?\d*)\s*(?:meters?|m)',
            r'(\d+\.?\d*)\s*(?:meters?|m)\s*(?:taller|higher)',
        ]
        
        for pattern in height_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['height'] = float(matches[0])
                    break
                except (ValueError, IndexError):
                    pass
        
        # ═══════════════════════════════════════════════════
        # Volume patterns
        # ═══════════════════════════════════════════════════
        volume_patterns = [
            r'(\d+\.?\d*)\s*(?:cubic\s*)?(?:meters?|m³|m3)',
            r'volume.*?(\d+\.?\d*)',
        ]
        
        for pattern in volume_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['volume'] = float(matches[0])
                    break
                except (ValueError, IndexError):
                    pass
        
        # ═══════════════════════════════════════════════════
        # Area patterns
        # ═══════════════════════════════════════════════════
        area_patterns = [
            r'(\d+\.?\d*)\s*(?:square\s*)?(?:meters?|m²|m2)',
            r'area.*?(\d+\.?\d*)',
        ]
        
        for pattern in area_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['area'] = float(matches[0])
                    break
                except (ValueError, IndexError):
                    pass
        
        return claims
    
    def check_claims(self, claims, constraints):
        """
        Check extracted claims against geometric constraints
        
        Args:
            claims: dict - Extracted claims {'height': 5.0, ...}
            constraints: dict - Ground truth {'height_change': 5.23, ...}
        
        Returns:
            dict with:
                - all_pass: bool - Whether all claims pass
                - violations: list - List of violations
                - score: float [0, 1] - Overall verification score
        """
        results = {
            'all_pass': True,
            'violations': [],
            'score': 1.0
        }
        
        if not claims:
            # No quantitative claims to verify
            return results
        
        errors = []
        
        for claim_type, claim_value in claims.items():
            # Map claim type to constraint key
            constraint_key = f'{claim_type}_change'
            
            if constraint_key in constraints:
                gt_value = abs(constraints[constraint_key])
                
                # Compute relative error
                if gt_value > 0:
                    error = abs(claim_value - gt_value) / gt_value
                else:
                    error = 0.0 if claim_value == 0 else 1.0
                
                errors.append(error)
                
                # Check against threshold (30% error tolerance)
                if error > 0.3:
                    results['all_pass'] = False
                    results['violations'].append({
                        'type': claim_type,
                        'claimed': claim_value,
                        'actual': gt_value,
                        'error': error,
                        'error_percentage': error * 100
                    })
        
        # Compute overall verification score
        if errors:
            results['score'] = max(0.0, 1.0 - sum(errors) / len(errors))
        
        return results
    
    def compute_plausibility(self, claims, constraints, domain='construction'):
        """
        Compute physical plausibility score based on domain knowledge
        
        Args:
            claims: dict - Extracted claims
            constraints: dict - Geometric constraints
            domain: str - Domain type (construction, vegetation, etc.)
        
        Returns:
            plausibility: float [0, 1]
        """
        scores = []
        
        # ═══════════════════════════════════════════════════
        # Check change rate plausibility
        # ═══════════════════════════════════════════════════
        if 'height' in claims:
            # Assume typical time period (could extract from metadata)
            time_delta = 6.0  # months
            change_rate = claims['height'] / time_delta
            
            rate_bounds = self.change_rates.get(domain, {'min': 0.01, 'max': 10.0})
            
            if rate_bounds['min'] <= change_rate <= rate_bounds['max']:
                scores.append(1.0)
            else:
                # Penalize based on how far out of bounds
                if change_rate < rate_bounds['min']:
                    # Too slow
                    scores.append(0.5)
                else:
                    # Too fast
                    over_factor = change_rate / rate_bounds['max']
                    scores.append(max(0.0, 1.0 - 0.5 * over_factor))
        
        # ═══════════════════════════════════════════════════
        # Check volume/area consistency
        # ═══════════════════════════════════════════════════
        if 'volume' in claims and 'area' in claims:
            # Rough check: volume ≈ area × typical height
            expected_height = claims['volume'] / (claims['area'] + 1e-6)
            
            # Reasonable height range for buildings
            if 1.0 <= expected_height <= 50.0:
                scores.append(1.0)
            else:
                scores.append(0.3)
        
        # Return average plausibility score
        return sum(scores) / len(scores) if scores else 0.5
    
    def _compute_constraints_from_geometry(self, geometry):
        """
        Compute geometric constraints directly from geometry dict
        
        This is used when geometry_encoder.compute_metrics() is not available
        
        Args:
            geometry: dict from Depth Pro
        
        Returns:
            dict with height_change, volume_change, area_change
        """
        pointmap_t1 = geometry['pointmap_t1']  # [1, H, W, 3]
        pointmap_t2 = geometry['pointmap_t2']
        change_mask = geometry['change_mask']  # [1, H, W]
        
        # Get changed regions
        mask = change_mask[0] > 0.5  # [H, W]
        
        if mask.sum() == 0:
            return {
                'height_change': 0.0,
                'volume_change': 0.0,
                'area_change': 0.0
            }
        
        changed_t1 = pointmap_t1[0][mask]  # [N, 3]
        changed_t2 = pointmap_t2[0][mask]
        
        # Height change (max depth difference)
        height_change = (changed_t2[:, 2].max() - changed_t1[:, 2].max()).item()
        
        # Approximate volume
        volume_change = geometry['change_magnitude'][0][mask].sum().item() * 0.01
        
        # Area of change
        area_change = mask.sum().item() * 0.01
        
        return {
            'height_change': float(height_change),
            'volume_change': float(volume_change),
            'area_change': float(area_change)
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)