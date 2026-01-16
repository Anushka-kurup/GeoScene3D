"""
Geometric verifier module
YOUR NOVEL CONTRIBUTION #2
"""

import torch
import torch.nn as nn
import re


class GeometricVerifier(nn.Module):
    """
    Verifies VLM outputs against geometric constraints
    
    Key innovations:
    1. Extracts quantitative claims from text
    2. Checks against geometric ground truth
    3. Computes physical plausibility scores
    4. Iteratively refines outputs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # ═══════════════════════════════════════════════════
        # Claim extraction network
        # ═══════════════════════════════════════════════════
        self.claim_extractor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
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
            nn.Sigmoid()
        )
        
        # ═══════════════════════════════════════════════════
        # Domain-specific change rate knowledge
        # ═══════════════════════════════════════════════════
        self.change_rates = {
            'construction': {'min': 0.1, 'max': 5.0},  # meters/month
            'vegetation': {'min': 0.01, 'max': 0.5},
            'erosion': {'min': 0.001, 'max': 1.0},
            'demolition': {'min': 1.0, 'max': 10.0}
        }
        
        print(f"GeometricVerifier initialized: {self.count_parameters()/1e6:.2f}M params")
    
    def forward(self, generated_text, geometry, geometry_encoder=None, max_iterations=3):
        """
        Verify and potentially refine generated text
        
        Args:
            generated_text: List[str] - generated descriptions
            geometry: dict from geometry encoder
            geometry_encoder: GeometryEncoder instance (for computing metrics)
            max_iterations: int
        
        Returns:
            verified_text: List[str]
            confidence_scores: [B]
        """
        B = len(generated_text)
        device = geometry['pointmap_t1'].device
        
        verified_texts = []
        confidence_scores = []
        
        for b in range(B):
            text = generated_text[b]
            geo_b = {k: v[b:b+1] for k, v in geometry.items()}
            
            # Compute geometric constraints
            if geometry_encoder is not None:
                constraints = geometry_encoder.compute_metrics(geo_b)
            else:
                constraints = self._compute_simple_constraints(geo_b)
            
            # Verify
            verified, confidence = self._verify_single(
                text, constraints, max_iterations
            )
            
            verified_texts.append(verified)
            confidence_scores.append(confidence)
        
        confidence_tensor = torch.tensor(confidence_scores, device=device)
        
        return verified_texts, confidence_tensor
    
    def _verify_single(self, text, constraints, max_iterations):
        """
        Verify a single text output
        
        Args:
            text: str
            constraints: dict with geometric measurements
            max_iterations: int
        
        Returns:
            verified_text: str
            confidence: float
        """
        current_text = text
        
        for iteration in range(max_iterations):
            # Extract claims
            claims = self.extract_claims(current_text)
            
            # Check against constraints
            verification_results = self.check_claims(claims, constraints)
            
            # Compute plausibility
            plausibility = self.compute_plausibility(claims, constraints)
            
            # If all good, return
            if verification_results['all_pass'] and plausibility > 0.85:
                return current_text, plausibility
            
            # Otherwise, would need to call VLM again to refine
            # For now, just return with confidence score
            # In full implementation, you'd generate feedback and re-generate
            break
        
        # Return with computed confidence
        final_confidence = min(
            verification_results['score'], 
            plausibility
        )
        
        return current_text, final_confidence
    
    def extract_claims(self, text):
        """
        Extract quantitative claims from text using regex
        
        Args:
            text: str
        
        Returns:
            claims: dict with extracted values
        """
        claims = {}
        
        text_lower = text.lower()
        
        # Height mentions
        height_patterns = [
            r'(\d+\.?\d*)\s*(?:meters?|m)\s*(?:tall|high|height)',
            r'(?:grew|increased|rose)\s*(?:by\s*)?(\d+\.?\d*)\s*(?:meters?|m)',
            r'height.*?(\d+\.?\d*)\s*(?:meters?|m)'
        ]
        
        for pattern in height_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['height'] = float(matches[0])
                    break
                except:
                    pass
        
        # Volume mentions
        volume_patterns = [
            r'(\d+\.?\d*)\s*(?:cubic\s*)?(?:meters?|m³|m3)',
            r'volume.*?(\d+\.?\d*)'
        ]
        
        for pattern in volume_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['volume'] = float(matches[0])
                    break
                except:
                    pass
        
        # Area mentions
        area_patterns = [
            r'(\d+\.?\d*)\s*(?:square\s*)?(?:meters?|m²|m2)',
            r'area.*?(\d+\.?\d*)'
        ]
        
        for pattern in area_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    claims['area'] = float(matches[0])
                    break
                except:
                    pass
        
        return claims
    
    def check_claims(self, claims, constraints):
        """
        Check claims against geometric constraints
        
        Args:
            claims: dict from extract_claims
            constraints: dict from geometry
        
        Returns:
            dict with verification results
        """
        results = {
            'all_pass': True,
            'violations': [],
            'score': 1.0
        }
        
        if not claims:
            return results
        
        errors = []
        
        for claim_type, claim_value in claims.items():
            # Map claim types
            constraint_key = f"{claim_type}_change"
            
            if constraint_key in constraints:
                gt_value = abs(constraints[constraint_key])
                
                # Compute relative error
                if gt_value > 0:
                    error = abs(claim_value - gt_value) / gt_value
                else:
                    error = 0.0 if claim_value == 0 else 1.0
                
                errors.append(error)
                
                # Check threshold
                if error > 0.3:  # 30% error threshold
                    results['all_pass'] = False
                    results['violations'].append({
                        'type': claim_type,
                        'claimed': claim_value,
                        'actual': gt_value,
                        'error': error
                    })
        
        # Compute overall score
        if errors:
            results['score'] = max(0.0, 1.0 - sum(errors) / len(errors))
        
        return results
    
    def compute_plausibility(self, claims, constraints, domain='construction'):
        """
        Compute physical plausibility score
        
        Args:
            claims: dict
            constraints: dict
            domain: str
        
        Returns:
            plausibility: float [0, 1]
        """
        scores = []
        
        # Check if change rate is plausible
        if 'height' in claims:
            # Assume 6 months (could extract from text)
            time_delta = 6.0  # months
            rate = claims['height'] / time_delta
            
            rate_bounds = self.change_rates.get(domain, {'min': 0.01, 'max': 10.0})
            
            if rate_bounds['min'] <= rate <= rate_bounds['max']:
                scores.append(1.0)
            else:
                # Penalize based on how far out of bounds
                if rate < rate_bounds['min']:
                    scores.append(0.5)
                else:
                    over_factor = rate / rate_bounds['max']
                    scores.append(max(0.0, 1.0 - 0.5 * over_factor))
        
        # Check volume/area consistency
        if 'volume' in claims and 'area' in claims:
            # Rough check: volume should be area × typical height
            expected_height = claims['volume'] / (claims['area'] + 1e-6)
            
            if 1.0 <= expected_height <= 50.0:  # Reasonable building height
                scores.append(1.0)
            else:
                scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _compute_simple_constraints(self, geometry):
        """
        Simple constraint computation if geometry_encoder not available
        """
        pointmap_t1 = geometry['pointmap_t1']
        pointmap_t2 = geometry['pointmap_t2']
        change_mask = geometry['change_mask']
        
        changed_t1 = pointmap_t1[change_mask > 0.5]
        changed_t2 = pointmap_t2[change_mask > 0.5]
        
        if len(changed_t1) == 0:
            return {
                'height_change': 0.0,
                'volume_change': 0.0,
                'area_change': 0.0
            }
        
        height_change = (changed_t2[:, 2].max() - changed_t1[:, 2].max()).item()
        volume = geometry['change_magnitude'][change_mask > 0.5].sum().item() * 0.01
        area = (change_mask > 0.5).sum().item() * 0.01
        
        return {
            'height_change': float(height_change),
            'volume_change': float(volume),
            'area_change': float(area)
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)