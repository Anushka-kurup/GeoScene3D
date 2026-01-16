"""
Temporal G2VLM: Geometry-Enhanced Vision-Language Model with Temporal Reasoning

Extends G2VLM to handle multiple frames over time, enabling:
- Object movement detection
- Layout change tracking
- Temporal spatial reasoning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import open3d as o3d


@dataclass
class FrameData:
    """Data structure for a single frame"""
    rgb: torch.Tensor  # (H, W, 3)
    depth: torch.Tensor  # (H, W)
    point_cloud: torch.Tensor  # (N, 3)
    timestamp: float
    frame_id: int


@dataclass
class TemporalChange:
    """Detected change between frames"""
    object_id: int
    change_type: str  # 'moved', 'rotated', 'added', 'removed', 'unchanged'
    translation: Optional[np.ndarray]  # (3,) translation vector
    rotation: Optional[np.ndarray]  # (3, 3) rotation matrix
    confidence: float


class PointCloudMatcher:
    """Match point clouds across frames to detect changes"""
    
    def __init__(self, voxel_size: float = 0.05, distance_threshold: float = 0.1):
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
    
    def extract_features(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Extract FPFH features from point cloud"""
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )
        
        return pcd_down, fpfh
    
    def register_point_clouds(
        self, 
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud
    ) -> o3d.pipelines.registration.RegistrationResult:
        """Register source point cloud to target using RANSAC + ICP"""
        
        # Extract features
        source_down, source_fpfh = self.extract_features(source)
        target_down, target_fpfh = self.extract_features(target)
        
        # RANSAC registration
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        # Refine with ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, self.distance_threshold,
            ransac_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return icp_result
    
    def detect_changes(
        self,
        pcd1: o3d.geometry.PointCloud,
        pcd2: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect changes between two point clouds
        Returns: (translation, rotation_matrix, fitness_score)
        """
        result = self.register_point_clouds(pcd1, pcd2)
        
        # Extract translation and rotation
        transformation = result.transformation
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        return translation, rotation, result.fitness


class ObjectTracker:
    """Track objects across multiple frames"""
    
    def __init__(self, distance_threshold: float = 0.3):
        self.distance_threshold = distance_threshold
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_id = 0
    
    def update(self, detected_objects: List[Dict], frame_id: int) -> List[Dict]:
        """
        Update tracked objects with new detections
        Args:
            detected_objects: List of dicts with keys: 'bbox', 'center', 'label', 'point_cloud'
            frame_id: Current frame ID
        Returns:
            List of tracked objects with persistent IDs
        """
        tracked = []
        
        if not self.tracked_objects:
            # First frame - initialize all objects
            for obj in detected_objects:
                obj_data = {
                    'id': self.next_id,
                    'label': obj['label'],
                    'center': np.array(obj['center']),
                    'bbox': obj['bbox'],
                    'point_cloud': obj.get('point_cloud'),
                    'first_seen': frame_id,
                    'last_seen': frame_id,
                    'trajectory': [np.array(obj['center'])]
                }
                self.tracked_objects[self.next_id] = obj_data
                tracked.append(obj_data)
                self.next_id += 1
        else:
            # Match with existing objects
            matched_ids = set()
            
            for obj in detected_objects:
                obj_center = np.array(obj['center'])
                
                # Find closest tracked object
                min_dist = float('inf')
                best_match_id = None
                
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if obj_id in matched_ids:
                        continue
                    
                    tracked_center = np.array(tracked_obj['center'])
                    dist = np.linalg.norm(obj_center - tracked_center)
                    
                    # Also check if labels match
                    if dist < min_dist and dist < self.distance_threshold:
                        if obj['label'] == tracked_obj['label']:
                            min_dist = dist
                            best_match_id = obj_id
                
                if best_match_id is not None:
                    # Update existing object
                    tracked_obj = self.tracked_objects[best_match_id]
                    tracked_obj['center'] = np.array(obj['center'])
                    tracked_obj['bbox'] = obj['bbox']
                    tracked_obj['point_cloud'] = obj.get('point_cloud')
                    tracked_obj['last_seen'] = frame_id
                    tracked_obj['trajectory'].append(np.array(obj['center']))
                    
                    tracked.append(tracked_obj)
                    matched_ids.add(best_match_id)
                else:
                    # New object
                    obj_data = {
                        'id': self.next_id,
                        'label': obj['label'],
                        'center': np.array(obj['center']),
                        'bbox': obj['bbox'],
                        'point_cloud': obj.get('point_cloud'),
                        'first_seen': frame_id,
                        'last_seen': frame_id,
                        'trajectory': [np.array(obj['center'])]
                    }
                    self.tracked_objects[self.next_id] = obj_data
                    tracked.append(obj_data)
                    self.next_id += 1
        
        return tracked
    
    def get_changes(self, frame_id: int, lookback: int = 1) -> List[TemporalChange]:
        """Get changes detected between current frame and lookback frames ago"""
        changes = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            if len(obj_data['trajectory']) < 2:
                continue
            
            # Compare current position with lookback position
            if len(obj_data['trajectory']) > lookback:
                current_pos = obj_data['trajectory'][-1]
                past_pos = obj_data['trajectory'][-lookback-1]
                
                translation = current_pos - past_pos
                distance_moved = np.linalg.norm(translation)
                
                if distance_moved > 0.05:  # 5cm threshold
                    change = TemporalChange(
                        object_id=obj_id,
                        change_type='moved',
                        translation=translation,
                        rotation=None,
                        confidence=min(1.0, distance_moved / 1.0)  # Normalize by 1m
                    )
                    changes.append(change)
        
        return changes


class TemporalGeometryEncoder(nn.Module):
    """Encode geometry information across multiple frames"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        # Spatial encoder (per-frame)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 128),  # xyz coordinates
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Temporal encoder (across frames)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Change encoder (for detected changes)
        self.change_encoder = nn.Sequential(
            nn.Linear(6, 128),  # translation (3) + rotation (3)
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self, 
        point_clouds: List[torch.Tensor],
        changes: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            point_clouds: List of (N, 3) point clouds for each frame
            changes: Optional list of (M, 6) change vectors
        Returns:
            (T, hidden_dim) temporal geometry features
        """
        # Encode each frame's geometry
        spatial_features = []
        for pc in point_clouds:
            # Mean pooling over points
            feat = self.spatial_encoder(pc).mean(dim=0)  # (hidden_dim,)
            spatial_features.append(feat)
        
        spatial_features = torch.stack(spatial_features)  # (T, hidden_dim)
        
        # Temporal encoding
        temporal_features, _ = self.temporal_encoder(spatial_features.unsqueeze(0))  # (1, T, 2*hidden_dim)
        temporal_features = temporal_features.squeeze(0)  # (T, 2*hidden_dim)
        
        # Fuse bidirectional features
        output = self.fusion(temporal_features)  # (T, hidden_dim)
        
        return output


class TemporalG2VLM(nn.Module):
    """
    Temporal Geometry-Enhanced Vision-Language Model
    
    Processes sequences of RGB-D frames to understand spatial changes over time
    """
    
    def __init__(
        self,
        vlm_model: nn.Module,  # Base VLM (e.g., LLaVA, BLIP-2)
        hidden_dim: int = 768,
        num_frames: int = 8
    ):
        super().__init__()
        
        self.vlm = vlm_model
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        # Temporal geometry encoder
        self.geometry_encoder = TemporalGeometryEncoder(hidden_dim)
        
        # Frame-level visual encoder (shared across frames)
        self.frame_encoder = nn.Identity()  # Use VLM's vision encoder
        
        # Temporal visual encoder
        self.temporal_visual = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # visual + geometry + changes
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Object tracker
        self.tracker = ObjectTracker()
        self.matcher = PointCloudMatcher()
    
    def encode_frame_sequence(
        self,
        frames: List[FrameData],
        detected_objects: List[List[Dict]]
    ) -> Tuple[torch.Tensor, List[TemporalChange]]:
        """
        Encode a sequence of frames with temporal reasoning
        
        Args:
            frames: List of FrameData objects
            detected_objects: List of detected objects per frame
        
        Returns:
            encoded_features: (T, hidden_dim) temporal features
            changes: List of detected changes
        """
        # Extract point clouds
        point_clouds = [f.point_cloud for f in frames]
        
        # Encode geometry across time
        geometry_features = self.geometry_encoder(point_clouds)
        
        # Track objects across frames
        all_changes = []
        for frame_id, (frame, objects) in enumerate(zip(frames, detected_objects)):
            tracked = self.tracker.update(objects, frame_id)
            
            if frame_id > 0:
                changes = self.tracker.get_changes(frame_id)
                all_changes.extend(changes)
        
        return geometry_features, all_changes
    
    def forward(
        self,
        frames: List[FrameData],
        detected_objects: List[List[Dict]],
        questions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal reasoning
        
        Args:
            frames: List of FrameData objects
            detected_objects: List of detected objects per frame
            questions: Questions to answer about the scene
        
        Returns:
            Dictionary with answers and detected changes
        """
        # Encode temporal geometry
        geometry_features, changes = self.encode_frame_sequence(frames, detected_objects)
        
        # Encode visual features per frame (use VLM's vision encoder)
        visual_features = []
        for frame in frames:
            # This would use the VLM's built-in vision encoder
            feat = self.vlm.encode_image(frame.rgb)
            visual_features.append(feat)
        
        visual_features = torch.stack(visual_features)  # (T, hidden_dim)
        
        # Temporal visual encoding
        temporal_visual, _ = self.temporal_visual(visual_features.unsqueeze(0))
        temporal_visual = temporal_visual.squeeze(0)  # (T, 2*hidden_dim)
        
        # Fuse everything
        combined = torch.cat([
            visual_features,
            geometry_features,
            temporal_visual
        ], dim=-1)  # (T, 3*hidden_dim)
        
        fused_features = self.fusion(combined)  # (T, hidden_dim)
        
        # Generate responses using VLM
        outputs = self.vlm.generate(
            visual_features=fused_features,
            questions=questions
        )
        
        return {
            'answers': outputs,
            'changes': changes,
            'tracked_objects': self.tracker.tracked_objects
        }


def create_temporal_g2vlm(
    base_vlm: nn.Module,
    num_frames: int = 8,
    hidden_dim: int = 768
) -> TemporalG2VLM:
    """Factory function to create Temporal G2VLM model"""
    return TemporalG2VLM(
        vlm_model=base_vlm,
        hidden_dim=hidden_dim,
        num_frames=num_frames
    )