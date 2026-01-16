"""
Dataset loaders for temporal RGB-D sequences

Supports:
- ScanNet (sequences of RGB-D frames with annotations)
- Matterport3D (building-scale sequences)
- Replica (synthetic sequences)
- 7-Scenes (tracking sequences)
- Custom temporal data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from dataclasses import dataclass


@dataclass
class SequenceInfo:
    """Metadata for a sequence"""
    sequence_id: str
    num_frames: int
    fps: float
    scene_type: str
    annotations: Optional[Dict] = None


class ScanNetSequenceDataset(Dataset):
    """
    ScanNet dataset with temporal sequences
    
    ScanNet provides RGB-D video sequences of indoor scenes with:
    - RGB frames (1920x1080 or 640x480)
    - Depth frames (aligned)
    - Camera poses
    - 3D semantic annotations
    - Instance segmentations
    
    Perfect for your temporal reasoning task!
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 8,
        frame_skip: int = 10,
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            data_root: Path to ScanNet data (e.g., '/data/scannet/scans')
            sequence_length: Number of frames per sequence
            frame_skip: Skip frames for temporal diversity (e.g., 10 = ~0.5s at 20fps)
            split: 'train', 'val', or 'test'
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.transform = transform
        
        # Load scene list
        split_file = self.data_root.parent / f'scannetv2_{split}.txt'
        with open(split_file, 'r') as f:
            self.scenes = [line.strip() for line in f.readlines()]
        
        # Build sequence list
        self.sequences = self._build_sequences()
    
    def _build_sequences(self) -> List[Dict]:
        """Build list of valid sequences"""
        sequences = []
        
        for scene_id in self.scenes:
            scene_path = self.data_root / scene_id
            
            # Check if scene has necessary data
            color_path = scene_path / 'color'
            depth_path = scene_path / 'depth'
            
            if not color_path.exists() or not depth_path.exists():
                continue
            
            # Get frame list
            frames = sorted([f.stem for f in color_path.glob('*.jpg')])
            
            # Create sequences with temporal spacing
            for start_idx in range(0, len(frames) - self.sequence_length * self.frame_skip, 
                                  self.frame_skip * 5):  # Stride between sequences
                frame_ids = []
                for i in range(self.sequence_length):
                    frame_idx = start_idx + i * self.frame_skip
                    if frame_idx < len(frames):
                        frame_ids.append(frames[frame_idx])
                
                if len(frame_ids) == self.sequence_length:
                    sequences.append({
                        'scene_id': scene_id,
                        'frame_ids': frame_ids,
                        'scene_path': scene_path
                    })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def load_intrinsics(self, scene_path: Path) -> np.ndarray:
        """Load camera intrinsics"""
        intrinsic_file = scene_path / 'intrinsic' / 'intrinsic_depth.txt'
        return np.loadtxt(intrinsic_file)
    
    def depth_to_pointcloud(
        self, 
        depth: np.ndarray, 
        intrinsics: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert depth map to point cloud"""
        h, w = depth.shape
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Compute 3D coordinates
        z = depth / 1000.0  # Convert mm to meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack to point cloud
        points = np.stack([x, y, z], axis=-1)
        
        # Filter invalid points
        valid = (z > 0) & (z < 10)  # Keep points within 10m
        points = points[valid]
        
        if rgb is not None:
            colors = rgb[valid]
            return points, colors
        
        return points
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a sequence"""
        seq_info = self.sequences[idx]
        scene_path = seq_info['scene_path']
        frame_ids = seq_info['frame_ids']
        
        # Load camera intrinsics
        intrinsics = self.load_intrinsics(scene_path)
        
        # Load frames
        frames = []
        for frame_id in frame_ids:
            # Load RGB
            rgb_path = scene_path / 'color' / f'{frame_id}.jpg'
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load depth
            depth_path = scene_path / 'depth' / f'{frame_id}.png'
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            
            # Convert to point cloud
            points, colors = self.depth_to_pointcloud(depth, intrinsics, rgb)
            
            frames.append({
                'rgb': torch.from_numpy(rgb).float() / 255.0,
                'depth': torch.from_numpy(depth).float(),
                'point_cloud': torch.from_numpy(points).float(),
                'point_colors': torch.from_numpy(colors).float() / 255.0,
                'frame_id': frame_id,
                'timestamp': int(frame_id) / 20.0  # Assume 20 FPS
            })
        
        return {
            'sequence_id': seq_info['scene_id'],
            'frames': frames,
            'intrinsics': torch.from_numpy(intrinsics).float()
        }


class Matterport3DSequenceDataset(Dataset):
    """
    Matterport3D with temporal sequences
    
    Large-scale building interiors with:
    - High-quality RGB-D
    - Multiple viewpoints per room
    - Semantic annotations
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 8,
        split: str = 'train'
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        
        # Load building list
        split_file = self.data_root / f'{split}_scans.txt'
        with open(split_file, 'r') as f:
            self.buildings = [line.strip() for line in f.readlines()]
        
        self.sequences = self._build_sequences()
    
    def _build_sequences(self) -> List[Dict]:
        """Build temporal sequences from panorama sequences"""
        sequences = []
        
        for building_id in self.buildings:
            building_path = self.data_root / building_id / 'matterport_color_images'
            
            if not building_path.exists():
                continue
            
            # Get all panoramas
            panos = sorted([d.name for d in building_path.iterdir() if d.is_dir()])
            
            # Create sequences
            for i in range(len(panos) - self.sequence_length + 1):
                sequences.append({
                    'building_id': building_id,
                    'pano_ids': panos[i:i + self.sequence_length]
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        # Implementation similar to ScanNet
        # Load RGB-D frames from each panorama
        pass


class ReplicaDataset(Dataset):
    """
    Replica: High-quality synthetic indoor scenes
    
    Features:
    - Photorealistic RGB
    - Perfect depth
    - Dense semantic labels
    - Controlled trajectories
    
    Great for testing!
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 8,
        trajectory: str = 'trajectory_0'
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.trajectory = trajectory
    
    def __len__(self) -> int:
        # Count frames in trajectory
        pass
    
    def __getitem__(self, idx: int) -> Dict:
        # Load frames from trajectory
        pass


class CustomSequenceDataset(Dataset):
    """
    Custom dataset loader for your own RGB-D sequences
    
    Expected structure:
    data_root/
        sequence_001/
            rgb/
                frame_000000.jpg
                frame_000001.jpg
                ...
            depth/
                frame_000000.png
                frame_000001.png
                ...
            metadata.json  # Optional: intrinsics, timestamps
        sequence_002/
            ...
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 8,
        frame_skip: int = 1
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        
        self.sequences = self._discover_sequences()
    
    def _discover_sequences(self) -> List[Dict]:
        """Discover all valid sequences"""
        sequences = []
        
        for seq_dir in self.data_root.iterdir():
            if not seq_dir.is_dir():
                continue
            
            rgb_dir = seq_dir / 'rgb'
            depth_dir = seq_dir / 'depth'
            
            if not rgb_dir.exists() or not depth_dir.exists():
                continue
            
            # Get frame list
            rgb_frames = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
            
            if len(rgb_frames) < self.sequence_length * self.frame_skip:
                continue
            
            # Create subsequences
            for start_idx in range(0, len(rgb_frames) - self.sequence_length * self.frame_skip,
                                  self.sequence_length):
                frame_indices = list(range(start_idx, 
                                          start_idx + self.sequence_length * self.frame_skip,
                                          self.frame_skip))
                
                sequences.append({
                    'sequence_dir': seq_dir,
                    'frame_indices': frame_indices,
                    'rgb_frames': [rgb_frames[i] for i in frame_indices]
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def load_metadata(self, seq_dir: Path) -> Dict:
        """Load metadata if available"""
        metadata_file = seq_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a sequence"""
        seq_info = self.sequences[idx]
        seq_dir = seq_info['sequence_dir']
        rgb_frames = seq_info['rgb_frames']
        
        # Load metadata
        metadata = self.load_metadata(seq_dir)
        intrinsics = metadata.get('intrinsics', None)
        
        # Default intrinsics if not provided
        if intrinsics is None:
            # Assume 640x480 with typical webcam parameters
            intrinsics = np.array([
                [525.0, 0, 319.5],
                [0, 525.0, 239.5],
                [0, 0, 1.0]
            ])
        else:
            intrinsics = np.array(intrinsics)
        
        frames = []
        depth_dir = seq_dir / 'depth'
        
        for rgb_path in rgb_frames:
            # Load RGB
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Load corresponding depth
            depth_path = depth_dir / rgb_path.name.replace('.jpg', '.png')
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            
            # Convert to point cloud
            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            
            z = depth / 1000.0
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            points = np.stack([x, y, z], axis=-1)
            valid = (z > 0) & (z < 10)
            points = points[valid]
            colors = rgb[valid]
            
            frames.append({
                'rgb': torch.from_numpy(rgb).float() / 255.0,
                'depth': torch.from_numpy(depth).float(),
                'point_cloud': torch.from_numpy(points).float(),
                'point_colors': torch.from_numpy(colors).float() / 255.0,
                'frame_id': rgb_path.stem,
                'timestamp': float(rgb_path.stem.split('_')[-1]) if '_' in rgb_path.stem else 0.0
            })
        
        return {
            'sequence_id': seq_dir.name,
            'frames': frames,
            'intrinsics': torch.from_numpy(intrinsics).float()
        }


def get_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create dataloader
    
    Args:
        dataset_name: 'scannet', 'matterport3d', 'replica', or 'custom'
        data_root: Path to dataset
        batch_size: Batch size (typically 1 for sequences)
        num_workers: Number of worker processes
    """
    
    if dataset_name == 'scannet':
        dataset = ScanNetSequenceDataset(data_root, **kwargs)
    elif dataset_name == 'matterport3d':
        dataset = Matterport3DSequenceDataset(data_root, **kwargs)
    elif dataset_name == 'replica':
        dataset = ReplicaDataset(data_root, **kwargs)
    elif dataset_name == 'custom':
        dataset = CustomSequenceDataset(data_root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )


# Dataset recommendations
DATASET_INFO = """
RECOMMENDED DATASETS FOR TEMPORAL RGB-D:

1. **ScanNet** (BEST CHOICE) ✅
   - 1513 RGB-D sequences of indoor scenes
   - 20 FPS video with depth
   - Semantic and instance annotations
   - Camera poses for accurate tracking
   - Download: http://www.scan-net.org/
   
2. **Matterport3D**
   - Building-scale sequences
   - Multiple viewpoints per location
   - High-quality textures
   - Download: https://niessner.github.io/Matterport/
   
3. **Replica** (Synthetic)
   - Perfect depth
   - Controlled trajectories
   - Good for testing
   - Download: https://github.com/facebookresearch/Replica-Dataset
   
4. **7-Scenes** (Smaller scale)
   - Indoor tracking sequences
   - Ground truth poses
   - Download: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

5. **SceneNN**
   - 100+ RGB-D sequences
   - Per-frame semantic labels
   - Download: http://www.scenenn.net/

WHY NOT SUN RGB-D?
- Only single snapshots (no temporal data)
- No frame sequences or video
- Cannot track objects over time
"""

if __name__ == '__main__':
    print(DATASET_INFO)
