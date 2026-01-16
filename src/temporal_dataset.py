"""
Dataset for temporal image pairs with change annotations
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms


class TemporalDataset(Dataset):
    """
    Dataset for temporal change detection
    
    Expected structure:
    data_root/
        ├── train/
        │   ├── scene_001/
        │   │   ├── t1.jpg
        │   │   ├── t2.jpg
        │   │   └── annotation.json
        │   ├── scene_002/
        │   ...
    
    annotation.json format:
    {
        "description": "A building was constructed...",
        "process_stages": ["site_clearing", "foundation", ...],
        "constraints": {
            "height_change": 12.5,
            "volume_change": 1500,
            "area_change": 300
        }
    }
    """
    def __init__(self, data_root, config):
        super().__init__()
        self.data_root = data_root
        self.config = config
        
        # Load scenes
        self.scenes = self._load_scenes()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Loaded {len(self.scenes)} scenes from {data_root}")
    
    def _load_scenes(self):
        """Load all scene directories"""
        scenes = []
        
        for scene_name in os.listdir(self.data_root):
            scene_path = os.path.join(self.data_root, scene_name)
            
            if not os.path.isdir(scene_path):
                continue
            
            # Check required files exist
            t1_path = os.path.join(scene_path, 't1.jpg')
            t2_path = os.path.join(scene_path, 't2.jpg')
            anno_path = os.path.join(scene_path, 'annotation.json')
            
            if all(os.path.exists(p) for p in [t1_path, t2_path, anno_path]):
                scenes.append({
                    'name': scene_name,
                    'path': scene_path,
                    't1': t1_path,
                    't2': t2_path,
                    'annotation': anno_path
                })
        
        return scenes
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        
        # Load images
        img_t1 = Image.open(scene['t1']).convert('RGB')
        img_t2 = Image.open(scene['t2']).convert('RGB')
        
        # Apply transforms
        img_t1 = self.transform(img_t1)
        img_t2 = self.transform(img_t2)
        
        # Load annotation
        with open(scene['annotation'], 'r') as f:
            annotation = json.load(f)
        
        return {
            'scene_name': scene['name'],
            'img_t1': img_t1,
            'img_t2': img_t2,
            'question': "What changed between these two images?",
            'description': annotation['description'],
            'process_stages': annotation.get('process_stages', []),
            'constraints': annotation.get('constraints', {})
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function
        """
        return {
            'scene_names': [item['scene_name'] for item in batch],
            'img_t1': torch.stack([item['img_t1'] for item in batch]),
            'img_t2': torch.stack([item['img_t2'] for item in batch]),
            'questions': [item['question'] for item in batch],
            'descriptions': [item['description'] for item in batch],
            'process_stages': [item['process_stages'] for item in batch],
            'constraints': [item['constraints'] for item in batch]
        }