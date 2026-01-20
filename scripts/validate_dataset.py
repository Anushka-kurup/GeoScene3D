"""
Validate Replica dataset setup
"""
import sys
sys.path.append('src')

from data.replica_dataset import ReplicaChangeDataset
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('configs/train_config.yaml')

print("="*80)
print("Validating Replica Dataset")
print("="*80)

# Test loading
print(f"\nDataset path: {config.data.replica_root}")

try:
    dataset = ReplicaChangeDataset(
        replica_root=config.data.replica_root,
        split='train',
        image_size=config.data.image_size,
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    
    # Load a sample
    print("\nTesting sample loading...")
    sample = dataset[0]
    
    print(f"✓ Sample loaded successfully!")
    print(f"  Image T1 shape: {sample['img_t1'].shape}")
    print(f"  Image T2 shape: {sample['img_t2'].shape}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    print(f"  Scene: {sample['scene']}")
    print(f"  Temporal gap: {sample['temporal_gap']}")
    
    print("\n" + "="*80)
    print("Dataset validation PASSED! ✓")
    print("="*80)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
