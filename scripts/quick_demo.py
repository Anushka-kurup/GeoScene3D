"""
Quick Demo: Test Temporal G2VLM Without Full Dataset

This creates synthetic test data so you can start developing right away
while Replica downloads in the background.
"""

import numpy as np
import cv2
import json
from pathlib import Path
import torch


def create_synthetic_room(size=(480, 640), num_objects=5):
    """
    Create a simple synthetic room scene with objects
    
    Returns:
        rgb: RGB image
        depth: Depth map
        objects: List of object bounding boxes and centers
    """
    # Create RGB image
    rgb = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200  # Gray background
    
    # Draw floor
    rgb[size[0]//2:, :] = [180, 150, 120]  # Brown floor
    
    # Create depth map (simple gradient)
    depth = np.linspace(1000, 3000, size[0])  # 1m to 3m
    depth = np.tile(depth[:, np.newaxis], (1, size[1]))
    depth = depth.astype(np.uint16)
    
    objects = []
    
    # Add some random objects
    for i in range(num_objects):
        # Random position
        x = np.random.randint(100, size[1] - 100)
        y = np.random.randint(100, size[0] - 100)
        w = np.random.randint(40, 80)
        h = np.random.randint(40, 80)
        
        # Random color
        color = tuple(np.random.randint(50, 255, 3).tolist())
        
        # Draw rectangle (object)
        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Border
        
        # Update depth (object is closer)
        depth[y:y+h, x:x+w] = np.random.randint(800, 1500)
        
        # Store object info
        objects.append({
            'bbox': [x, y, w, h],
            'center': np.array([x + w/2, y + h/2]),
            'label': f'object_{i}',
            'depth': float(np.mean(depth[y:y+h, x:x+w]))
        })
    
    return rgb, depth, objects


def move_object(rgb, depth, objects, obj_idx, dx, dy):
    """Move an object in the scene"""
    obj = objects[obj_idx]
    x, y, w, h = obj['bbox']
    
    # Get object region
    obj_rgb = rgb[y:y+h, x:x+w].copy()
    obj_depth = depth[y:y+h, x:x+w].copy()
    
    # Clear old position (replace with background)
    rgb[y:y+h, x:x+w] = [180, 150, 120]  # Floor color
    depth[y:y+h, x:x+w] = 2000  # Background depth
    
    # Calculate new position
    new_x = max(0, min(rgb.shape[1] - w, x + dx))
    new_y = max(0, min(rgb.shape[0] - h, y + dy))
    
    # Place at new position
    rgb[new_y:new_y+h, new_x:new_x+w] = obj_rgb
    depth[new_y:new_y+h, new_x:new_x+w] = obj_depth
    
    # Update object info
    objects[obj_idx]['bbox'] = [new_x, new_y, w, h]
    objects[obj_idx]['center'] = np.array([new_x + w/2, new_y + h/2])
    
    return rgb, depth, objects


def create_demo_sequence(output_dir='./demo_data', num_frames=8):
    """
    Create a demo temporal sequence with object movements
    
    Args:
        output_dir: Where to save the sequence
        num_frames: Number of frames per sequence
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Creating demo sequence...")
    print(f"📁 Output: {output_path}")
    
    # Create initial scene
    rgb_before, depth_before, objects_before = create_synthetic_room(num_objects=5)
    
    # Save "before" sequence
    before_dir = output_path / 'before'
    before_dir.mkdir(exist_ok=True)
    
    print("\n📸 Generating 'before' frames...")
    for i in range(num_frames):
        # Add slight camera movement for variety
        noise = np.random.randint(-5, 5, size=(2,))
        
        # Save RGB
        rgb_path = before_dir / f'rgb_{i:06d}.jpg'
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_before, cv2.COLOR_RGB2BGR))
        
        # Save depth
        depth_path = before_dir / f'depth_{i:06d}.png'
        cv2.imwrite(str(depth_path), depth_before)
        
        print(f"  ✅ Frame {i}: {rgb_path.name}")
    
    # Create "after" scene with movements
    rgb_after = rgb_before.copy()
    depth_after = depth_before.copy()
    objects_after = [obj.copy() for obj in objects_before]
    
    # Move some objects
    movements = []
    print("\n🔄 Simulating object movements...")
    
    # Move object 0 to the right
    rgb_after, depth_after, objects_after = move_object(
        rgb_after, depth_after, objects_after, 0, dx=100, dy=0
    )
    movements.append("Object 0 moved 100 pixels right")
    print(f"  📦 {movements[-1]}")
    
    # Move object 2 up
    if len(objects_after) > 2:
        rgb_after, depth_after, objects_after = move_object(
            rgb_after, depth_after, objects_after, 2, dx=0, dy=-50
        )
        movements.append("Object 2 moved 50 pixels up")
        print(f"  📦 {movements[-1]}")
    
    # Move object 3 diagonally
    if len(objects_after) > 3:
        rgb_after, depth_after, objects_after = move_object(
            rgb_after, depth_after, objects_after, 3, dx=-60, dy=40
        )
        movements.append("Object 3 moved diagonally")
        print(f"  📦 {movements[-1]}")
    
    # Save "after" sequence
    after_dir = output_path / 'after'
    after_dir.mkdir(exist_ok=True)
    
    print("\n📸 Generating 'after' frames...")
    for i in range(num_frames):
        # Save RGB
        rgb_path = after_dir / f'rgb_{i:06d}.jpg'
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_after, cv2.COLOR_RGB2BGR))
        
        # Save depth
        depth_path = after_dir / f'depth_{i:06d}.png'
        cv2.imwrite(str(depth_path), depth_after)
        
        print(f"  ✅ Frame {i}: {rgb_path.name}")
    
    # Save metadata
    metadata = {
        'sequence_id': 'demo_seq_001',
        'num_frames': num_frames,
        'camera_intrinsics': {
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 240.0
        },
        'objects_before': [
            {
                'id': i,
                'bbox': obj['bbox'],
                'center': obj['center'].tolist(),
                'label': obj['label']
            }
            for i, obj in enumerate(objects_before)
        ],
        'objects_after': [
            {
                'id': i,
                'bbox': obj['bbox'],
                'center': obj['center'].tolist(),
                'label': obj['label']
            }
            for i, obj in enumerate(objects_after)
        ],
        'ground_truth_changes': movements
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Metadata saved: {metadata_path}")
    
    # Create visualization
    print("\n🎨 Creating comparison visualization...")
    h, w = rgb_before.shape[:2]
    vis = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    vis[:, :w] = rgb_before
    vis[:, w+20:] = rgb_after
    
    # Add labels
    cv2.putText(vis, 'BEFORE', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(vis, 'AFTER', (w + 40, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Add change descriptions
    y_offset = h - 120
    cv2.putText(vis, 'Ground Truth Changes:', (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    for i, change in enumerate(movements):
        y_offset += 25
        cv2.putText(vis, f'- {change}', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    vis_path = output_path / 'comparison.jpg'
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"💾 Visualization saved: {vis_path}")
    
    print("\n" + "="*60)
    print("✅ Demo sequence created successfully!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Location: {output_path}")
    print(f"   Frames: {num_frames} before + {num_frames} after")
    print(f"   Objects: {len(objects_before)}")
    print(f"   Changes: {len(movements)}")
    print(f"\n🔍 Ground truth changes:")
    for change in movements:
        print(f"   • {change}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test your model on this demo data:")
    print(f"      python test_temporal_tracking.py --sequence {output_path}")
    print(f"   2. View comparison: open {vis_path}")
    print(f"   3. Once Replica downloads, generate more realistic data")
    
    return output_path


def test_temporal_tracking(sequence_path='./demo_data'):
    """
    Quick test of temporal tracking on demo data
    """
    from temporal_g2vlm import ObjectTracker
    
    print("🧪 Testing Object Tracker...")
    print("="*60)
    
    # Load metadata
    with open(Path(sequence_path) / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Initialize tracker
    tracker = ObjectTracker(distance_threshold=50.0)  # Pixels
    
    # Track objects in "before" frames
    print("\n📍 Tracking objects in 'before' frames...")
    objects_before = metadata['objects_before']
    tracked_before = tracker.update(objects_before, frame_id=0)
    
    print(f"   Tracked {len(tracked_before)} objects")
    for obj in tracked_before:
        print(f"   • Object {obj['id']} ({obj['label']}): {obj['center']}")
    
    # Track objects in "after" frames
    print("\n📍 Tracking objects in 'after' frames...")
    objects_after = metadata['objects_after']
    tracked_after = tracker.update(objects_after, frame_id=1)
    
    # Detect changes
    changes = tracker.get_changes(frame_id=1, lookback=1)
    
    print(f"\n🔍 Detected Changes:")
    print("="*60)
    
    if changes:
        for change in changes:
            obj_label = tracked_after[change.object_id]['label']
            distance = np.linalg.norm(change.translation)
            print(f"  ✅ {obj_label}:")
            print(f"     Type: {change.change_type}")
            print(f"     Distance moved: {distance:.1f} pixels")
            print(f"     Translation: {change.translation}")
            print(f"     Confidence: {change.confidence:.2f}")
    else:
        print("  No significant changes detected")
    
    print("\n📊 Ground Truth Changes:")
    print("="*60)
    for i, gt_change in enumerate(metadata['ground_truth_changes']):
        print(f"  {i+1}. {gt_change}")
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Create demo data for Temporal G2VLM")
    parser.add_argument('--output', type=str, default='./demo_data',
                       help='Output directory')
    parser.add_argument('--num-frames', type=int, default=8,
                       help='Number of frames per sequence')
    parser.add_argument('--test', action='store_true',
                       help='Run tracking test after generating data')
    
    args = parser.parse_args()
    
    # Create demo sequence
    output_path = create_demo_sequence(args.output, args.num_frames)
    
    # Run test if requested
    if args.test:
        print("\n" + "="*60)
        test_temporal_tracking(output_path)


# Quick usage:
"""
# Create demo data (works WITHOUT Replica!)
python quick_demo.py --output ./demo_data --num-frames 8

# Create demo data AND test tracking
python quick_demo.py --output ./demo_data --test

# Now you can develop while Replica downloads!
"""
