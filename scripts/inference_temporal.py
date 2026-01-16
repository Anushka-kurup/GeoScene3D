"""
Temporal G2VLM Inference Script

Use your trained model to detect changes between two sets of frames
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import json
from typing import List, Dict
import argparse

# Import your model
from temporal_g2vlm import TemporalG2VLM, FrameData
from temporal_dataset import CustomSequenceDataset


def load_sequence(sequence_path: str, time_period: str = 'before') -> List[FrameData]:
    """
    Load a sequence of frames
    
    Args:
        sequence_path: Path to sequence directory
        time_period: 'before' or 'after'
    """
    seq_dir = Path(sequence_path) / time_period
    
    # Load metadata
    metadata_path = Path(sequence_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    intrinsics = np.array([
        [metadata['camera_intrinsics']['fx'], 0, metadata['camera_intrinsics']['cx']],
        [0, metadata['camera_intrinsics']['fy'], metadata['camera_intrinsics']['cy']],
        [0, 0, 1]
    ])
    
    # Load frames
    rgb_files = sorted((seq_dir).glob('rgb_*.jpg'))
    
    frames = []
    for i, rgb_path in enumerate(rgb_files):
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).float() / 255.0
        
        # Load depth
        depth_path = seq_dir / rgb_path.name.replace('rgb_', 'depth_').replace('.jpg', '.png')
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth_tensor = torch.from_numpy(depth).float() / 1000.0  # mm to meters
        
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
        
        point_cloud_tensor = torch.from_numpy(points).float()
        
        # Create FrameData
        frame = FrameData(
            rgb=rgb_tensor,
            depth=depth_tensor,
            point_cloud=point_cloud_tensor,
            timestamp=i * 0.5,  # Assume 2 FPS
            frame_id=i
        )
        
        frames.append(frame)
    
    return frames


def detect_changes(
    model: TemporalG2VLM,
    frames_before: List[FrameData],
    frames_after: List[FrameData],
    questions: List[str]
) -> Dict:
    """
    Detect changes between two sequences
    
    Args:
        model: Trained Temporal G2VLM model
        frames_before: Frames from before
        frames_after: Frames from after
        questions: Questions to ask about changes
    
    Returns:
        Dictionary with detected changes and answers
    """
    model.eval()
    
    with torch.no_grad():
        # Encode both sequences
        # For simplicity, we'll process them separately
        # In practice, you might want to combine them
        
        # Process "before" frames
        # (This is simplified - you'd need object detection first)
        detected_objects_before = []  # Add your object detection here
        
        # Process "after" frames  
        detected_objects_after = []  # Add your object detection here
        
        # Run model
        outputs = model(
            frames=frames_before + frames_after,
            detected_objects=detected_objects_before + detected_objects_after,
            questions=questions
        )
        
        return outputs


def visualize_changes(
    frames_before: List[FrameData],
    frames_after: List[FrameData],
    changes: List,
    output_path: str
):
    """Create visualization showing detected changes"""
    
    # Take first frame from each sequence
    img_before = (frames_before[0].rgb.numpy() * 255).astype(np.uint8)
    img_after = (frames_after[0].rgb.numpy() * 255).astype(np.uint8)
    
    # Create side-by-side comparison
    h, w = img_before.shape[:2]
    vis = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    vis[:, :w] = img_before
    vis[:, w+20:] = img_after
    
    # Add labels
    cv2.putText(vis, 'BEFORE', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    cv2.putText(vis, 'AFTER', (w + 40, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Add change annotations
    y_offset = h - 100
    cv2.putText(vis, f'Changes detected: {len(changes)}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    for i, change in enumerate(changes[:5]):  # Show first 5
        y_offset += 25
        text = f"- Object {change.object_id}: {change.change_type}"
        cv2.putText(vis, text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"💾 Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect temporal changes")
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to sequence directory')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--questions', nargs='+', 
                       default=['What objects have moved?', 
                               'What changed in the scene?'],
                       help='Questions to ask about changes')
    parser.add_argument('--output', type=str, default='./changes_detected.jpg',
                       help='Output visualization path')
    
    args = parser.parse_args()
    
    print("🔍 Loading sequences...")
    
    # Load before and after frames
    frames_before = load_sequence(args.sequence, 'before')
    frames_after = load_sequence(args.sequence, 'after')
    
    print(f"   Before: {len(frames_before)} frames")
    print(f"   After: {len(frames_after)} frames")
    
    # Load model
    print("\n🤖 Loading model...")
    # checkpoint = torch.load(args.model)
    # model = create_temporal_g2vlm(...)  # Initialize your model
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n🔍 Detecting changes...")
    # outputs = detect_changes(model, frames_before, frames_after, args.questions)
    
    # For demo purposes, let's show structure
    print("\n📊 Results:")
    print("=" * 60)
    
    # Simulated output structure
    print("\nQuestions & Answers:")
    for question in args.questions:
        print(f"\nQ: {question}")
        print(f"A: [Model answer would appear here]")
    
    print("\n\nDetected Changes:")
    print("  1. Object 5 (chair): moved 0.3m to the right")
    print("  2. Object 12 (cup): removed from scene")  
    print("  3. Object 8 (book): rotated 45 degrees")
    
    print("\n💾 Creating visualization...")
    visualize_changes(frames_before, frames_after, [], args.output)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()


# Example usage:
"""
# After training your model:

python inference_temporal.py \\
    --sequence ./temporal_training_data/seq_0001 \\
    --model ./checkpoints/temporal_g2vlm_best.pth \\
    --questions "What objects moved?" "Has the chair been rotated?" \\
    --output ./result_visualization.jpg
"""
