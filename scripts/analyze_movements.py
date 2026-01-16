"""
Simple script to visualize and verify object movements in demo data
"""

import json
import numpy as np
from pathlib import Path
import cv2


def analyze_movements(sequence_path='./demo_data'):
    """Analyze object movements between before and after"""
    
    print("🔍 Analyzing Object Movements")
    print("="*60)
    
    # Load metadata
    with open(Path(sequence_path) / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    objects_before = metadata['objects_before']
    objects_after = metadata['objects_after']
    
    print(f"\n📊 Scene Info:")
    print(f"   Objects: {len(objects_before)}")
    print(f"   Frames: {metadata['num_frames']}")
    
    # Match objects by ID and compute movements
    print(f"\n📍 Detected Movements:")
    print("="*60)
    
    for i in range(len(objects_before)):
        obj_before = objects_before[i]
        obj_after = objects_after[i]
        
        center_before = np.array(obj_before['center'])
        center_after = np.array(obj_after['center'])
        
        # Compute movement
        displacement = center_after - center_before
        distance = np.linalg.norm(displacement)
        
        print(f"\n  Object {i} ({obj_before['label']}):")
        print(f"    Before: {center_before}")
        print(f"    After:  {center_after}")
        print(f"    Displacement: [{displacement[0]:+.1f}, {displacement[1]:+.1f}] pixels")
        print(f"    Distance: {distance:.1f} pixels")
        
        if distance > 5:  # Significant movement
            direction = ""
            if abs(displacement[0]) > abs(displacement[1]):
                direction = "RIGHT" if displacement[0] > 0 else "LEFT"
            else:
                direction = "DOWN" if displacement[1] > 0 else "UP"
            
            print(f"    ✅ MOVED {direction}")
        else:
            print(f"    ⭕ No significant movement")
    
    # Compare with ground truth
    print(f"\n📋 Ground Truth:")
    print("="*60)
    for i, change in enumerate(metadata['ground_truth_changes']):
        print(f"  {i+1}. {change}")
    
    # Show visualization
    comparison_path = Path(sequence_path) / 'comparison.jpg'
    print(f"\n🖼️  Visual comparison saved at:")
    print(f"   {comparison_path}")
    print(f"\n💡 To view: open {comparison_path}")
    
    # Create a detailed movement visualization
    create_movement_viz(sequence_path, objects_before, objects_after)


def create_movement_viz(sequence_path, objects_before, objects_after):
    """Create visualization showing movement vectors"""
    
    # Load before image
    before_img_path = Path(sequence_path) / 'before' / 'rgb_000000.jpg'
    img = cv2.imread(str(before_img_path))
    
    # Draw movement arrows
    for i in range(len(objects_before)):
        obj_before = objects_before[i]
        obj_after = objects_after[i]
        
        center_before = tuple(map(int, obj_before['center']))
        center_after = tuple(map(int, obj_after['center']))
        
        # Draw arrow showing movement
        cv2.arrowedLine(img, center_before, center_after, (0, 255, 0), 3, tipLength=0.3)
        
        # Draw circles at start and end
        cv2.circle(img, center_before, 8, (255, 0, 0), -1)  # Blue = before
        cv2.circle(img, center_after, 8, (0, 0, 255), -1)   # Red = after
        
        # Add label
        cv2.putText(img, f"Obj {i}", 
                   (center_before[0] - 20, center_before[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add legend
    cv2.circle(img, (30, 30), 8, (255, 0, 0), -1)
    cv2.putText(img, "Before", (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.circle(img, (30, 60), 8, (0, 0, 255), -1)
    cv2.putText(img, "After", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save
    output_path = Path(sequence_path) / 'movement_vectors.jpg'
    cv2.imwrite(str(output_path), img)
    
    print(f"\n📍 Movement vectors visualization saved:")
    print(f"   {output_path}")


if __name__ == '__main__':
    import sys
    
    sequence_path = sys.argv[1] if len(sys.argv) > 1 else './demo_data'
    analyze_movements(sequence_path)
    
    print("\n✅ Analysis complete!")


# Usage:
"""
python analyze_movements.py ./demo_data
"""
