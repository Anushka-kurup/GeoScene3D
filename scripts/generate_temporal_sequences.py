"""
Generate Temporal Training Sequences from Replica Dataset

This script creates synthetic "before/after" scenarios by:
1. Rendering views of a Replica scene
2. Simulating object movements
3. Rendering the same views again
4. Saving as temporal sequences for training

Requirements:
    pip install habitat-sim opencv-python numpy
"""

import habitat_sim
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class ReplicaTemporalGenerator:
    """Generate temporal sequences from Replica scenes"""
    
    def __init__(self, scene_path: str, output_dir: str):
        """
        Args:
            scene_path: Path to Replica scene mesh (e.g., apartment_0/mesh.ply)
            output_dir: Where to save generated sequences
        """
        self.scene_path = scene_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Habitat-Sim
        self.sim = self._init_simulator()
        
    def _init_simulator(self) -> habitat_sim.Simulator:
        """Initialize Habitat simulator"""
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = True
        
        # RGB sensor
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # 1.5m height
        
        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [480, 640]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        return sim
    
    def generate_camera_trajectory(
        self, 
        num_waypoints: int = 8,
        room_bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5)
    ) -> List[Dict]:
        """
        Generate camera waypoints for a trajectory
        
        Args:
            num_waypoints: Number of camera positions
            room_bounds: (min_x, max_x, min_z, max_z)
        
        Returns:
            List of camera poses (position + rotation)
        """
        min_x, max_x, min_z, max_z = room_bounds
        
        waypoints = []
        for i in range(num_waypoints):
            # Random position in room
            x = np.random.uniform(min_x, max_x)
            z = np.random.uniform(min_z, max_z)
            y = 1.5  # Eye height
            
            # Random rotation (look around)
            yaw = np.random.uniform(0, 2 * np.pi)
            
            waypoints.append({
                'position': [x, y, z],
                'rotation': yaw
            })
        
        return waypoints
    
    def capture_frame(self, agent_state: habitat_sim.AgentState) -> Dict:
        """Capture RGB-D frame at given agent state"""
        self.sim.agents[0].set_state(agent_state)
        
        observations = self.sim.get_sensor_observations()
        
        rgb = observations['color'][:, :, :3]  # Remove alpha
        depth = observations['depth']
        
        return {
            'rgb': rgb,
            'depth': depth,
            'position': agent_state.position.tolist(),
            'rotation': agent_state.rotation
        }
    
    def save_sequence(
        self,
        frames_before: List[Dict],
        frames_after: List[Dict],
        sequence_id: str,
        changes_description: str
    ):
        """Save a temporal sequence"""
        seq_dir = self.output_dir / sequence_id
        seq_dir.mkdir(exist_ok=True)
        
        # Save "before" frames
        before_dir = seq_dir / 'before'
        before_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames_before):
            # RGB
            rgb_path = before_dir / f'rgb_{i:06d}.jpg'
            cv2.imwrite(str(rgb_path), cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))
            
            # Depth (save as 16-bit PNG, millimeters)
            depth_path = before_dir / f'depth_{i:06d}.png'
            depth_mm = (frame['depth'] * 1000).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm)
        
        # Save "after" frames
        after_dir = seq_dir / 'after'
        after_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames_after):
            rgb_path = after_dir / f'rgb_{i:06d}.jpg'
            cv2.imwrite(str(rgb_path), cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))
            
            depth_path = after_dir / f'depth_{i:06d}.png'
            depth_mm = (frame['depth'] * 1000).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_mm)
        
        # Save metadata
        metadata = {
            'sequence_id': sequence_id,
            'num_frames': len(frames_before),
            'changes_description': changes_description,
            'camera_intrinsics': {
                'fx': 320.0,  # Approximate
                'fy': 320.0,
                'cx': 320.0,
                'cy': 240.0
            },
            'frames_before': [
                {
                    'frame_id': i,
                    'position': frame['position'],
                    'rotation': [frame['rotation'].x, frame['rotation'].y, 
                                frame['rotation'].z, frame['rotation'].w]
                }
                for i, frame in enumerate(frames_before)
            ],
            'frames_after': [
                {
                    'frame_id': i,
                    'position': frame['position'],
                    'rotation': [frame['rotation'].x, frame['rotation'].y,
                                frame['rotation'].z, frame['rotation'].w]
                }
                for i, frame in enumerate(frames_after)
            ]
        }
        
        with open(seq_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved sequence: {sequence_id}")
        print(f"   Changes: {changes_description}")
    
    def generate_sequence(
        self,
        sequence_id: str,
        num_frames: int = 8,
        simulate_changes: bool = True
    ):
        """
        Generate one temporal sequence
        
        Args:
            sequence_id: Unique identifier
            num_frames: Number of frames to capture
            simulate_changes: Whether to simulate object movements
        """
        # Generate camera trajectory
        waypoints = self.generate_camera_trajectory(num_frames)
        
        # Capture "before" frames
        frames_before = []
        for waypoint in waypoints:
            agent_state = habitat_sim.AgentState()
            agent_state.position = waypoint['position']
            agent_state.rotation = habitat_sim.utils.quat_from_angle_axis(
                waypoint['rotation'], np.array([0, 1, 0])
            )
            
            frame = self.capture_frame(agent_state)
            frames_before.append(frame)
        
        # Simulate changes (in real version, you'd move objects)
        # For now, just add small camera perturbation as demo
        changes_description = "No changes (demo)" if not simulate_changes else "Camera slightly moved"
        
        # Capture "after" frames (same trajectory)
        frames_after = []
        for waypoint in waypoints:
            agent_state = habitat_sim.AgentState()
            
            if simulate_changes:
                # Add small random perturbation
                perturbed_pos = [
                    waypoint['position'][0] + np.random.normal(0, 0.1),
                    waypoint['position'][1],
                    waypoint['position'][2] + np.random.normal(0, 0.1)
                ]
                agent_state.position = perturbed_pos
            else:
                agent_state.position = waypoint['position']
            
            agent_state.rotation = habitat_sim.utils.quat_from_angle_axis(
                waypoint['rotation'], np.array([0, 1, 0])
            )
            
            frame = self.capture_frame(agent_state)
            frames_after.append(frame)
        
        # Save sequence
        self.save_sequence(frames_before, frames_after, sequence_id, changes_description)
    
    def generate_dataset(self, num_sequences: int = 100):
        """Generate multiple sequences"""
        print(f"🎬 Generating {num_sequences} temporal sequences...")
        print(f"📁 Output: {self.output_dir}")
        
        for i in range(num_sequences):
            sequence_id = f"seq_{i:04d}"
            self.generate_sequence(sequence_id, simulate_changes=(i % 2 == 0))
        
        print(f"\n✅ Generated {num_sequences} sequences!")
        print(f"📊 Dataset structure:")
        print(f"   {self.output_dir}/")
        print(f"   ├── seq_0000/")
        print(f"   │   ├── before/ (RGB + Depth)")
        print(f"   │   ├── after/ (RGB + Depth)")
        print(f"   │   └── metadata.json")
        print(f"   ├── seq_0001/")
        print(f"   └── ...")


def main():
    parser = argparse.ArgumentParser(description="Generate temporal sequences from Replica")
    parser.add_argument('--scene', type=str, required=True,
                       help='Path to Replica scene mesh (e.g., replica_data/apartment_0/mesh.ply)')
    parser.add_argument('--output', type=str, default='./temporal_sequences',
                       help='Output directory for sequences')
    parser.add_argument('--num-sequences', type=int, default=100,
                       help='Number of sequences to generate')
    parser.add_argument('--frames-per-seq', type=int, default=8,
                       help='Number of frames per sequence')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ReplicaTemporalGenerator(args.scene, args.output)
    
    # Generate dataset
    generator.generate_dataset(args.num_sequences)


if __name__ == '__main__':
    main()


# Example usage after Replica downloads:
"""
# Install requirements
pip install habitat-sim opencv-python

# Generate sequences from apartment_0
python generate_temporal_sequences.py \\
    --scene replica_data/apartment_0/mesh.ply \\
    --output ./temporal_training_data \\
    --num-sequences 100

# This creates:
temporal_training_data/
├── seq_0000/
│   ├── before/
│   │   ├── rgb_000000.jpg
│   │   ├── depth_000000.png
│   │   └── ...
│   ├── after/
│   │   ├── rgb_000000.jpg
│   │   ├── depth_000000.png
│   │   └── ...
│   └── metadata.json
└── ...
"""
