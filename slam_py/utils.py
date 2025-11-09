import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def load_tum_dataset(dataset_path):
    
    dataset_path = Path(dataset_path)
    
    # Read association file or create association
    rgb_txt = dataset_path / 'rgb.txt'
    depth_txt = dataset_path / 'depth.txt'
    
    if not rgb_txt.exists() or not depth_txt.exists():
        raise ValueError(f"Could not find rgb.txt or depth.txt in {dataset_path}")
    
    # Parse RGB file
    rgb_data = []
    with open(rgb_txt, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                filename = parts[1]
                rgb_data.append((timestamp, dataset_path / filename))
    
    # Parse depth file
    depth_data = []
    with open(depth_txt, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                filename = parts[1]
                depth_data.append((timestamp, dataset_path / filename))
    
    # Associate RGB and depth by timestamp
    rgb_files = []
    depth_files = []
    timestamps = []
    
    max_time_diff = 0.02  # 20ms tolerance
    
    for rgb_ts, rgb_file in rgb_data:
        # Find closest depth image
        min_diff = float('inf')
        best_depth = None
        
        for depth_ts, depth_file in depth_data:
            diff = abs(rgb_ts - depth_ts)
            if diff < min_diff:
                min_diff = diff
                best_depth = depth_file
        
        if min_diff < max_time_diff and best_depth is not None:
            rgb_files.append(rgb_file)
            depth_files.append(best_depth)
            timestamps.append(rgb_ts)
    
    return rgb_files, depth_files, timestamps


def draw_trajectory(poses, output_path=None):
    
    if len(poses) == 0:
        return None
    
    # Extract positions
    positions = np.array([pose[:3, 3] for pose in poses])
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # 3D plot
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True)
    
    # XY plot
    ax = fig.add_subplot(222)
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top View (XY)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # XZ plot
    ax = fig.add_subplot(223)
    ax.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Side View (XZ)')
    ax.legend()
    ax.grid(True)
    
    # YZ plot
    ax = fig.add_subplot(224)
    ax.plot(positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Side View (YZ)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to {output_path}")
    
    return fig