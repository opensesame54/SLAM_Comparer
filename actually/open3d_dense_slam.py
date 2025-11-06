import open3d as o3d
import numpy as np
import os
import sys
from PIL import Image

# ----------------- CONFIGURATION -----------------
DATASET_BASE_PATH = "rgbd_dataset_freiburg1_desk2" 
ASSOCIATIONS_FILE = "associations.txt"
OUTPUT_PLY_FILE = "dense_slam_reconstruction.ply"
TRAJECTORY_FILE = "dense_slam_trajectory.txt"

# Camera parameters for TUM Freiburg1 dataset
FX = 517.3
FY = 516.5
CX = 318.6
CY = 255.3
DEPTH_SCALE = 5000.0  # TUM depth factor: 5000

# Open3D SLAM Configuration
VOXEL_SIZE = 0.01  # 10mm resolution
MAX_DEPTH = 2.5     # Maximum depth to consider
MIN_DEPTH = 0.2     # Minimum depth to consider
BLOCK_COUNT = 1000  # Number of Voxel blocks to allocate
DEVICE = o3d.core.Device("CPU:0")


# ----------------- Utility Functions -----------------

def load_associations(base_path, file_path):
    """Loads synchronized RGB-D file paths with validation."""
    full_path = os.path.join(base_path, file_path)
    pairs = []
    
    try:
        with open(full_path, "r") as f:
            for i, line in enumerate(f):
                if line.startswith("#") or len(line.strip().split()) < 4: 
                    continue
                parts = line.strip().split()
                rgb_file = os.path.join(base_path, parts[1])
                depth_file = os.path.join(base_path, parts[3])
                
                if not os.path.isfile(rgb_file):
                    print(f"[WARNING] Missing RGB file: {rgb_file}")
                    continue
                if not os.path.isfile(depth_file):
                    print(f"[WARNING] Missing depth file: {depth_file}")
                    continue
                    
                pairs.append((rgb_file, depth_file))
                
    except FileNotFoundError:
        print(f"[ERROR] Associations file not found at {full_path}.")
        print("[INFO] Make sure you are running this script from the PARENT directory")
        print(f"[INFO] (i.e., the script and the '{DATASET_BASE_PATH}' folder are in the same directory).")
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(pairs)} valid association pairs")
    return pairs


def filter_depth(depth_tensor, depth_scale, min_depth, max_depth):
    """Filter depth values outside valid range."""
    try:
        depth_array = depth_tensor.as_tensor().cpu().numpy()

        if depth_array.size == 0:
            print("[WARNING] Depth filtering received an empty numpy array.")
            return o3d.t.geometry.Image().to(depth_tensor.device)
        
        min_val = min_depth * depth_scale
        max_val = max_depth * depth_scale
        
        filtered_array = depth_array.copy()
        mask = (filtered_array < min_val) | (filtered_array > max_val) | (filtered_array == 0)
        filtered_array[mask] = 0
        
        # ------------------- FIX 1 -------------------
        # Force a C-contiguous memory layout before tensor conversion
        contiguous_array = np.ascontiguousarray(filtered_array)
        filtered_depth = o3d.t.geometry.Image(
            o3d.core.Tensor(contiguous_array, device=depth_tensor.device)
        )
        # ---------------------------------------------
        return filtered_depth
    
    except Exception as e:
        print(f"[WARNING] Error in depth filtering: {e}")
        return o3d.t.geometry.Image().to(depth_tensor.device)


def is_valid_pose(pose):
    """Check if pose transformation matrix is valid."""
    if np.isnan(pose).any():
        return False
    if np.abs(pose).max() > 1e3:
        return False
    R = pose[:3, :3]
    if np.abs(np.linalg.det(R)) < 0.1:
        return False
    return True


def calculate_pose_change(pose1, pose2):
    """Calculate the magnitude of pose change between two frames."""
    translation_change = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    return translation_change


# ----------------- Main Dense SLAM Function -----------------

def run_dense_slam():
    pairs = load_associations(DATASET_BASE_PATH, ASSOCIATIONS_FILE)
    if not pairs:
        print("[ERROR] No image pairs loaded. Exiting.")
        return

    assert FX > 0 and FY > 0 and CX > 0 and CY > 0, "[ERROR] Invalid camera intrinsics"
    
    intrinsic = o3d.core.Tensor(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], 
        o3d.core.Dtype.Float64, DEVICE
    )
    
    print(f"[INFO] Camera Intrinsics:")
    print(intrinsic.cpu().numpy())
    print(f"[INFO] Depth Scale: {DEPTH_SCALE}")
    print(f"[INFO] Depth Range: {MIN_DEPTH}m - {MAX_DEPTH}m")
    print(f"[INFO] Voxel Size: {VOXEL_SIZE}m")
    
    T_frame_to_model = o3d.core.Tensor(np.identity(4), o3d.core.Dtype.Float64, DEVICE)

    model = o3d.t.pipelines.slam.Model(
        VOXEL_SIZE, 16, BLOCK_COUNT, T_frame_to_model, DEVICE
    )
    
    # Initialize frames using first depth image dimensions
    try:
        depth_ref_pil = Image.open(pairs[0][1])
        depth_ref_np = np.ascontiguousarray(np.asarray(depth_ref_pil)) # <-- Added contiguous
        if depth_ref_np.size == 0:
            print("[ERROR] Failed to load first depth image. Exiting.")
            sys.exit(1)
        rows, cols = depth_ref_np.shape
        print(f"[INFO] Initializing frames with resolution: {cols}x{rows}")
    except Exception as e:
        print(f"[ERROR] Could not load initial depth image {pairs[0][1]}: {e}")
        sys.exit(1)

    input_frame = o3d.t.pipelines.slam.Frame(
        rows, cols, intrinsic, DEVICE
    )
    raycast_frame = o3d.t.pipelines.slam.Frame(
        rows, cols, intrinsic, DEVICE
    )
    
    poses = [np.identity(4)]
    frame_stats = {
        'processed': 0,
        'skipped': 0,
        'tracking_failed': 0,
        'invalid_pose': 0
    }

    print("\n[INFO] Starting Open3D Dense SLAM (T-SLAM)...")
    print("=" * 60)

    for i, (color_path, depth_path) in enumerate(pairs):
        if i % 50 == 0 and i > 0:
            print(f"[INFO] Processing frame {i}/{len(pairs)} - " +
                  f"Processed: {frame_stats['processed']}, " +
                  f"Skipped: {frame_stats['skipped']}")

        try:
            # Load images using Pillow (PIL)
            try:
                depth_pil = Image.open(depth_path)
                color_pil = Image.open(color_path)
            except Exception as e:
                print(f"[WARNING] Pillow failed to load images for frame {i}: {e}. Skipping.")
                poses.append(poses[-1])
                frame_stats['skipped'] += 1
                continue

            # ------------------- FIX 2 -------------------
            # Force C-contiguous arrays
            depth_np = np.ascontiguousarray(np.asarray(depth_pil))
            color_np = np.ascontiguousarray(np.asarray(color_pil))
            # ---------------------------------------------
            
            if depth_np.size == 0 or color_np.size == 0:
                print(f"[WARNING] Loaded empty image data for frame {i}. Skipping.")
                poses.append(poses[-1])
                frame_stats['skipped'] += 1
                continue

            if i == 0:
                print("\n[DIAGNOSTIC] First frame loaded successfully:")
                print(f"  Depth shape: {depth_np.shape}, dtype: {depth_np.dtype}")
                print(f"  Color shape: {color_np.shape}, dtype: {color_np.dtype}")
                print("="*60)

            # Use the main Tensor constructor
            depth = o3d.t.geometry.Image(
                o3d.core.Tensor(depth_np, device=DEVICE)
            )
            color = o3d.t.geometry.Image(
                o3d.core.Tensor(color_np, device=DEVICE)
            )
            
            depth = filter_depth(depth, DEPTH_SCALE, MIN_DEPTH, MAX_DEPTH)

            if depth.is_empty():
                 print(f"[WARNING] Frame {i} has no valid depth after filtering. Skipping.")
                 poses.append(poses[-1])
                 frame_stats['skipped'] += 1
                 continue
            
            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)
            
            if i == 0:
                model.integrate(input_frame, DEPTH_SCALE, MAX_DEPTH)
                model.synthesize_frame(raycast_frame, T_frame_to_model, DEPTH_SCALE, MAX_DEPTH)
                frame_stats['processed'] += 1
                print(f"[INFO] Frame 0: Initialized with identity pose")
                
            else:
                result = model.track_frame_to_model(
                    input_frame, raycast_frame, DEPTH_SCALE, MAX_DEPTH
                )
                
                T_world_curr = result.transformation.cpu().numpy()
                
                if not is_valid_pose(T_world_curr):
                    print(f"[WARNING] Invalid pose at frame {i}. Skipping integration.")
                    poses.append(poses[-1])
                    frame_stats['invalid_pose'] += 1
                    continue
                
                pose_change = calculate_pose_change(poses[-1], T_world_curr)
                if pose_change > 0.5:
                    print(f"[WARNING] Large pose jump at frame {i}: {pose_change:.3f}m. " +
                          "Possible tracking failure.")
                    frame_stats['tracking_failed'] += 1
                
                poses.append(T_world_curr)
                input_frame.set_pose(o3d.core.Tensor(T_world_curr, o3d.core.Dtype.Float64, DEVICE))
                model.integrate(input_frame, DEPTH_SCALE, MAX_DEPTH)
                model.synthesize_frame(raycast_frame, input_frame.get_pose(), DEPTH_SCALE, MAX_DEPTH)
                frame_stats['processed'] += 1
                
                if i % 100 == 0:
                    print(f"[INFO] Frame {i}: Pose translation norm = {np.linalg.norm(T_world_curr[:3, 3]):.3f}")

        except Exception as e:
            print(f"[ERROR] Failed to process frame {i}: {e}")
            poses.append(poses[-1])
            frame_stats['skipped'] += 1
            continue

    print("\n" + "=" * 60)
    print("[INFO] SLAM Processing Complete!")
    print(f"  Total frames: {len(pairs)}")
    print(f"  Successfully processed: {frame_stats['processed']}")
    print(f"  Skipped: {frame_stats['skipped']}")
    print(f"  Invalid poses: {frame_stats['invalid_pose']}")
    print(f"  Tracking failures: {frame_stats['tracking_failed']}")

    print("\n[INFO] Extracting final point cloud...")
    try:
        pcd = model.extract_pointcloud()
        # Note: Corrected a typo here, 'pd' to 'pcd'
        points = np.asarray(pcd.point.positions.cpu().numpy())
        print(f"[INFO] Point cloud contains {len(points)} points")
        
        if len(points) > 0:
            print(f"[INFO] Point cloud bounds:")
            print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            o3d.io.write_point_cloud(OUTPUT_PLY_FILE, pcd)
            print(f"\n[INFO] ✓ Dense point cloud saved to: {OUTPUT_PLY_FILE}")
        else:
            print(f"[WARNING] No points extracted. Output file will be empty.")
            
    except Exception as e:
        print(f"[ERROR] Failed to extract point cloud: {e}")
    
    with open(TRAJECTORY_FILE, "w") as f:
        for idx, pose in enumerate(poses):
            f.write(f"# Frame {idx}\n")
            f.write(" ".join(map(str, pose.flatten())) + "\n")
    print(f"[INFO] ✓ Full trajectory saved to: {TRAJECTORY_FILE}")

    print("\n" + "=" * 60)
    print("[INFO] SLAM COMPLETE - Ready for MeshLab Visualization")
    print("=" * 60)
    print(f"\nTo view in MeshLab:")
    print(f"  1. Open MeshLab")
    print(f"  2. File → Import Mesh → Select '{OUTPUT_PLY_FILE}'")
    print("=" * 60)


if __name__ == '__main__':
    run_dense_slam()