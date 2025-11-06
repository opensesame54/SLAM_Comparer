#!/usr/bin/env python3

import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
import bisect
import time

# --- 1. SET YOUR FILE PATHS AND PARAMETERS HERE ---

DATASET_DIR = "/home/aashrith/Desktop/project_cv/actually/rgbd_dataset_freiburg1_room"
KEYFRAME_FILE = "/home/aashrith/Desktop/project_cv/KeyFrameTrajectory.txt"
OUTPUT_PLY = "/home/aashrith/Desktop/project_cv/dense_point_cloud_room_FULL.ply"
ASSOCIATIONS_FILE = os.path.join(DATASET_DIR, "associations.txt")

# !!! --- TUNE THIS PARAMETER ---
# This is the gap between keyframes for stereo pairs.
# A smaller value (e.g., 20) is slower but denser (more overlap).
# A larger value (e.g., 50) is faster but might have gaps.
# Start with 50.
KEYFRAME_STEP_SIZE = 1 

# Camera Intrinsics for TUM Freiburg1
K = np.array([
    [517.3, 0.0,   318.6],
    [0.0,   516.5, 255.3],
    [0.0,   0.0,   1.0  ]
])

# Distortion Coefficients for TUM Freiburg1
DIST_COEFFS = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


# --- 2. HELPER FUNCTIONS (Unchanged) ---

def read_associations(associations_file_path):
    """
    Reads an associations.txt file (TUM format: rgb_ts rgb_file depth_ts depth_file)
    Returns:
        rgb_map (dict): {rgb_timestamp_str: rgb_filename_str}
        rgb_timestamps_sorted (list): A sorted list of rgb_timestamp_str
        rgb_timestamps_float (list): A sorted list of rgb_timestamp_float
    """
    rgb_map = {}
    timestamps_str_float = []
    
    with open(associations_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 4:
                continue
            
            rgb_timestamp_str = parts[0]
            rgb_filename = parts[1]
            
            if rgb_timestamp_str not in rgb_map:
                rgb_map[rgb_timestamp_str] = rgb_filename
                timestamps_str_float.append((rgb_timestamp_str, float(rgb_timestamp_str)))

    timestamps_str_float.sort(key=lambda x: x[1])
    rgb_timestamps_sorted = [ts_str for ts_str, ts_float in timestamps_str_float]
    rgb_timestamps_float = [ts_float for ts_str, ts_float in timestamps_str_float]
    
    return rgb_map, rgb_timestamps_sorted, rgb_timestamps_float


def read_poses(pose_file_path):
    """
    Reads a TUM-format trajectory file (timestamp tx ty tz qx qy qz qw)
    Returns a dictionary mapping timestamp (str) to 4x4 pose matrix (np.array)
    """
    poses = {}
    with open(pose_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 8:
                continue
            
            timestamp = parts[0]
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            
            R_matrix = Rotation.from_quat(q).as_matrix()
            pose = np.eye(4)
            pose[0:3, 0:3] = R_matrix
            pose[0:3, 3] = t
            poses[timestamp] = pose
            
    return poses

def find_closest_timestamp(target_ts_str, available_ts_sorted_str, available_ts_sorted_float):
    """
    Finds the closest timestamp in the available list to the target timestamp.
    """
    if not available_ts_sorted_str:
        return None
        
    target_ts_float = float(target_ts_str)
    idx = bisect.bisect_left(available_ts_sorted_float, target_ts_float)
    
    if idx == 0:
        return available_ts_sorted_str[0]
    if idx == len(available_ts_sorted_float):
        return available_ts_sorted_str[-1]
    
    ts_before_float = available_ts_sorted_float[idx - 1]
    ts_after_float = available_ts_sorted_float[idx]
    
    diff_before = target_ts_float - ts_before_float
    diff_after = ts_after_float - target_ts_float
    
    if diff_before < diff_after:
        return available_ts_sorted_str[idx - 1]
    else:
        return available_ts_sorted_str[idx]


# --- 3. HELPER FUNCTION TO SAVE A PLY FILE (Unchanged) ---

def save_ply(path, points_3d, colors_rgb=None):
    """
    Saves a 3D point cloud to a .PLY file.
    Assumes points_3d and colors_rgb are already filtered.
    """
    num_points = len(points_3d)
    if num_points == 0:
        print("Warning: No valid 3D points to save.")
        return

    header = [
        "ply", "format ascii 1.0", f"element vertex {num_points}",
        "property float x", "property float y", "property float z",
    ]
    
    if colors_rgb is not None:
        header.extend([
            "property uchar red", "property uchar green", "property uchar blue",
        ])
    
    header.append("end_header")
    
    with open(path, 'w') as f:
        f.write("\n".join(header) + "\n")
        
        for i in range(num_points):
            line = f"{points_3d[i][0]} {points_3d[i][1]} {points_3d[i][2]}"
            if colors_rgb is not None:
                line += f" {int(colors_rgb[i][0])} {int(colors_rgb[i][1])} {int(colors_rgb[i][2])}"
            f.write(line + "\n")
            
    print(f"Successfully saved {num_points} points to {path}")


# --- 4. MAIN RECONSTRUCTION SCRIPT (MODIFIED WITH LOOP) ---

def main():
    print("Reading associations file...")
    try:
        rgb_map, assoc_ts_sorted, assoc_ts_float = read_associations(ASSOCIATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: Associations file not found at {ASSOCIATIONS_FILE}")
        return
        
    print("Reading keyframe poses...")
    poses_dict = read_poses(KEYFRAME_FILE)
    
    if len(poses_dict) < KEYFRAME_STEP_SIZE or not rgb_map:
        print(f"Error: Not enough poses ({len(poses_dict)}) or associations found.")
        print(f"Need at least {KEYFRAME_STEP_SIZE} poses to create one pair.")
        return

    pose_timestamps_sorted = sorted(poses_dict.keys())
    
    # --- (NEW) Create lists to hold all point clouds ---
    all_points_world = []
    all_colors_rgb = []
    
    num_pairs = len(pose_timestamps_sorted) // KEYFRAME_STEP_SIZE
    print(f"Found {len(pose_timestamps_sorted)} keyframes. Will process {num_pairs} pairs...")
    
    # --- (NEW) Main loop over all keyframe pairs ---
    for i in range(num_pairs):
        
        idx1 = i * KEYFRAME_STEP_SIZE
        idx2 = idx1 + KEYFRAME_STEP_SIZE
        
        # Ensure idx2 is not out of bounds (for the last partial batch)
        if idx2 >= len(pose_timestamps_sorted):
            idx2 = len(pose_timestamps_sorted) - 1
            # If the last batch is too small, skip it
            if idx2 <= idx1:
                continue

        start_time = time.time()
        print(f"\n--- Processing Pair {i+1}/{num_pairs} (Frames {idx1} and {idx2}) ---")

        try:
            ts_pose1 = pose_timestamps_sorted[idx1]
            ts_pose2 = pose_timestamps_sorted[idx2]
            
            pose1 = poses_dict[ts_pose1] 
            pose2 = poses_dict[ts_pose2]

            # --- Find closest matching timestamps for images ---
            ts_assoc1 = find_closest_timestamp(ts_pose1, assoc_ts_sorted, assoc_ts_float)
            ts_assoc2 = find_closest_timestamp(ts_pose2, assoc_ts_sorted, assoc_ts_float)
            
            # --- Load corresponding images ---
            img_path1 = os.path.join(DATASET_DIR, rgb_map[ts_assoc1])
            img_path2 = os.path.join(DATASET_DIR, rgb_map[ts_assoc2])

            img1_bgr = cv2.imread(img_path1)
            img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
            img2_bgr = cv2.imread(img_path2)
            img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

            if img1_bgr is None or img2_bgr is None:
                print(f"Warning: Could not read images for pair {i+1}. Skipping.")
                continue

            img_size = img1_gray.shape[::-1]
            
            # --- 1. Calculate Relative Pose & Rectify ---
            pose1_inv = np.linalg.inv(pose1)
            pose_rel = pose1_inv @ pose2
            R_rel = pose_rel[0:3, 0:3]
            t_rel = pose_rel[0:3, 3]

            (R1, R2, P1, P2, Q, _, _) = cv2.stereoRectify(
                K, DIST_COEFFS, K, DIST_COEFFS, img_size, R_rel, t_rel, 
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
            )

            # --- 2. Warp the Images ---
            map1_left, map2_left = cv2.initUndistortRectifyMap(K, DIST_COEFFS, R1, P1, img_size, cv2.CV_32FC1)
            map1_right, map2_right = cv2.initUndistortRectifyMap(K, DIST_COEFFS, R2, P2, img_size, cv2.CV_32FC1)

            img1_rect = cv2.remap(img1_gray, map1_left, map2_left, cv2.INTER_LINEAR)
            img2_rect = cv2.remap(img2_gray, map1_right, map2_right, cv2.INTER_LINEAR)
            img1_color_rect = cv2.remap(img1_bgr, map1_left, map2_left, cv2.INTER_LINEAR)

            # --- 3. Calculate Disparity Map ---
            window_size = 5
            min_disp = 0
            num_disp = 16 * 10
            
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                P1=8 * 3 * window_size**2, P2=32 * 3 * window_size**2,
                disp12MaxDiff=1, uniquenessRatio=10,
                speckleWindowSize=100, speckleRange=32,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY
            )
            
            disparity_map = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0

            # --- 4. Get 3D Point Cloud ---
            points_3d = cv2.reprojectImageTo3D(disparity_map, Q)

            # --- 5. Transform points to World Frame & Filter ---
            H, W = points_3d.shape[:2]
            colors_flat = cv2.cvtColor(img1_color_rect, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            points_3d_rect = points_3d.reshape(-1, 3)
            
            valid_mask = np.all(np.isfinite(points_3d_rect), axis=1)
            
            points_3d_rect_valid = points_3d_rect[valid_mask]
            colors_flat_valid = colors_flat[valid_mask]
            
            if points_3d_rect_valid.shape[0] == 0:
                print(f"Warning: No valid points found for pair {i+1}. Disparity map was bad. Skipping.")
                continue

            points_h_valid = np.hstack((points_3d_rect_valid, np.ones((points_3d_rect_valid.shape[0], 1))))
            
            R1_inv = np.linalg.inv(R1)
            T_cam1_rect = np.eye(4)
            T_cam1_rect[0:3, 0:3] = R1_inv
            T_world_rect = pose1 @ T_cam1_rect
            
            points_world_valid = (T_world_rect @ points_h_valid.T).T[:, :3]
            
            # --- (NEW) Add this pair's points to the master lists ---
            all_points_world.append(points_world_valid)
            all_colors_rgb.append(colors_flat_valid)
            
            end_time = time.time()
            print(f"Pair {i+1} done. Found {len(points_world_valid)} points. (Took {end_time - start_time:.2f}s)")
        
        except Exception as e:
            print(f"!!! Error processing pair {i+1}: {e}. Skipping this pair. !!!")
            continue

    # --- (NEW) Final step: Combine all points and save ---
    print("\n--- All pairs processed. Combining all point clouds... ---")
    
    if not all_points_world:
        print("Error: No point clouds were generated. Check SGBM parameters or keyframe step size.")
        return

    final_points = np.concatenate(all_points_world, axis=0)
    final_colors = np.concatenate(all_colors_rgb, axis=0)
    
    print(f"Total points in final cloud: {len(final_points)}")
    save_ply(OUTPUT_PLY, final_points, final_colors)


if __name__ == "__main__":
    main()