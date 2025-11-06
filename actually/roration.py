import open3d as o3d
import numpy as np
import os
import sys

# ----------------- CONFIGURATION -----------------
DATASET_BASE_PATH = "rgbd_dataset_freiburg1_desk2" 
ASSOCIATIONS_FILE = "associations.txt"
OUTPUT_TRAJECTORY_FILE = "dense_slam_trajectory.txt" 

# ---------- Camera intrinsics (TUM Freiburg1 dataset) ----------
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640,  
    height=480, 
    fx=517.3, 
    fy=516.5, 
    cx=318.6, 
    cy=255.3
)

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def load_associations(base_path, file_path):
    """Loads associations and prepends the base path to the filenames."""
    full_path = os.path.join(base_path, file_path)
    pairs = []
    try:
        with open(full_path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                parts = line.strip().split()
                if len(parts) < 4: continue
                rgb_file = os.path.join(base_path, parts[1])
                depth_file = os.path.join(base_path, parts[3])
                pairs.append((rgb_file, depth_file))
    except FileNotFoundError:
        print(f"Error: Associations file not found at {full_path}.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(pairs)} image pairs.")
    return pairs

def save_trajectory(file_path, poses):
    """Saves the list of 4x4 poses to a text file."""
    with open(file_path, "w") as f:
        for idx, pose in enumerate(poses):
            # Write a comment line for readability
            f.write(f"# Frame {idx}\n")
            # Flatten the 4x4 matrix into a single line of 16 numbers
            f.write(" ".join(map(str, pose.flatten())) + "\n")
    print(f"\n[SUCCESS] Trajectory with {len(poses)} poses saved to '{file_path}'")

# -----------------------------------------------------
# Main Odometry Calculation
# -----------------------------------------------------

def run_odometry(pairs):
    """
    Processes image pairs to calculate camera trajectory using RGB-D odometry.
    """
    # Initialize the global pose as the origin (identity matrix)
    global_pose = np.identity(4)
    # The list of all camera poses throughout the sequence
    poses = [global_pose]

    # We need the first frame to start the process
    source_rgbd_image = None

    print(f"[INFO] Starting RGB-D odometry calculation for {len(pairs)} frames...")

    for i in range(len(pairs)):
        rgb_path, depth_path = pairs[i]

        # Load the current frame's images
        try:
            color_image = o3d.io.read_image(rgb_path)
            depth_image = o3d.io.read_image(depth_path)
        except Exception as e:
            print(f"Warning: Could not read frame {i}, skipping. Error: {e}")
            continue

        # Create an RGBDImage object
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
            color_image, depth_image, convert_rgb_to_intensity=False
        )

        # The first frame is our starting point
        if i == 0:
            source_rgbd_image = target_rgbd_image
            continue

        # --- Core Odometry Step ---
        # Calculate the transformation from the SOURCE (previous frame) to the TARGET (current frame)
        option = o3d.pipelines.odometry.OdometryOption()
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image,
            target_rgbd_image,
            o3d_intrinsics,
            np.identity(4),  # Initial guess for the transformation
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )

        if success:
            # The 'trans' matrix represents the motion from the previous frame to the current one.
            # We update the global pose by chaining this transformation.
            global_pose = np.dot(global_pose, trans)
            poses.append(global_pose)
        else:
            # If tracking fails, we assume the camera didn't move.
            # A more advanced system might try to re-localize.
            print(f"Warning: Odometry failed at frame {i}. Reusing last known pose.")
            poses.append(poses[-1])

        # The current frame becomes the source for the next iteration
        source_rgbd_image = target_rgbd_image

        if i > 0 and i % 50 == 0:
            print(f"Processed {i} frames...")

    return poses


if __name__ == '__main__':
    # Load the synchronized image file pairs
    image_pairs = load_associations(DATASET_BASE_PATH, ASSOCIATIONS_FILE)
    
    # Calculate the full trajectory
    camera_poses = run_odometry(image_pairs)
    
    # Save the trajectory to the output file
    save_trajectory(OUTPUT_TRAJECTORY_FILE, camera_poses)
