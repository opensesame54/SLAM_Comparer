import cv2
import numpy as np
import os
import open3d as o3d
import sys
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import scipy.optimize
import concurrent.futures
from tqdm import tqdm
import time

# ----------------- CONFIGURATION -----------------
DATASET_BASE_PATH = "actually/rgbd_dataset_freiburg1_room"
ASSOCIATIONS_FILE = "associations.txt"
TRAJECTORY_FILE = "CameraTrajectory.txt"
OUTPUT_PLY_FILE = "room_point_cloud_with_loop_closure.ply"

# Process every Nth frame
MAX_FEATURES_PER_FRAME = 500  # Limit SIFT features per frame
FRAME_STRIDE = 5  # Increased from 3 to process fewer frames

# Loop closure parameters
MIN_LOOP_CLOSURE_DISTANCE = 0.5  # meters
BOW_VOCABULARY_SIZE = 500  # Reduced from 1000
FEATURE_DETECTOR = cv2.SIFT_create(nfeatures=MAX_FEATURES_PER_FRAME)

# Camera intrinsics (TUM Freiburg1 dataset)
o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640,  
    height=480, 
    fx=517.3, 
    fy=516.5, 
    cx=318.6, 
    cy=255.3
)

class LoopClosureDetector:
    def __init__(self, vocabulary_size=BOW_VOCABULARY_SIZE):
        self.vocabulary_size = vocabulary_size
        self.bow_trainer = cv2.BOWKMeansTrainer(vocabulary_size)
        self.descriptor_matcher = cv2.BFMatcher()
        self.vocabulary = None
        self.bow_extractor = None
        self.frame_descriptors = []
        self.frame_poses = []
        self.frame_timestamps = []

    def add_frame(self, rgb_image, pose, timestamp):
        keypoints, descriptors = FEATURE_DETECTOR.detectAndCompute(rgb_image, None)
        
        if descriptors is not None and len(descriptors) > 0:
            # Limit number of descriptors per frame
            if len(descriptors) > MAX_FEATURES_PER_FRAME:
                indices = np.random.choice(len(descriptors), MAX_FEATURES_PER_FRAME, replace=False)
                descriptors = descriptors[indices]
            
            self.bow_trainer.add(descriptors.astype(np.float32))
            self.frame_descriptors.append(descriptors)
            self.frame_poses.append(pose)
            self.frame_timestamps.append(timestamp)

    def train_vocabulary(self):
        print("[INFO] Training BoW vocabulary...")
        start_time = time.time()
        
        if len(self.bow_trainer.getDescriptors()) > 0:
            total_descriptors = sum(len(desc) for desc in self.frame_descriptors if desc is not None)
            print(f"[INFO] Training with {total_descriptors} descriptors across {len(self.frame_descriptors)} frames")
            
            # Reduce vocabulary size if we have few descriptors
            if total_descriptors < self.vocabulary_size * 10:
                adjusted_size = max(100, total_descriptors // 10)
                print(f"[WARNING] Reducing vocabulary size to {adjusted_size} due to limited descriptors")
                self.vocabulary_size = adjusted_size
                self.bow_trainer = cv2.BOWKMeansTrainer(self.vocabulary_size)
                for desc in self.frame_descriptors:
                    if desc is not None and len(desc) > 0:
                        self.bow_trainer.add(desc.astype(np.float32))
            
            print("[INFO] Clustering descriptors (this may take a few minutes)...")
            self.vocabulary = self.bow_trainer.cluster()
            self.bow_extractor = cv2.BOWImgDescriptorExtractor(FEATURE_DETECTOR, 
                                                             self.descriptor_matcher)
            self.bow_extractor.setVocabulary(self.vocabulary)
            
            duration = time.time() - start_time
            print(f"[INFO] Vocabulary training completed in {duration:.1f} seconds")
        else:
            print("[WARNING] No descriptors available for vocabulary training")

    def detect_loop_closures(self):
        if not self.vocabulary is not None:
            print("[WARNING] Vocabulary not trained, no loop closures detected")
            return []

        print("[INFO] Detecting loop closures...")
        loop_closures = []
        
        # Convert all frames to BoW vectors
        bow_vectors = []
        for descriptors in tqdm(self.frame_descriptors, desc="Computing BoW vectors"):
            if descriptors is not None and len(descriptors) > 0:
                # Create dummy keypoints for the descriptors
                keypoints = [cv2.KeyPoint(0, 0, 1) for _ in range(len(descriptors))]
                
                # Create a dummy image (required by compute method)
                dummy_image = np.zeros((1, 1), dtype=np.uint8)
                
                bow_vector = self.bow_extractor.compute(
                    dummy_image, 
                    keypoints, 
                    descriptors.astype(np.float32)
                )
                if bow_vector is not None:
                    bow_vectors.append(bow_vector.reshape(-1))

        if not bow_vectors:
            print("[WARNING] No BoW vectors computed")
            return []

        print(f"[INFO] Created {len(bow_vectors)} BoW vectors")
        bow_vectors = np.array(bow_vectors)
        bow_vectors = normalize(bow_vectors)

        # Find potential loop closures
        nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(bow_vectors)
        distances, indices = nbrs.kneighbors(bow_vectors)

        for i in range(len(bow_vectors)):
            for j, dist in zip(indices[i], distances[i]):
                if j <= i + 10:  # Skip recent frames
                    continue
                
                pose_diff = np.linalg.norm(
                    self.frame_poses[i][:3, 3] - self.frame_poses[j][:3, 3]
                )
                
                if pose_diff < MIN_LOOP_CLOSURE_DISTANCE and dist < 0.3:
                    loop_closures.append((i, j))

        return loop_closures

def optimize_pose_graph(poses_dict, loop_closures):
    print("[INFO] Optimizing pose graph...")
    
    timestamps = sorted(poses_dict.keys())
    n_poses = len(timestamps)
    
    # Create timestamp to index mapping
    timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}
    
    # Initialize optimization vector (translation + quaternion for each pose)
    initial_params = np.zeros(n_poses * 7)
    
    for i, timestamp in enumerate(timestamps):
        pose = poses_dict[timestamp]
        initial_params[i*7:i*7+3] = pose[:3, 3]
        R = pose[:3, :3]
        
        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2,1] - R[1,2]) / s
            qy = (R[0,2] - R[2,0]) / s
            qz = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s
                
        initial_params[i*7+3:i*7+7] = [qw, qx, qy, qz]
    
    def pose_graph_error(params):
        error = []
        
        # Sequential constraints only (removed regularization)
        for i in range(n_poses - 1):
            p1 = params[i*7:i*7+7]
            p2 = params[(i+1)*7:(i+1)*7+7]
            error.extend(relative_pose_error(p1, p2))
        
        # Loop closure constraints (reduced weight)
        for idx1, idx2 in loop_closures:
            t1, t2 = timestamps[idx1], timestamps[idx2]
            i1, i2 = timestamp_to_idx[t1], timestamp_to_idx[t2]
            p1 = params[i1*7:i1*7+7]
            p2 = params[i2*7:i2*7+7]
            error.extend(relative_pose_error(p1, p2))
        
        return np.array(error).flatten()
    
    def relative_pose_error(p1, p2):
        # Extract translation and rotation
        t1, q1 = p1[:3], p1[3:]
        t2, q2 = p2[:3], p2[3:]
        
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute relative translation error
        t_error = np.linalg.norm(t2 - t1)
        
        # Compute relative rotation error (quaternion distance)
        q_error = 1 - np.abs(np.dot(q1, q2))
        
        return [t_error, q_error]
    
    # Optimize using 'trf' method with better parameters
    print("[INFO] Running pose graph optimization...")
    result = scipy.optimize.least_squares(
        pose_graph_error,
        initial_params,
        method='trf',
        loss='soft_l1',      # Changed from huber
        f_scale=1.0,
        ftol=1e-3,          # Made more lenient
        xtol=1e-3,          # Made more lenient
        gtol=1e-3,          # Made more lenient
        max_nfev=20,        # Reduced from 50 to 20 iterations
        verbose=2,
        x_scale='jac'
    )
    
    # Add status check
    if not result.success:
        print(f"\n[WARNING] Optimization status: {result.status}")
        print(f"Message: {result.message}")
    else:
        print("\n[SUCCESS] Optimization converged successfully")
    
    # Convert optimized parameters back to pose matrices
    optimized_poses = {}
    for i, timestamp in enumerate(timestamps):
        params = result.x[i*7:i*7+7]
        t = params[:3]
        q = params[3:] / np.linalg.norm(params[3:])  # normalized quaternion
        
        # Convert quaternion to rotation matrix
        w, x, y, z = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        optimized_poses[timestamp] = pose
    
    print("[INFO] Pose graph optimization completed")
    return optimized_poses

def load_associations(base_path, file_path):
    """Loads associations (timestamp, rgb_path, depth_path) and sorts them."""
    full_path = os.path.join(base_path, file_path)
    pairs = []
    try:
        with open(full_path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                parts = line.strip().split()
                if len(parts) == 4:
                    timestamp = float(parts[0])
                    rgb_file = os.path.join(base_path, parts[1])
                    depth_file = os.path.join(base_path, parts[3])
                    pairs.append((timestamp, rgb_file, depth_file))
                elif len(parts) == 8:
                    timestamp = float(parts[0])
                    rgb_file = os.path.join(base_path, parts[1])
                    depth_file = os.path.join(base_path, parts[4])
                    pairs.append((timestamp, rgb_file, depth_file))
                
    except FileNotFoundError:
        print(f"Error: Associations file not found at {full_path}.")
        sys.exit(1)
    
    pairs.sort()
    print(f"[INFO] Loaded {len(pairs)} image pairs from {full_path}")
    return pairs

def load_poses_from_file(file_path):
    poses = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                
                parts = line.strip().split()
                
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    translation = np.array([float(p) for p in parts[1:4]])
                    quaternion = np.array([
                        float(parts[7]),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6])
                    ])
                    
                    pose_matrix = np.eye(4)
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(
                        quaternion)
                    pose_matrix[:3, :3] = rotation_matrix
                    pose_matrix[:3, 3] = translation
                    
                    poses[timestamp] = pose_matrix
                    
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at '{file_path}'.")
        sys.exit(1)
        
    if not poses:
        print(f"[ERROR] Loaded 0 poses from {file_path}")
    else:
        print(f"[INFO] Loaded {len(poses)} poses from {file_path}")
        
    return poses

def generate_and_visualize_point_cloud(pairs, poses_dict):
    # Initialize loop closure detector
    loop_detector = LoopClosureDetector()
    
    # First pass: collect features for loop closure detection
    print("[INFO] First pass: collecting features for loop closure detection...")
    for idx in tqdm(range(0, len(pairs), FRAME_STRIDE), desc="Collecting features"):
        timestamp, rgb_path, _ = pairs[idx]
        if timestamp in poses_dict:
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is not None:
                loop_detector.add_frame(rgb_image, poses_dict[timestamp], timestamp)

    # Train vocabulary and detect loop closures
    loop_detector.train_vocabulary()
    loop_closures = loop_detector.detect_loop_closures()
    print(f"[INFO] Detected {len(loop_closures)} loop closures")

    # Optimize pose graph if loop closures found
    if loop_closures:
        poses_dict = optimize_pose_graph(poses_dict, loop_closures)
        print("[INFO] Pose graph optimization completed")

    # Generate point cloud with optimized poses
    final_pcd = o3d.geometry.PointCloud()
    
    print(f"Starting 3D point cloud generation across {len(pairs)} frames...")
    processed_frames = 0

    for idx in range(0, len(pairs), FRAME_STRIDE):
        timestamp, rgb_path, depth_path = pairs[idx]

        try:
            pose_timestamp = min(poses_dict.keys(), 
                               key=lambda k: abs(k - timestamp))
        except ValueError:
            continue

        if abs(pose_timestamp - timestamp) > 0.02:
            continue
            
        T_world_camera = poses_dict[pose_timestamp]

        try:
            rgb_image = o3d.io.read_image(rgb_path)
            depth_image = o3d.io.read_image(depth_path)
        except Exception as e:
            print(f"Warning: Could not read frame {idx}, skipping. Error: {e}")
            continue
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
            rgb_image, depth_image, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            o3d_intrinsics
        )
        
        pcd.transform(T_world_camera)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)
        final_pcd += pcd_downsampled

        processed_frames += 1
        if processed_frames % 10 == 0:
             print(f"Processed {processed_frames} frames...")
        
        del rgb_image, depth_image, rgbd_image, pcd, pcd_downsampled
        gc.collect()

    print(f"\n[INFO] Total points before final cleaning: {len(final_pcd.points)}")

    print("[INFO] Removing statistical outliers...")
    cl, ind = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    final_pcd = final_pcd.select_by_index(ind)
    print(f"[INFO] Final point count after cleaning: {len(final_pcd.points)}")

    o3d.io.write_point_cloud(OUTPUT_PLY_FILE, final_pcd)
    print(f"\n[SUCCESS] Point cloud saved to {OUTPUT_PLY_FILE}")
    
    print("[INFO] Displaying final point cloud. Press 'q' to close.")
    o3d.visualization.draw_geometries([final_pcd], 
                                    window_name="3D Reconstruction with Loop Closure")

if __name__ == '__main__':
    pairs = load_associations(DATASET_BASE_PATH, ASSOCIATIONS_FILE)
    poses_dict = load_poses_from_file(TRAJECTORY_FILE)

    if not pairs:
        print("Error: No image pairs were loaded. Check associations.txt.")
        sys.exit(1)
        
    if not poses_dict:
        print("Error: No poses were loaded. Check CameraTrajectory.txt.")
        sys.exit(1)
        
    generate_and_visualize_point_cloud(pairs, poses_dict)