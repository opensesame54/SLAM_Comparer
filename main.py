import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
import time
import os
import sys

from slam_py.camera import Camera
from slam_py.slam_system import SLAMSystem
from slam_py.utils import load_tum_dataset, draw_trajectory

def print_progress_bar(iteration, total, prefix='', suffix='', length=50):

    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '>' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

def main():
    parser = argparse.ArgumentParser(description='RGB-D SLAM with SIFT')
    parser.add_argument('--dataset', type=str, 
                       default='rgbd_dataset_freiburg1_room',
                       help='Path to TUM RGB-D dataset')
    parser.add_argument('--config', type=str, 
                       default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, 
                       default='output/point_clouds/reconstruction.ply',
                       help='Output PLY file')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process (default: all frames)')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Frame to start from')
    parser.add_argument('--dense', action='store_true',
                       help='Generate dense point cloud')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Process every Nth frame (for faster processing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed frame-by-frame info')
    parser.add_argument('--postprocess', action='store_true',
                       help='Run post-processing after SLAM')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize final result')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'camera': {
                'fx': 525.0, 'fy': 525.0,
                'cx': 319.5, 'cy': 239.5,
                'width': 640, 'height': 480,
                'depth_scale': 5000.0
            },
            'sift': {
                'n_features': 2000,
                'contrast_threshold': 0.04,
                'edge_threshold': 10,
                'sigma': 1.6
            },
            'matcher': {
                'ratio_threshold': 0.75,
                'method': 'FLANN'
            },
            'min_matches': 20,
            'keyframe_interval': 5,
            'max_tracking_lost': 10,
            'use_dense_cloud': False,
            'dense_cloud_step': 5,
            'use_loop_closure': True,
            'loop_min_matches': 50
        }
    
    # Override config if dense flag set
    if args.dense:
        config['use_dense_cloud'] = True
        config['dense_cloud_step'] = 3
        config['keyframe_interval'] = 10
        print("\n🔷 Dense reconstruction mode enabled")
    
    # Initialize camera and SLAM
    camera = Camera.from_config(config['camera'])
    slam = SLAMSystem(camera, config)
    
    # Load dataset
    print(f"\n Loading dataset: {args.dataset}")
    try:
        rgb_files, depth_files, _ = load_tum_dataset(Path(args.dataset))
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return
    
    total_frames = len(rgb_files)
    print(f" Found {total_frames} RGB-D pairs")
    
    # Apply frame range and skipping
    start_idx = args.start_frame
    end_idx = min(args.max_frames + start_idx if args.max_frames else total_frames, total_frames)
    
    rgb_files = rgb_files[start_idx:end_idx:args.skip_frames]
    depth_files = depth_files[start_idx:end_idx:args.skip_frames]
    
    num_frames = len(rgb_files)
    
    print(f" Processing frames {start_idx} to {end_idx}")
    if args.skip_frames > 1:
        print(f"  Skipping every {args.skip_frames} frames")
    print(f" Total frames to process: {num_frames}\n")
    
    # Process frames
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    for idx, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        # Load images
        rgb = cv2.imread(str(rgb_file))
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        
        if rgb is None or depth is None:
            if args.verbose:
                print(f"Frame {idx:04d}: Failed to load images")
            failed_count += 1
            continue
        
        # Process frame
        success = slam.process_frame(rgb, depth)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # Update progress bar (only if not verbose)
        if not args.verbose:
            elapsed = time.time() - start_time
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (num_frames - idx - 1) / fps if fps > 0 else 0
            
            reconstruction = slam.get_reconstruction()
            point_count = reconstruction.get_point_count()
            
            suffix = f"Success: {success_count}, Failed: {failed_count}, Points: {point_count:,}, FPS: {fps:.1f}, ETA: {int(eta)}s"
            print_progress_bar(idx + 1, num_frames, prefix='Processing:', suffix=suffix, length=40)
    
    elapsed = time.time() - start_time
    
    # Print statistics
    print("\n")
    slam.print_statistics()
    
    # Create output directory
    output_path = Path(args.output).resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    
    print(f" Saving results to {output_path.parent}...")
    
    # Save reconstruction
    reconstruction_saved = False
    if slam.save_reconstruction(str(output_path)):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f" Point cloud: {output_path.name} ({file_size:.2f} MB)")
        reconstruction_saved = True
    
    # Save trajectory
    traj_path = str(output_path.parent / 'trajectory.txt')
    if slam.save_trajectory(traj_path):
        print(f" Trajectory: trajectory.txt")
    
    # Draw trajectory visualization
    reconstruction = slam.get_reconstruction()
    if len(reconstruction.keyframe_poses) > 1:
        traj_img_path = str(output_path.parent / 'trajectory.png')
        draw_trajectory(reconstruction.keyframe_poses, output_path=traj_img_path)
        print(f" Trajectory plot: trajectory.png")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"{' SLAM Complete!':^70}")
    print(f"{'='*70}")
    print(f"  Processed:      {success_count}/{num_frames} frames ({success_count/num_frames*100:.1f}%)")
    print(f"  Failed:         {failed_count} frames")
    print(f"  Time:           {elapsed:.2f}s ({success_count/elapsed:.2f} fps)")
    print(f"  Keyframes:      {len(reconstruction.keyframe_ids)}")
    print(f"  Total points:   {reconstruction.get_point_count():,}")
    print(f"\n Output: {output_path.parent}")
    print(f"{'='*70}\n")
    
    # Post-processing
    if args.postprocess and reconstruction_saved:
        print("\n Running post-processing...")
        from slam_py.postprocessor import PointCloudPostProcessor
        
        processor = PointCloudPostProcessor()
        processed_path = str(output_path).replace('.ply', '_processed.ply')
        
        processor.full_pipeline(
            input_ply=str(output_path),
            output_ply=processed_path,
            remove_outliers=True,
            fill_floor=True,
            create_mesh=True,
            mesh_depth=9,
            sample_points=500000
        )
        
        print(f" Processed point cloud saved to: {processed_path}")
        
        if args.visualize:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(processed_path)
            o3d.visualization.draw_geometries([pcd])
    

if __name__ == '__main__':
    main()