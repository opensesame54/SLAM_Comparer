import argparse
from pathlib import Path
from slam_py.postprocessor import PointCloudPostProcessor

def main():
    parser = argparse.ArgumentParser(description='Post-process SLAM reconstruction')
    parser.add_argument('--input', type=str,
                       default='output/point_clouds/reconstruction.ply',
                       help='Input PLY file')
    parser.add_argument('--output', type=str,
                       default='output/point_clouds/reconstruction_processed.ply',
                       help='Output PLY file')
    parser.add_argument('--no-outliers', action='store_true',
                       help='Skip outlier removal')
    parser.add_argument('--no-floor', action='store_true',
                       help='Skip floor filling')
    parser.add_argument('--no-mesh', action='store_true',
                       help='Skip mesh reconstruction')
    parser.add_argument('--mesh-depth', type=int, default=8,
                       help='Poisson mesh depth (8-10 recommended)')
    parser.add_argument('--sample-points', type=int, default=500000,
                       help='Number of points to sample from mesh')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize result')
    parser.add_argument('--downsample-first', type=float, default=0.01,
                       help='Downsample voxel size before processing (0 = skip)')
    
    args = parser.parse_args()
    
    # Check input exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Process
    processor = PointCloudPostProcessor()
    
    # For large point clouds, downsample first
    if args.downsample_first > 0:
        print(f"\n  Large point cloud detected - downsample (voxel={args.downsample_first}m)")
        success = processor.full_pipeline_large(
            input_ply=args.input,
            output_ply=args.output,
            downsample_voxel=args.downsample_first,
            remove_outliers=not args.no_outliers,
            fill_floor=not args.no_floor,
            create_mesh=not args.no_mesh,
            mesh_depth=args.mesh_depth,
            sample_points=args.sample_points
        )
    else:
        success = processor.full_pipeline(
            input_ply=args.input,
            output_ply=args.output,
            remove_outliers=not args.no_outliers,
            fill_floor=not args.no_floor,
            create_mesh=not args.no_mesh,
            mesh_depth=args.mesh_depth,
            sample_points=args.sample_points
        )
    
    if success and args.visualize:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(args.output)
        o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()