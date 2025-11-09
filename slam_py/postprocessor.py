import numpy as np
import open3d as o3d

class PointCloudPostProcessor:
    
    def __init__(self):
        pass
    
    def load_ply(self, filename):
        """Load PLY file using Open3D"""
        print(f"Loading {filename}...")
        pcd = o3d.io.read_point_cloud(filename)
        print(f"  Loaded {len(pcd.points)} points")
        return pcd
    
    def remove_statistical_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        print(f"Removing outliers (neighbors={nb_neighbors}, std={std_ratio})...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        removed = len(pcd.points) - len(cl.points)
        print(f"  Removed {removed} outliers ({removed/len(pcd.points)*100:.1f}%)")
        return cl
    
    def remove_radius_outliers(self, pcd, nb_points=16, radius=0.05):
        """Remove points with few neighbors in radius"""
        print(f"Removing radius outliers (radius={radius}m, min_points={nb_points})...")
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        removed = len(pcd.points) - len(cl.points)
        print(f"  Removed {removed} outliers ({removed/len(pcd.points)*100:.1f}%)")
        return cl
    
    def downsample(self, pcd, voxel_size=0.01):
        print(f"Downsampling (voxel_size={voxel_size}m)...")
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  Points: {len(pcd.points)} → {len(downsampled.points)}")
        return downsampled
    
    def fill_floor_plane(self, pcd, distance_threshold=0.02, grid_step=0.05):
        
        print(f"Detecting floor plane (threshold={distance_threshold}m)...")
        
        # Segment plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        [a, b, c, d] = plane_model
        print(f"  Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        print(f"  Inliers: {len(inliers)}/{len(pcd.points)} ({len(inliers)/len(pcd.points)*100:.1f}%)")
        
        # Get floor points
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        
        # Get bounding box of floor
        min_bound = np.asarray(inlier_cloud.get_min_bound())
        max_bound = np.asarray(inlier_cloud.get_max_bound())
        
        print(f"  Filling floor grid (step={grid_step}m)...")
        
        # Generate grid
        x = np.arange(min_bound[0], max_bound[0], grid_step)
        y = np.arange(min_bound[1], max_bound[1], grid_step)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate z using plane equation: ax + by + cz + d = 0 => z = (-d - ax - by) / c
        if abs(c) < 1e-6:
            print("  Warning: Nearly vertical plane detected, skipping fill")
            return pcd
        
        zz = (-d - a * xx - b * yy) / c
        
        # Create filled floor points
        filled_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        filled_floor = o3d.geometry.PointCloud()
        filled_floor.points = o3d.utility.Vector3dVector(filled_points)
        filled_floor.paint_uniform_color([0.3, 0.6, 0.3])  # Green for filled floor
        
        print(f"  Added {len(filled_points)} floor points")
        
        # Combine
        combined = outlier_cloud + filled_floor
        
        return combined
    
    def create_mesh_poisson(self, pcd, depth=9, estimate_normals=True):
        print(f"Creating mesh using Poisson (depth={depth})...")
        
        # Estimate normals if needed
        if estimate_normals or not pcd.has_normals():
            print("  Estimating normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        print(f"  Created mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Remove low-density vertices (outliers)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"  After filtering: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        return mesh
    
    def mesh_to_dense_pointcloud(self, mesh, num_points=1000000):
        
        print(f"Sampling {num_points:,} points from mesh...")
        
        # Sample points from mesh
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=num_points,
            init_factor=5
        )
        
        print(f"  Generated {len(pcd.points)} points")
        
        return pcd
    
    def full_pipeline(self, input_ply, output_ply, 
                     remove_outliers=True,
                     fill_floor=True,
                     create_mesh=True,
                     mesh_depth=9,
                     sample_points=500000):
        
        print(f"\n{'='*70}")
        print(f"{'Point Cloud Post-Processing Pipeline':^70}")
        print(f"{'='*70}\n")
        
        # Load
        pcd = self.load_ply(input_ply)
        
        if len(pcd.points) == 0:
            print("Error: Empty point cloud!")
            return False
        
        # 1. Remove outliers
        if remove_outliers:
            pcd = self.remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
            pcd = self.remove_radius_outliers(pcd, nb_points=10, radius=0.05)
        
        # 2. Fill floor
        if fill_floor:
            pcd = self.fill_floor_plane(pcd, distance_threshold=0.03, grid_step=0.05)
        
        # 3. Create mesh and sample
        if create_mesh:
            mesh = self.create_mesh_poisson(pcd, depth=mesh_depth)
            
            # Save mesh
            mesh_file = output_ply.replace('.ply', '_mesh.ply')
            print(f"Saving mesh to {mesh_file}...")
            o3d.io.write_triangle_mesh(mesh_file, mesh)
            
            # Sample points from mesh
            pcd = self.mesh_to_dense_pointcloud(mesh, num_points=sample_points)
        
        # Save result
        print(f"\nSaving final point cloud to {output_ply}...")
        o3d.io.write_point_cloud(output_ply, pcd)
        
        print(f"\n{'='*70}")
        print(f"{' Post-processing complete!':^70}")
        print(f"{'='*70}\n")
        
        return True
    
    def visualize(self, pcd_or_mesh):
        """Visualize point cloud or mesh"""
        o3d.visualization.draw_geometries([pcd_or_mesh])
    def full_pipeline_large(self, input_ply, output_ply,
                           downsample_voxel=0.01,
                           remove_outliers=True,
                           fill_floor=True,
                           create_mesh=True,
                           mesh_depth=8,
                           sample_points=500000):
        
        print(f"\n{'='*70}")
        print(f"{'Point Cloud Post-Processing (Large Dataset)':^70}")
        print(f"{'='*70}\n")
        
        # Load
        pcd = self.load_ply(input_ply)
        
        if len(pcd.points) == 0:
            print("Error: Empty point cloud!")
            return False
        
        n_original = len(pcd.points)
        
        # 1. Downsample FIRST for large clouds
        if downsample_voxel > 0:
            print(f"\n Step 1/5: Downsampling")
            pcd = self.downsample(pcd, voxel_size=downsample_voxel)
            print(f"  Reduction: {n_original:,} → {len(pcd.points):,} points ({len(pcd.points)/n_original*100:.1f}%)")
        
        # 2. Remove outliers (faster on downsampled cloud)
        if remove_outliers:
            print(f"\n Step 2/5: Removing outliers")
            # Use more lenient parameters for large clouds
            pcd = self.remove_statistical_outliers(pcd, nb_neighbors=10, std_ratio=2.5)
            # Skip radius outlier removal for large clouds (too slow)
            print("  Skipping radius outlier removal ")
        
        # 3. Fill floor
        if fill_floor:
            print(f"\n Step 3/5: Filling floor plane")
            pcd = self.fill_floor_plane(pcd, distance_threshold=0.03, grid_step=0.03)
        
        # 4. Create mesh and sample
        if create_mesh:
            print(f"\n Step 4/5: Creating mesh")
            mesh = self.create_mesh_poisson(pcd, depth=mesh_depth)
            
            # Save mesh
            mesh_file = output_ply.replace('.ply', '_mesh.ply')
            print(f"  Saving mesh to {mesh_file}...")
            o3d.io.write_triangle_mesh(mesh_file, mesh)
            
            print(f"\n Step 5/5: Sampling points from mesh")
            pcd = self.mesh_to_dense_pointcloud(mesh, num_points=sample_points)
        
        # Save result
        print(f"\n Saving final point cloud to {output_ply}...")
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"  Final point count: {len(pcd.points):,}")
        
        print(f"\n{'='*70}")
        print(f"{'✓ Post-processing complete!':^70}")
        print(f"{'='*70}\n")
        
        return True