import open3d as o3d

# Define the path to your point cloud file
file_path = "/home/aashrith/Desktop/project_cv/desk_point_cloud.ply"

# 1. Load the point cloud
print(f"Loading the point cloud from: {file_path}")
pcd = o3d.io.read_point_cloud(file_path)

# Check if the point cloud was loaded successfully
if not pcd.has_points():
    print("Error: Could not read the point cloud file.")
else:
    # 2. Visualize the point cloud
    print("Displaying the point cloud. Press 'q' to close the window.")
    o3d.visualization.draw_geometries([pcd])