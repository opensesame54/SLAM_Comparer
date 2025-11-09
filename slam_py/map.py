class Map:
    def __init__(self):
        self.keyframes = []
        self.point_cloud = []

    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)

    def update_map(self, new_points):
        self.point_cloud.extend(new_points)

    def get_keyframes(self):
        return self.keyframes

    def get_point_cloud(self):
        return self.point_cloud

    def save_to_ply(self, filename):
        with open(filename, 'w') as ply_file:
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {len(self.point_cloud)}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("end_header\n")
            for point in self.point_cloud:
                ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")