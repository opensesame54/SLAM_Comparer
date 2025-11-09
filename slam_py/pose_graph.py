class PoseGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, pose):
        self.nodes.append(pose)

    def add_edge(self, from_node, to_node, transformation):
        self.edges.append((from_node, to_node, transformation))

    def optimize(self):
        # Implement optimization logic here
        pass

    def get_optimized_poses(self):
        # Return optimized poses after optimization
        return self.nodes