import open3d as o3d
import numpy as np
import os
import glob


class MeshViewer:
    def __init__(self, category_path, vis_option="flatshade", edgesDrawn=True):

        self.category_path = category_path
        self.vis_option = vis_option
        self.edgesDrawn = edgesDrawn

        self.mesh_files = sorted(glob.glob(os.path.join(category_path, "*.obj")))

        if not self.mesh_files:
            raise ValueError(f"No .obj files found in {category_path}")

        print(f"Found {len(self.mesh_files)} mesh files")
        self.current_index = 0

    def load_mesh(self, mesh_path):
        """Load and prepare mesh"""
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        print(f"\nCurrent file: {os.path.basename(mesh_path)}")
        print(f"Vertices: {vertices.shape[0]}")
        print(f"Triangles: {triangles.shape[0]}")

        return mesh

    def visualize_mesh(self, mesh):
        """Visualize mesh"""
        geometries = []

        # Set shading option
        if self.vis_option == "smoothshade":
            mesh.compute_vertex_normals()
        elif self.vis_option == "flatshade":
            mesh.compute_triangle_normals()

        # Add mesh surface
        if self.vis_option != "wireframe":
            geometries.append(mesh)

        # Add wireframe edges
        if self.edgesDrawn or self.vis_option == "wireframe":
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            geometries.append(wireframe)

        o3d.visualization.draw(geometries, width=1280, height=720, show_skybox=False)

    def view_all(self):
        """View all meshes sequentially"""
        print(f"\nViewing category: {os.path.basename(self.category_path)}")
        print(f"Total files: {len(self.mesh_files)}")
        print("Close the window to load the next mesh\n")

        for i, mesh_path in enumerate(self.mesh_files):
            print(f"{'=' * 60}")
            print(f"Progress: {i + 1}/{len(self.mesh_files)}")

            mesh = self.load_mesh(mesh_path)
            self.visualize_mesh(mesh)

        print(f"\n{'=' * 60}")
        print("All meshes have been viewed!")

    def view_next(self):
        """View next mesh"""
        if self.current_index < len(self.mesh_files):
            mesh_path = self.mesh_files[self.current_index]
            print(f"{'=' * 60}")
            print(f"Progress: {self.current_index + 1}/{len(self.mesh_files)}")

            mesh = self.load_mesh(mesh_path)
            self.visualize_mesh(mesh)

            self.current_index += 1
            return True
        else:
            print("All meshes have been viewed!")
            return False

    def view_previous(self):
        """View previous mesh"""
        if self.current_index > 0:
            self.current_index -= 1
            mesh_path = self.mesh_files[self.current_index]
            print(f"{'=' * 60}")
            print(f"Progress: {self.current_index + 1}/{len(self.mesh_files)}")

            mesh = self.load_mesh(mesh_path)
            self.visualize_mesh(mesh)
            return True
        else:
            print("Already at the first mesh!")
            return False


def main():
    category_path = "step3_1_results/Normalized/Bicycle"
    # category_path = "step3_1_5_results"

    # Visualization options: "smoothshade", "flatshade", "wireframe", "unshaded"
    vis_option = "flatshade"

    # Draw wireframe edges
    edgesDrawn = True

    viewer = MeshViewer(category_path, vis_option, edgesDrawn)

    viewer.view_all()


if __name__ == "__main__":
    main()