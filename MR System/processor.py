import open3d as o3d
import numpy as np
import pandas as pd
import os
import glob
import argparse


class MeshAnalyzer:
    def __init__(self, output_dir="analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data = []

    def analyze_mesh(self, file_path):
        """Analyze single mesh file"""
        try:
            category = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)

            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) == 0:
                return None

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Basic counts
            num_vertices = len(vertices)
            num_faces = len(triangles)

            # Face types
            face_types = self.classify_face_types(file_path)

            # Bounding box: bbox_min: [x_min, y_min, z_min] - Lower left corner； bbox_max: [x_max, y_max, z_max] - Upper right corner
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_min = bbox.min_bound
            bbox_max = bbox.max_bound
            bbox_size = bbox_max - bbox_min

            result = {
                'filename': filename,
                'category': category,
                'vertices': num_vertices,
                'faces': num_faces,
                'face_type': face_types,
                'bbox_width': bbox_size[0],
                'bbox_height': bbox_size[1],
                'bbox_depth': bbox_size[2],
                'bbox_volume': np.prod(bbox_size)
            }

            print(f"{filename}: {category}, V:{num_vertices}, F:{num_faces}, "
                  f"Box:[{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")

            return result

        except Exception as e:
            print(f"Failed: {file_path} - {e}")
            return None

    def classify_face_types(self, file_path):
        """Classify mesh as: triangles_only, quads_only, or mixed"""
        triangle_count = 0
        quad_count = 0
        other_count = 0

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('f '):
                        vertex_count = len(line.split()) - 1
                        if vertex_count == 3:
                            triangle_count += 1
                        elif vertex_count == 4:
                            quad_count += 1
                        else:
                            other_count += 1
        except Exception:
            return "triangles_only"  # fallback for Open3D default

        # Classify based on counts
        has_triangles = triangle_count > 0
        has_quads = quad_count > 0
        has_other = other_count > 0

        if has_triangles and not has_quads and not has_other:
            return "triangles_only"
        elif has_quads and not has_triangles and not has_other:
            return "quads_only"
        else:
            return "mixed"  # combination of different face types

    def analyze_database(self, data_path):
        """Analyze all meshes in database"""
        obj_files = glob.glob(os.path.join(data_path, "**/*.obj"), recursive=True)

        for file_path in obj_files:
            result = self.analyze_mesh(file_path)
            if result:
                self.data.append(result)

        print(f"Successfully analyzed: {len(self.data)}/{len(obj_files)}")
        return len(self.data)

    def save_results(self):
        """Save analysis results to CSV"""
        if not self.data:
            print("No data to save")
            return

        df = pd.DataFrame(self.data)
        csv_path = os.path.join(self.output_dir, "analysis.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Mesh Analysis Tool - Step 2.1")
    parser.add_argument("--data", default="../data", help="Path to mesh database")
    parser.add_argument("--output", default="analysis", help="Output directory")

    args = parser.parse_args()

    analyzer = MeshAnalyzer(args.output)
    success_count = analyzer.analyze_database(args.data)

    if success_count > 0:
        analyzer.save_results()
        print(f"\nAnalysis complete! {success_count} meshes analyzed")
    else:
        print("No meshes analyzed")


if __name__ == "__main__":
    main()