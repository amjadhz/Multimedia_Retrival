import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

            # Bounding box: bbox_min: [x_min, y_min, z_min] - Lower left corner; bbox_max: [x_max, y_max, z_max] - Upper right corner
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

    def calculate_statistics(self):
        """Calculate the average of vertices and faces"""
        if not self.data:
            print("No data to calculate statistics")
            return None, None

        df = pd.DataFrame(self.data)

        avg_vertices = df['vertices'].mean()
        avg_faces = df['faces'].mean()

        print(f"\n=== statistics ===")
        print(f"average of vertices: {avg_vertices:.0f}")
        print(f"average of faces: {avg_faces:.0f}")

        return avg_vertices, avg_faces

    def create_histograms(self, avg_vertices, avg_faces):
        """Generate histograms of the number of vertices and faces"""
        if not self.data:
            print("No data to create histograms")
            return

        df = pd.DataFrame(self.data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Vertex count histogram
        vertices_data = df['vertices']

        # Horizontal axis range: 0 to 2 times the average value
        vertices_range = (0, avg_vertices * 2)

        ax1.hist(vertices_data, bins=50, range=vertices_range, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Vertices')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Vertices Distribution Histogram\n(Range: 0 - {avg_vertices*2:.0f})')
        ax1.axvline(avg_vertices, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_vertices:.0f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Face count histogram
        faces_data = df['faces']

        faces_range = (0, avg_faces * 2)

        ax2.hist(faces_data, bins=50, range=faces_range, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Number of Faces')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Faces Distribution Histogram\n(Range: 0 - {avg_faces*2:.0f})')
        ax2.axvline(avg_faces, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_faces:.0f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # save histograms
        histogram_path = os.path.join(self.output_dir, "histograms.png")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Histograms saved to: {histogram_path}")

        plt.show()

    def detect_outliers(self):
        """Detect outliers: low sampling rate and very high sampling rate shapes"""
        if not self.data:
            print("No data to detect outliers")
            return []

        df = pd.DataFrame(self.data)
        outliers = []

        # Low sampling rate outliers (vertices or faces < 100)
        low_vertices = df[df['vertices'] < 100]
        low_faces = df[df['faces'] < 100]

        # High sampling rate outliers (vertices or faces >= 100,000)
        high_vertices = df[df['vertices'] >= 100000]
        high_faces = df[df['faces'] >= 100000]

        # Collect low sampling outliers
        for _, row in low_vertices.iterrows():
            outliers.append({
                'filename': row['filename'],
                'category': row['category'],
                'vertices': row['vertices'],
                'faces': row['faces'],
                'outlier_type': 'low_vertices'
            })

        for _, row in low_faces.iterrows():
            if row['vertices'] >= 100:  # Avoid duplicates
                outliers.append({
                    'filename': row['filename'],
                    'category': row['category'],
                    'vertices': row['vertices'],
                    'faces': row['faces'],
                    'outlier_type': 'low_faces'
                })

        # Collect high sampling outliers
        for _, row in high_vertices.iterrows():
            outliers.append({
                'filename': row['filename'],
                'category': row['category'],
                'vertices': row['vertices'],
                'faces': row['faces'],
                'outlier_type': 'high_vertices'
            })

        for _, row in high_faces.iterrows():
            if row['vertices'] < 100000:  # Avoid duplicates
                outliers.append({
                    'filename': row['filename'],
                    'category': row['category'],
                    'vertices': row['vertices'],
                    'faces': row['faces'],
                    'outlier_type': 'high_faces'
                })

        print(f"\n=== Outlier Detection ===")
        print(f"Low vertices outliers (<100): {len(low_vertices)}")
        print(f"Low faces outliers (<100): {len(low_faces)}")
        print(f"High vertices outliers (>=100,000): {len(high_vertices)}")
        print(f"High faces outliers (>=100,000): {len(high_faces)}")
        print(f"Total unique outliers detected: {len(outliers)}")

        return outliers

    def save_outliers(self, outliers):
        """Save outliers to a separate CSV file"""
        if not outliers:
            print("No outliers to save")
            return

        outliers_df = pd.DataFrame(outliers)
        outliers_path = os.path.join(self.output_dir, "outliers.csv")
        outliers_df.to_csv(outliers_path, index=False)
        print(f"Outliers saved to: {outliers_path}")

    def run_analysis_with_statistics(self, data_path):
        """Run a full analysis including statistics, histograms generation and outlier detection"""
        print("start to analyze")

        # Step 2.1: Analyzing shapes
        success_count = self.analyze_database(data_path)

        if success_count > 0:
            self.save_results()

            # Step 2.2: Statistics
            print("\nstart statistics analysis")
            avg_vertices, avg_faces = self.calculate_statistics()

            if avg_vertices is not None and avg_faces is not None:
                # Generate histograms
                print("generate histograms")
                self.create_histograms(avg_vertices, avg_faces)

                # Detect and save outliers
                print("\nDetecting outliers...")
                outliers = self.detect_outliers()
                if outliers:
                    self.save_outliers(outliers)

                print(f"\nAnalysis completed! {success_count} mesh models analyzed")
            else:
                print("No mesh models successfully analyzed")

def main():
    parser = argparse.ArgumentParser(description="Mesh Analysis Tool - Step 2.1")
    parser.add_argument("--data", default="../data", help="Path to mesh database")
    parser.add_argument("--output", default="analysis", help="Output directory")

    args = parser.parse_args()

    analyzer = MeshAnalyzer(args.output)
    analyzer.run_analysis_with_statistics(args.data)


if __name__ == "__main__":
    main()