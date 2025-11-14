import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple OBJ mesh viewer with multiple shading modes."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to the mesh file (e.g., data/Tool/D01165.obj)",
    )
    parser.add_argument(
        "--vis_option",
        type=str,
        default="flatshade",
        choices=["smoothshade", "flatshade", "wireframe", "unshaded"],
        help="Visualization mode.",
    )
    parser.add_argument(
        "--no_edges",
        action="store_true",
        help="Disable drawing the wireframe edges on top of the mesh.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.is_file():
        print(f"[ERROR] Mesh file not found: {mesh_path}")
        sys.exit(1)

    print(f"[INFO] Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if mesh.is_empty():
        print("[ERROR] Loaded mesh is empty.")
        sys.exit(1)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"[INFO] Vertices: {vertices.shape[0]}")
    print(f"[INFO] Faces (triangles): {triangles.shape[0]}")

    vis_option = args.vis_option
    edges_drawn = not args.no_edges

    # Create wireframe LineSet from triangles
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    # Choose shading / visualization mode
    if vis_option == "smoothshade":
        mesh.compute_vertex_normals()
        draw_objects = [mesh, wireframe] if edges_drawn else [mesh]

    elif vis_option == "flatshade":
        mesh.compute_triangle_normals()
        draw_objects = [mesh, wireframe] if edges_drawn else [mesh]

    elif vis_option == "wireframe":
        draw_objects = [wireframe]

    elif vis_option == "unshaded":
        # No normals computed → unshaded
        draw_objects = [mesh, wireframe] if edges_drawn else [mesh]

    else:
        print(f"[ERROR] Unknown vis_option: {vis_option}")
        sys.exit(1)

    print(f"[INFO] Visualization mode: {vis_option}")
    print(f"[INFO] Edges drawn: {edges_drawn}")

    o3d.visualization.draw(
        draw_objects,
        width=1280,
        height=720,
        show_skybox=False,
    )


if __name__ == "__main__":
    main()
