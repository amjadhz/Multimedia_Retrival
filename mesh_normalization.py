import open3d as o3d
import numpy as np
from pathlib import Path
import shutil
import os
import argparse


def triangleCenter(v1, v2, v3):
    return np.mean([v1, v2, v3], axis=0)


def sign(n):
    if n < 0:
        return -1
    else:
        return 1


# 1. Translation: Move the center of gravity to the origin
def mesh_translate(mesh):
    barycenter = mesh.get_center()
    mesh.translate(-barycenter)
    return mesh


# 2. Size Normalization: Scale to unit size
def mesh_resize(mesh, target_size=1.0):
    vertices = np.asarray(mesh.vertices)
    max_distance = np.max(np.linalg.norm(vertices, axis=1))
    if max_distance > 0:
        scaling_factor = target_size / max_distance
        mesh.scale(scaling_factor, center=(0, 0, 0))
    return mesh


# 3. Pose Alignment: Align to principal axes
def mesh_pose_alignment(mesh):
    vertices = np.asarray(mesh.vertices)
    covariance = np.cov(vertices.T)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Sort by eigenvalues (descending)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    l1 = eigenvectors[:, 0]
    l2 = eigenvectors[:, 1]
    l3 = np.cross(l1, l2)
    l3 = l3 / np.linalg.norm(l3)

    # Project vertices onto eigenvectors
    for v in np.asarray(mesh.vertices):
        oldV = np.copy(v)
        v[0] = np.dot(l1, oldV)
        v[1] = np.dot(l2, oldV)
        v[2] = np.dot(l3, oldV)

    return mesh


# 4. Flipping: Use moment test to flip axes
def mesh_flipping(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    fx = fy = fz = 0

    for a, b, c in triangles:
        tricenter = triangleCenter(vertices[a], vertices[b], vertices[c])
        fx += sign(tricenter[0]) * (tricenter[0] ** 2)
        fy += sign(tricenter[1]) * (tricenter[1] ** 2)
        fz += sign(tricenter[2]) * (tricenter[2] ** 2)

    for v in np.asarray(mesh.vertices):
        oldV = np.copy(v)
        v[0] = oldV[0] * sign(fx)
        v[1] = oldV[1] * sign(fy)
        v[2] = oldV[2] * sign(fz)

    return mesh


# Complete normalization pipeline
def mesh_normalize(mesh_path, save_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = mesh_translate(mesh)
    mesh = mesh_resize(mesh)
    mesh = mesh_pose_alignment(mesh)
    mesh = mesh_flipping(mesh)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(save_path, mesh, write_vertex_normals=True)


# For query processing
def mesh_normalize_for_new(mesh):
    mesh = mesh_translate(mesh)
    mesh = mesh_resize(mesh)
    mesh = mesh_pose_alignment(mesh)
    mesh = mesh_flipping(mesh)
    return mesh

def normalize_database(input_folder, output_folder):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    output_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    failed = 0

    for root, dirs, files in os.walk(input_dir):
        print(f"Processing files in: {root} ({i}/69)")
        i += 1
        for file in files:
            if file.endswith(".obj"):
                input_mesh_path = os.path.join(root, file)
                output_mesh_path = os.path.join(output_dir, os.path.relpath(input_mesh_path, input_dir))

                output_subdir = os.path.dirname(output_mesh_path)
                os.makedirs(output_subdir, exist_ok=True)

                try:
                    mesh_normalize(input_mesh_path, output_mesh_path)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
                    failed += 1
            else:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))
                shutil.copy2(input_file_path, output_file_path)

    print(f"\n✓ Finished normalization ({failed} failed shapes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="step2_results/resampled",
                        help='Path to input database')
    parser.add_argument('--output', default="step3_1_results/normalized",
                        help='Path to output database')
    args = parser.parse_args()

    normalize_database(args.input, args.output)