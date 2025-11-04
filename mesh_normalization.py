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


def normalize_database(input_folder, output_folder, copy_other_files=False, skip_existing=True):
    """
    Normalize all .obj files in the input folder

    Args:
        input_folder: Path to input database
        output_folder: Path to output database
        copy_other_files: If True, copy non-.obj files (textures, etc.)
        skip_existing: If True, skip files that already exist in output
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    # SAFETY CHECK: Prevent duplicating files
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("ERROR: Input and output folders are the same! This would overwrite your files.")

    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("ERROR: Output folder is inside input folder! This would duplicate files.")

    if input_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("ERROR: Input folder is inside output folder! This would cause issues.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Mesh Normalization")
    print(f"{'=' * 70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Copy other files: {copy_other_files}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'=' * 70}\n")

    # Count total files
    all_obj_files = list(input_dir.rglob("*.obj"))
    total_obj = len(all_obj_files)

    print(f"Found {total_obj} .obj files to process\n")

    processed = 0
    skipped = 0
    failed = 0
    copied = 0

    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)

        for file in files:
            if file.endswith(".obj"):
                input_mesh_path = os.path.join(root, file)
                output_mesh_path = os.path.join(output_dir, os.path.relpath(input_mesh_path, input_dir))

                # Check if already processed
                if skip_existing and os.path.exists(output_mesh_path):
                    print(
                        f"[SKIP] [{processed + skipped + 1}/{total_obj}] Already exists: {os.path.relpath(input_mesh_path, input_dir)}")
                    skipped += 1
                    continue

                output_subdir = os.path.dirname(output_mesh_path)
                os.makedirs(output_subdir, exist_ok=True)

                try:
                    print(
                        f"[PROC] [{processed + skipped + failed + 1}/{total_obj}] Processing: {os.path.relpath(input_mesh_path, input_dir)}")
                    mesh_normalize(input_mesh_path, output_mesh_path)
                    print(f"       [DONE] Saved to: {os.path.relpath(output_mesh_path, output_dir)}")
                    processed += 1
                except Exception as e:
                    print(f"       [FAIL] Failed: {e}")
                    failed += 1

            elif copy_other_files:
                # Only copy non-.obj files if explicitly requested
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))

                output_subdir = os.path.dirname(output_file_path)
                os.makedirs(output_subdir, exist_ok=True)

                if not os.path.exists(output_file_path) or not skip_existing:
                    shutil.copy2(input_file_path, output_file_path)
                    copied += 1

    print(f"\n{'=' * 70}")
    print(f"Normalization Complete!")
    print(f"{'=' * 70}")
    print(f"[DONE] Processed:  {processed} meshes")
    print(f"[SKIP] Skipped:    {skipped} meshes (already existed)")
    print(f"[FAIL] Failed:     {failed} meshes")
    if copy_other_files:
        print(f"[COPY] Copied:     {copied} other files")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize 3D mesh database')
    parser.add_argument('--input', default="step2_results/normalized",
                        help='Path to input database')
    parser.add_argument('--output', default="step3_1_results/normalized",
                        help='Path to output database')
    parser.add_argument('--copy-other-files', action='store_true',
                        help='Copy non-.obj files (textures, materials, etc.)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files in output (default: skip existing)')
    args = parser.parse_args()

    try:
        normalize_database(
            args.input,
            args.output,
            copy_other_files=args.copy_other_files,
            skip_existing=not args.overwrite
        )
    except ValueError as e:
        print(f"\n{e}")
        print("\n[TIP] Make sure input and output paths are different and not nested!")
        exit(1)