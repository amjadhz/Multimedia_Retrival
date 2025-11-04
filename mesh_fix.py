from pathlib import Path
import os.path
import open3d as o3d
import shutil
import numpy as np
import pymeshfix


# Function to fix the mesh
def mesh_fix(mesh_path, save_path):
    try:
        # Using open3d to load the mesh
        print(f"Reading mesh from {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Convert open3d mesh to numpy array
        vertices = np.asarray(mesh.vertices)

        unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
        # Update triangle faces and replace them with new vertex indices
        triangles = np.asarray(mesh.triangles)
        new_triangles = indices[triangles]

        # Create a new grid
        merged_mesh = o3d.geometry.TriangleMesh()
        merged_mesh.vertices = o3d.utility.Vector3dVector(unique_vertices)
        merged_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        merged_mesh.compute_vertex_normals()

        # Use pymeshfix to fix non-manifold issues and holes in the mesh
        print("Running pymeshfix repair...")
        meshfix = pymeshfix.MeshFix(unique_vertices, new_triangles)
        # Do the repair
        meshfix.repair(verbose=True)

        # Create a new open3d mesh using the repaired vertices and faces
        fixed_mesh = o3d.geometry.TriangleMesh()
        fixed_mesh.vertices = o3d.utility.Vector3dVector(meshfix.v)
        fixed_mesh.triangles = o3d.utility.Vector3iVector(meshfix.f)

        # Recompute vertex normals
        print("Recomputing normals...")
        fixed_mesh.compute_vertex_normals()

        # Save file
        print(f"Saving fixed mesh to {save_path}")
        o3d.io.write_triangle_mesh(save_path, fixed_mesh, write_vertex_normals=True)
        return True
    except Exception as e:
        print(f"Error processing mesh {mesh_path}: {e}")
        return False


# Function to fix the database
def fix_database(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".obj")):
                input_mesh_path = os.path.join(root, file)
                output_mesh_path = os.path.join(output_dir, os.path.relpath(input_mesh_path, input_dir))

                output_subdir = os.path.dirname(output_mesh_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                print(f"Processing file: {input_mesh_path}")
                mesh_fix(input_mesh_path, output_mesh_path)
            else:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir))
                shutil.copy2(input_file_path, output_file_path)

if __name__ == '__main__':
    # Call the function to fix the mesh
    input_directory = "step3_1_results/normalized"
    output_directory = "step3_1_5_results"

    fix_database(input_directory, output_directory)
    print("finished")