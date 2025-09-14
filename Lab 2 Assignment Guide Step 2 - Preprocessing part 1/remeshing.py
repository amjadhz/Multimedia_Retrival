import open3d as o3d
import pymeshlab as pml
import numpy as np


# Parameters
lowpoly_path = "./lowpoly.obj"
highpoly_path = "./highpoly.obj"
target_edge_length = 0.02
iter_low = 2
iter_high = 7
do_remeshing = True   # Turn off if you don't want to do the remeshing on the fly.
                       # Useful for viewing meshes you've already remeshed.

# Remeshing
if do_remeshing:
    # --- REFINEMENT --- #
    # Load a low-poly input mesh using PyMeshLab
    mesh_lowpoly = pml.MeshSet()
    mesh_lowpoly.load_new_mesh(lowpoly_path)

    # Create refined (i.e. high-poly) versions of it
    print(f"Refining lowpoly mesh ({iter_low} iterations)... ", end="")
    mesh_lowpoly.meshing_isotropic_explicit_remeshing(
        targetlen=pml.PureValue(target_edge_length), iterations=iter_low)
    mesh_lowpoly.save_current_mesh(f"refined_iter{iter_low}.obj")
    print("Finished.")
    print(f"Refining lowpoly mesh ({iter_high} iterations)... ", end="")
    mesh_lowpoly.meshing_isotropic_explicit_remeshing(
        targetlen=pml.PureValue(target_edge_length), iterations=iter_high)
    mesh_lowpoly.save_current_mesh(f"refined_iter{iter_high}.obj")
    print("Finished.")

    # --- DECIMATION --- #
    # Load a high-poly input mesh
    mesh_highpoly = pml.MeshSet()
    mesh_highpoly.load_new_mesh(highpoly_path)

    # Create decimated (i.e. low-poly) versions of it
    print(f"Decimating highpoly mesh ({iter_low} iterations)... ", end="")
    mesh_highpoly.meshing_isotropic_explicit_remeshing(
        targetlen=pml.PureValue(target_edge_length), iterations=iter_low)
    mesh_highpoly.save_current_mesh(f"decimated_iter{iter_low}.obj")
    print("Finished.")
    print(f"Decimating highpoly mesh ({iter_high} iterations)... ", end="")
    mesh_highpoly.meshing_isotropic_explicit_remeshing(
        targetlen=pml.PureValue(target_edge_length), iterations=iter_high)
    mesh_highpoly.save_current_mesh(f"decimated_iter{iter_high}.obj")
    print("Finished.")

# --- VISUALISATION --- #
# Load the meshes with open3d
mesh_lowpoly = o3d.io.read_triangle_mesh(lowpoly_path)
mesh_highpoly = o3d.io.read_triangle_mesh(highpoly_path)
mesh_refined_iter_low = o3d.io.read_triangle_mesh(f"refined_iter{iter_low}.obj")
mesh_refined_iter_high = o3d.io.read_triangle_mesh(f"refined_iter{iter_high}.obj")
mesh_decimated_iter_low = o3d.io.read_triangle_mesh(f"decimated_iter{iter_low}.obj")
mesh_decimated_iter_high = o3d.io.read_triangle_mesh(f"decimated_iter{iter_high}.obj")
mesh_lowpoly.compute_vertex_normals()
mesh_highpoly.compute_vertex_normals()
mesh_refined_iter_low.compute_vertex_normals()
mesh_refined_iter_high.compute_vertex_normals()
mesh_decimated_iter_low.compute_vertex_normals()
mesh_decimated_iter_high.compute_vertex_normals()

# Position the meshes in a 2x3 grid
translation_vector = np.array([1.2, 0, 0])
mesh_refined_iter_low.translate(translation_vector)
translation_vector = np.array([2.4, 0, 0])
mesh_refined_iter_high.translate(translation_vector)
translation_vector = np.array([0, -1, 0])
mesh_highpoly.translate(translation_vector)
translation_vector = np.array([1.2, -1, 0])
mesh_decimated_iter_low.translate(translation_vector)
translation_vector = np.array([2.4, -1, 0])
mesh_decimated_iter_high.translate(translation_vector)

# Print number of triangles per mesh
print("Number of triangles in original lowpoly mesh:",
      np.asarray(mesh_lowpoly.triangles).shape[0])
print("Number of triangles in original highpoly mesh:",
      np.asarray(mesh_highpoly.triangles).shape[0])
print(f"Number of triangles in refined version of lowpoly mesh ({iter_low} iterations):",
      np.asarray(mesh_refined_iter_low.triangles).shape[0])
print(f"Number of triangles in refined version of lowpoly mesh ({iter_high} iterations):",
      np.asarray(mesh_refined_iter_high.triangles).shape[0])
print(f"Number of triangles in decimated version of highpoly mesh ({iter_low} iterations):",
      np.asarray(mesh_decimated_iter_low.triangles).shape[0])
print(f"Number of triangles in decimated version of highpoly mesh ({iter_high} iterations):",
      np.asarray(mesh_decimated_iter_high.triangles).shape[0])

# Visualize the meshes
o3d.visualization.draw_geometries(
    [
        mesh_lowpoly, mesh_highpoly, mesh_refined_iter_low, mesh_refined_iter_high,
        mesh_decimated_iter_low, mesh_decimated_iter_high
    ],
    width=1280,
    height=720,
    mesh_show_wireframe=True
)
