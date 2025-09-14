import open3d as o3d
from copy import deepcopy
import numpy as np


mesh_path = r"./D00386.obj"

# Load the mesh with open3d
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

# Add a coordinate frame at the origin
world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame() 

# Scale the mesh
mesh_s = deepcopy(mesh).translate((2,0,0))
mesh_s2 = deepcopy(mesh).translate((2,0,0))
mesh_s.scale(0.5, center=mesh_s.get_center())
mesh_s2.scale(0.5, center=(0, 0, 0))

# Rotate the mesh
mesh_r = deepcopy(mesh).translate((-2,0,0))
mesh_r2 = deepcopy(mesh).translate((-2,0,0))
mesh_r.rotate(mesh_r.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4)))
mesh_r2.rotate(mesh_r2.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4)), center=(0, 0, 0))

# Translate the mesh
mesh.translate((-2, 2, 0))

# Visualize all meshes
o3d.visualization.draw_geometries([mesh, mesh_s, mesh_s2, mesh_r, mesh_r2, world_axes])
