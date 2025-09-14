import open3d as o3d
import numpy as np
from copy import deepcopy


# Parameters
mesh_path = "./D00386.obj"
norm_range = (0, 3)

# Load the mesh with open3d
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# Obtain triangle normals (open3d)
normals_o3d = mesh.triangle_normals

# Obtain triangle normals (custom implementation)
normals_custom_crossmethod = []
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
for i in range(np.asarray(mesh.triangles).shape[0]):
    # Compute triangle normal by taking the crossproduct of two edges and normalizing the result
    vtx_idx_0 = triangles[i][0]
    vtx_idx_1 = triangles[i][1]
    vtx_idx_2 = triangles[i][2]
    v0 = vertices[vtx_idx_0]
    v1 = vertices[vtx_idx_1]
    v2 = vertices[vtx_idx_2]
    edge0 = v1 - v0
    edge1 = v2 - v0
    normal_crossmethod = np.cross(edge0, edge1)
    normal_crossmethod /= np.linalg.norm(normal_crossmethod)
    normals_custom_crossmethod.append(normal_crossmethod)

# Print results
normals_o3d = [str(norm) for norm in normals_o3d]
normals_custom_crossmethod = [str(norm) for norm in normals_custom_crossmethod]
print(f"First {norm_range[1] - norm_range[0]} normals (open3d):\n   ",
      ",\n    ".join(normals_o3d[norm_range[0]: norm_range[1]]))
print(f"First {norm_range[1] - norm_range[0]} normals (custom crossmethod):\n   ",
      ",\n    ".join(normals_custom_crossmethod[norm_range[0]: norm_range[1]]))
