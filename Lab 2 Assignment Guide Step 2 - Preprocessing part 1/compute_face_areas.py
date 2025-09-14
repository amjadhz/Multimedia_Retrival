import open3d as o3d
import numpy as np
import pymeshlab as pml


mesh_path = r"./D00386.obj"

# Load the mesh with open3d
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
vertices_o3d = np.asarray(mesh.vertices)
triangles_o3d = np.asarray(mesh.triangles)

# Load the mesh with pymeshlab
meshset_pml = pml.MeshSet()
meshset_pml.load_new_mesh(mesh_path)
meshset_pml.compute_scalar_by_aspect_ratio_per_face(metric="Area")
mesh_pml = meshset_pml.current_mesh()
vertices_pml = mesh_pml.vertex_matrix()
triangles_pml = mesh_pml.face_matrix()

# Custom Euclidean distance function
def get_dist(p1, p2):
    diff = p2 - p1
    dist = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    return dist

# Custom implementation of triangle area computation
def compute_triangle_area(_tri_idx):
    tri = triangles_pml[_tri_idx]
    tri_vertices = [vertices_pml[tri[0]], vertices_pml[tri[1]], vertices_pml[tri[2]]]
    side_a = get_dist(tri_vertices[1], tri_vertices[0])
    side_b = get_dist(tri_vertices[2], tri_vertices[0])
    side_c = get_dist(tri_vertices[2], tri_vertices[1])
    s = 0.5 * (side_a + side_b + side_c)
    tri_area = np.sqrt(s*(s - side_a)*(s - side_b)*(s - side_c)) # Heron's formula
    return tri_area

# Compute single triangle's area using custom implementation
tri_idx = 0
tri_area = compute_triangle_area(tri_idx)
print(f"Area of triangle {tri_idx} (custom implementation) is:   ", tri_area)

# Compute the area of all triangles in the mesh using pymeshlab
areas = mesh_pml.face_scalar_array() # NOTE: the pymeshlab documentation page states that this method is called 
                                     # 'face_scalar_attribute_array', but this appears to be false (since there's no
                                     # such mesh member)
print(f"Area of triangle {tri_idx} (pymeshlab) is:               ", areas[tri_idx])

# Compute geometric measures (including, but not limited to, total mesh area & volume) using pymeshlab
pml_geom_measures = meshset_pml.get_geometric_measures()
print("Total mesh surface area (pymeshlab):             ", pml_geom_measures["surface_area"])
print("Total mesh volume (pymeshlab):                   ", pml_geom_measures["mesh_volume"])

# Compute the area of the entire mesh surface using open3d
mesh_area = mesh.get_surface_area()
print("Total mesh surface area (open3d):                ", mesh_area)

# Compute the area of the entire mesh surface using the custom implementation shown above
summed_custom_area = 0
for i, _ in enumerate(triangles_pml):
    tri_area = compute_triangle_area(i)
    summed_custom_area += tri_area
print("Total mesh surface area (custom implementation): ", summed_custom_area)

# Compute the volume of the entire mesh using open3d
mesh_volume = mesh.get_volume()
print("Total mesh volume (open3d):                      ", mesh_volume)




