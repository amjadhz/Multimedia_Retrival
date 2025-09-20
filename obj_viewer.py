import open3d as o3d
import numpy as np

mesh_path = "data\Apartment\D00045.obj"

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

vis_option = "wireframe_on_shaded"
if vis_option == "smoothshade":
    o3d.visualization.draw_geometries([mesh], width=1280, height=720)
elif vis_option == "wireframe_on_shaded":
    o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
elif vis_option == "wireframe":
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
    o3d.visualization.draw_geometries([wireframe], width=1280, height=720)