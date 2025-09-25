import open3d as o3d
import numpy as np

mesh_path = "data\Starship\m1386.obj"

mesh = o3d.io.read_triangle_mesh(mesh_path)

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(vertices.shape[0])
print(triangles.shape[0])

vis_option = "smoothshade"
if vis_option == "smoothshade":
    mesh.compute_vertex_normals()
    o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)
elif vis_option == "flatshade":
    mesh.compute_triangle_normals()
    o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)
elif vis_option == "wireframe_on_shaded":
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
    o3d.visualization.draw([mesh, wireframe], width=1280, height=720, show_skybox=False)
elif vis_option == "wireframe":
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
    o3d.visualization.draw([wireframe], width=1280, height=720, show_skybox=False)
elif vis_option == "unshaded":
    o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)