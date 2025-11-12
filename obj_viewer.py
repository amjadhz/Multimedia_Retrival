import open3d as o3d
import numpy as np

mesh_path = "data/Tool/D01165.obj"

mesh = o3d.io.read_triangle_mesh(mesh_path)

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(vertices.shape[0])
print(triangles.shape[0])

vis_option = "flatshade"
edgesDrawn = True

wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
if vis_option == "smoothshade":
    mesh.compute_vertex_normals()
    if edgesDrawn: o3d.visualization.draw([mesh, wireframe], width=1280, height=720, show_skybox=False) 
    else: o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)
elif vis_option == "flatshade":
    mesh.compute_triangle_normals()
    if edgesDrawn: o3d.visualization.draw([mesh, wireframe], width=1280, height=720, show_skybox=False) 
    else: o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)
elif vis_option == "wireframe":
    o3d.visualization.draw([wireframe], width=1280, height=720, show_skybox=False)
elif vis_option == "unshaded":
    if edgesDrawn: o3d.visualization.draw([mesh, wireframe], width=1280, height=720, show_skybox=False) 
    else: o3d.visualization.draw([mesh], width=1280, height=720, show_skybox=False)