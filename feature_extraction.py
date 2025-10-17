import open3d as o3d
import numpy as np
from pathlib import Path
import os

directory = r"step2_results\\resampled\\"

def finish_norm(mesh_path):
    path = Path(mesh_path)
    stem = path.stem

    underscore_pos = stem.find('_')
        
    if underscore_pos != -1: # This part removes the old suffixes "refined_norm", "simplified_norm", etc. from the file name
        new_stem = stem[:underscore_pos]
    else:
        new_stem = stem


    mesh= o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # PCA

    matrix = [[],
            [],
            [],]

    for x,y,z in vertices:
        matrix[0].append(x)
        matrix[1].append(y)
        matrix[2].append(z)

    A_cov = np.cov(matrix)

    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    major = eigenvectors[:, 0]
    medium = eigenvectors[:, 1]
    minor = eigenvectors[:, 2]

    updated_matrix = []
    for v in vertices:
        x_updated = (np.dot(v, major))
        y_updated = (np.dot(v, medium))
        z_updated = (np.dot(v, np.cross(major, medium)))
        updated_matrix.append([x_updated, y_updated, z_updated])

    # Moment test

    f = np.zeros(3)

    for t in triangles:
        center = (vertices[t[0]] + vertices[t[1]] + vertices[t[2]])/3
        for i in range(3):
            f[i] += np.sign(center[i]) * (center[i] ** 2) 


    new_matrix = []
    for v in updated_matrix:
        v[0] = v[0] * np.sign(f[0])
        v[1] = v[1] * np.sign(f[1])
        v[2] = v[2] * np.sign(f[2])
        new_matrix.append([v[0], v[1], v[2]])
    new_matrix = np.array(new_matrix)


    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_matrix)
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    subdir = os.path.split(os.path.split(mesh_path)[0])[1]
    out_path = Path(f"step3_results/full_normalized/{subdir}/{new_stem}_full_norm.obj")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), new_mesh, write_ascii=True)


folders = Path(directory).glob("*")
for folder in folders:
    files = Path(directory + folder.name).glob("*")
    for file in files:
        finish_norm(directory + folder.name + "\\" + file.name)