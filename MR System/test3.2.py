from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def safe_read_mesh(p: Path):
    """Read mesh safely."""
    try:
        m = o3d.io.read_triangle_mesh(str(p))
        if m is None or len(m.vertices) == 0:
            return None
        m.remove_degenerate_triangles()
        m.remove_duplicated_vertices()
        m.remove_duplicated_triangles()
        m.remove_non_manifold_edges()
        m.compute_vertex_normals()
        return m
    except Exception:
        return None


def compute_single_descriptors(mesh: o3d.geometry.TriangleMesh):
    """Compute single-value global descriptors"""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    print("\n=== Single-Value Descriptors ===")

    # 1. Surface area
    surface_area = mesh.get_surface_area()
    print(f"1. Surface area: {surface_area:.6f}")

    # 2. Volume
    try:
        volume = abs(mesh.get_volume())
        print(f"2. Volume: {volume:.6f}")
    except:

        try:
            volume = compute_signed_volume(v, f)
            print(f"2. Volume: {volume:.6f} (approximation for non-watertight mesh)")
        except:
            volume = np.nan
            print(f"2. Volume: NaN (computation failed)")

    # 3. Compactness (relative to sphere)
    # Formula: c = S³/(36πV²); sphere = 1.0, others > 1.0
    if surface_area > 0 and not np.isnan(volume):
        compactness = (surface_area ** 3) / (36 * np.pi * (volume ** 2))
        print(f"3. Compactness: {compactness:.6f} (sphere=1.0)")
    else:
        compactness = np.nan
        print(f"3. Compactness: NaN (requires volume)")

    # 4. 3D Rectangularity
    obb = mesh.get_oriented_bounding_box()
    obb_volume = obb.volume()
    rectangularity = volume / obb_volume if obb_volume > 0 else np.nan
    print(f"4. Rectangularity: {rectangularity:.6f}")

    # 5. Diameter
    hull = ConvexHull(v)
    hull_points = v[hull.vertices]
    diameter = np.max(pdist(hull_points))
    print(f"5. Diameter: {diameter:.6f} (convex hull)")

    # 6. Convexity
    hull = ConvexHull(v)
    convex_hull_volume = hull.volume
    convexity = volume / convex_hull_volume if convex_hull_volume > 0 else np.nan
    print(f"6. Convexity: {convexity:.6f}")

    # 7. Eccentricity
    cov_matrix = np.cov(v.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eccentricity = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 1e-12 else np.nan
    print(f"7. Eccentricity: {eccentricity:.6f}")

    return {
        'surface_area': float(surface_area),
        'volume': float(volume),
        'compactness': float(compactness),
        'rectangularity': float(rectangularity),
        'diameter': float(diameter),
        'convexity': float(convexity),
        'eccentricity': float(eccentricity),
    }


def compute_signed_volume(vertices, triangles):
    """
    Compute volume using the tetrahedron decomposition method.
    Formula: V = (1/6) * Σ |det([v1, v2, v3])|
    where v1, v2, v3 are vectors from origin to triangle vertices.
    """
    origin = np.array([0.0, 0.0, 0.0])
    total_volume = 0.0

    for tri in triangles:
        # Get triangle vertices
        x1 = vertices[tri[0]]
        x2 = vertices[tri[1]]
        x3 = vertices[tri[2]]

        # Vectors from origin to vertices
        v1 = x1 - origin
        v2 = x2 - origin
        v3 = x3 - origin

        # Volume of tetrahedron: V = (1/6) * det([v1, v2, v3])
        # Cross product: v1 × v2
        cross_product = np.cross(v1, v2)

        # Scalar triple product: (v1 × v2) · v3
        scalar_triple = np.dot(cross_product, v3)

        # Signed volume of this tetrahedron
        tetrahedron_volume = scalar_triple / 6.0

        total_volume += tetrahedron_volume

    return abs(total_volume)

def compute_distribution_descriptors(mesh: o3d.geometry.TriangleMesh, n_samples=10000, n_bins=64):
    """Compute distribution descriptors (A3, D1, D2, D3, D4)"""
    v = np.asarray(mesh.vertices)
    n_verts = len(v)

    print(f"\n=== Distribution Descriptors ===")
    print(f"Sampling {n_samples} random points, {n_bins} bins")

    centroid = v.mean(axis=0)
    idx = np.random.randint(0, n_verts, (n_samples, 4))

    # A3: angle between 3 random vertices
    idx = np.random.randint(0, n_verts, (n_samples, 4))
    v1, v2, v3 = v[idx[:, 0]], v[idx[:, 1]], v[idx[:, 2]]

    # remove duplicate vertice
    mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
    v1, v2, v3 = v1[mask], v2[mask], v3[mask]

    vec1 = v1 - v2
    vec2 = v3 - v2
    cos_angle = np.sum(vec1 * vec2, axis=1) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1) + 1e-12)
    cos_angle = np.clip(cos_angle, -1, 1)
    A3 = np.arccos(cos_angle)
    print(f"A3 range: [{A3.min():.4f}, {A3.max():.4f}]")

    # D1: distance between barycenter and random vertex
    D1 = np.linalg.norm(v[idx[:, 0]] - centroid, axis=1)
    print(f"D1 range: [{D1.min():.4f}, {D1.max():.4f}]")

    # D2: distance between 2 random vertices
    D2 = np.linalg.norm(v[idx[:, 0]] - v[idx[:, 1]], axis=1)
    print(f"D2 range: [{D2.min():.4f}, {D2.max():.4f}]")

    # D3: square root of area of triangle
    v1, v2, v3 = v[idx[:, 0]], v[idx[:, 1]], v[idx[:, 2]]
    cross = np.cross(v2 - v1, v3 - v1)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    D3 = np.sqrt(area)
    print(f"D3 range: [{D3.min():.4f}, {D3.max():.4f}]")

    # D4: cube root of volume of tetrahedron
    v1, v2, v3, v4 = v[idx[:, 0]], v[idx[:, 1]], v[idx[:, 2]], v[idx[:, 3]]
    mat = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)
    vol = np.abs(np.linalg.det(mat)) / 6.0
    D4 = np.cbrt(vol)
    print(f"D4 range: [{D4.min():.4f}, {D4.max():.4f}]")

    # Compute histograms
    def hist_to_features(data, name, n_bins):
        hist, _ = np.histogram(data, bins=n_bins, range=(data.min(), data.max()))
        hist = hist / hist.sum()
        return {f'{name}_bin{i}': float(hist[i]) for i in range(n_bins)}

    features = {}
    features.update(hist_to_features(A3, 'A3', n_bins))
    features.update(hist_to_features(D1, 'D1', n_bins))
    features.update(hist_to_features(D2, 'D2', n_bins))
    features.update(hist_to_features(D3, 'D3', n_bins))
    features.update(hist_to_features(D4, 'D4', n_bins))

    return features


def main():
    # Test file path
    mesh_path = "step3_results/full_normalized/Car/D00236_full_norm.obj"

    print(f"Testing feature extraction on: {mesh_path}")

    # Read mesh
    mesh = safe_read_mesh(Path(mesh_path))
    if mesh is None:
        print("Failed to read mesh")
        return

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    print(f"\nMesh info:")
    print(f"  Vertices: {len(v)}")
    print(f"  Faces: {len(f)}")

    # Compute features
    single_features = compute_single_descriptors(mesh)
    dist_features = compute_distribution_descriptors(mesh, n_samples=10000, n_bins=64)

    print(f"\n=== Summary ===")
    print(f"  Single-value: {len(single_features)}")
    print(f"  Distribution: {len(dist_features)} ({len(dist_features) // 5} per descriptor)")


if __name__ == "__main__":
    main()