from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import random
import math


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


def calculate_A3(vertices, n):
    """
    A3: Angle between 3 random vertices.
    Args:
        vertices: Array of vertex coordinates
        n: Target number of samples
    Returns:
        Array of angles in degrees [0, 180]
    """
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 3.0))
    angles = []

    for i in range(k):
        vi = random.randint(0, N - 1)
        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue
            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                # Get coordinates of three vertices
                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # Calculate vectors
                AB = B - A
                AC = C - A

                # Skip zero vectors
                norm_AB = np.linalg.norm(AB)
                norm_AC = np.linalg.norm(AC)
                if norm_AB < 1e-12 or norm_AC < 1e-12:
                    continue

                # Calculate the angle
                cos_theta = np.dot(AB, AC) / (norm_AB * norm_AC)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                # Convert radians to degrees and store (as per document)
                angles.append(np.degrees(angle))

    return np.array(angles)

def calculate_D1(vertices):
    """
    D1: Distance between barycenter and each vertex.
    Args:
        vertices: Array of vertex coordinates
    Returns:
        Array of distances
    """
    barycenter = np.mean(vertices, axis=0)
    distances = []

    # Traverse all vertices
    for vertex in vertices:
        distance = np.linalg.norm(np.array(vertex) - barycenter)
        distances.append(distance)

    return np.array(distances)

def calculate_D2(vertices, n):
    """
    D2: Distance between 2 random vertices.
    Args:
        vertices: Array of vertex coordinates
        n: Target number of samples
    Returns:
        Array of distances
    """
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 2.0))
    distances = []

    for i in range(k):
        vi = random.randint(0, N - 1)
        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue

            A = np.array(vertices[vi])
            B = np.array(vertices[vj])

            distance = np.linalg.norm(B - A)
            distances.append(distance)

    return np.array(distances)

def calculate_D3(vertices, n):
    """
    D3: Square root of area of triangle.
    Args:
        vertices: Array of vertex coordinates
        n: Target number of samples
    Returns:
        Array of square roots of areas
    """
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 3.0))
    sqrt_areas = []

    for i in range(k):
        vi = random.randint(0, N - 1)
        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue
            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue

                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                # Calculate cross product
                AB = B - A
                AC = C - A
                cross_product = np.cross(AB, AC)

                # Area of triangle
                area = np.linalg.norm(cross_product) / 2.0
                # IMPORTANT: Take square root as per document specification
                sqrt_areas.append(np.sqrt(area))

    return np.array(sqrt_areas)


def calculate_D4(vertices, n):
    """
    D4: Cube root of volume of tetrahedron.
    Args:
        vertices: Array of vertex coordinates
        n: Target number of samples
    Returns:
        Array of cube roots of volumes
    """
    N = len(vertices)
    k = int(math.pow(n, 1.0 / 4.0))
    cbrt_volumes = []

    for i in range(k):
        vi = random.randint(0, N - 1)
        for j in range(k):
            vj = random.randint(0, N - 1)
            if vj == vi:
                continue
            for l in range(k):
                vl = random.randint(0, N - 1)
                if vl == vi or vl == vj:
                    continue
                for m in range(k):
                    vm = random.randint(0, N - 1)
                    if vm == vi or vm == vj or vm == vl:
                        continue

                    A = np.array(vertices[vi])
                    B = np.array(vertices[vj])
                    C = np.array(vertices[vl])
                    D = np.array(vertices[vm])

                    # Volume of tetrahedron
                    volume = np.abs(np.dot(A - D, np.cross(B - D, C - D))) / 6.0
                    cbrt_volumes.append(np.cbrt(volume))

    return np.array(cbrt_volumes)

def compute_distribution_descriptors(mesh: o3d.geometry.TriangleMesh, n_samples=1000000, n_bins=100):
    """
    Compute distribution descriptors using nested loop sampling method.
    Args:
        mesh: Input mesh
        n_samples: Target number of samples.
        n_bins: Number of histogram bins.
    Returns:
        Dictionary of feature values
    """
    vertices = np.asarray(mesh.vertices)
    n_verts = len(vertices)

    print(f"\n=== Distribution Descriptors (Nested Loop Method) ===")
    print(f"Target samples: {n_samples:,}, Vertices: {n_verts:,}, Bins: {n_bins}")

    # Calculate descriptors using nested loops
    print("\nComputing A3 (angles)...")
    A3 = calculate_A3(vertices, n_samples)
    print(f"  Collected {len(A3):,} samples, range: [{A3.min():.2f}°, {A3.max():.2f}°]")

    print("Computing D1 (barycenter distances)...")
    D1 = calculate_D1(vertices)
    print(f"  Collected {len(D1):,} samples, range: [{D1.min():.4f}, {D1.max():.4f}]")

    print("Computing D2 (pairwise distances)...")
    D2 = calculate_D2(vertices, n_samples)
    print(f"  Collected {len(D2):,} samples, range: [{D2.min():.4f}, {D2.max():.4f}]")

    print("Computing D3 (sqrt of triangle areas)...")
    D3 = calculate_D3(vertices, n_samples)
    print(f"  Collected {len(D3):,} samples, range: [{D3.min():.4f}, {D3.max():.4f}]")

    print("Computing D4 (cbrt of tetrahedron volumes)...")
    D4 = calculate_D4(vertices, n_samples)
    print(f"  Collected {len(D4):,} samples, range: [{D4.min():.4f}, {D4.max():.4f}]")

    # Compute normalized histograms with FIXED ranges for comparability
    def hist_to_features(data, name, n_bins):
        if len(data) == 0:
            # Return zero histogram if no data
            return {f'{name}_bin{i}': 0.0 for i in range(n_bins)}

        # CRITICAL FIX: Use FIXED ranges for each descriptor to ensure comparability across shapes
        # Without fixed ranges, different meshes will have different bin boundaries!
        fixed_ranges = {
            'A3': (0, 180),  # Angles in degrees [0°, 180°]
            'D1': (0, 1.0),  # Normalized distance to barycenter (max ≈ 0.866 for unit cube)
            'D2': (0, 2.0),  # Normalized distance between vertices (max ≈ √3 ≈ 1.73)
            'D3': (0, 1.0),  # Normalized sqrt of area
            'D4': (0, 1.0),  # Normalized cbrt of volume
        }

        # Get the appropriate range
        hist_range = fixed_ranges.get(name, (data.min(), data.max()))

        # Compute histogram with fixed range
        hist, _ = np.histogram(data, bins=n_bins, range=hist_range)

        # CRITICAL FIX: Normalize histogram - make each bin a percentage in [0, 1]
        # This allows meaningful comparison between meshes with different numbers of samples
        hist = hist / hist.sum() if hist.sum() > 0 else hist

        return {f'{name}_bin{i}': float(hist[i]) for i in range(n_bins)}

    print("\nGenerating normalized histograms...")
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
    dist_features = compute_distribution_descriptors(mesh, n_samples=1000000, n_bins=100)

    print(f"\n=== Summary ===")
    print(f"  Single-value: {len(single_features)}")
    print(f"  Distribution: {len(dist_features)} ({len(dist_features) // 5} per descriptor)")


if __name__ == "__main__":
    main()