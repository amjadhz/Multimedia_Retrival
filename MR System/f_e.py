from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import random
import math
import pandas as pd
from tqdm import tqdm
import os

# Suppress Open3D warning messages
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


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


def uniform_sample_mesh(mesh, n_points):
    """
    Uniformly sample points on mesh surface (area-weighted sampling).
    """
    try:
        # Use Open3D's built-in uniform sampling
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)
        return points
    except:
        # Fallback: use vertices if sampling fails
        return np.asarray(mesh.vertices)


def compute_single_descriptors(mesh: o3d.geometry.TriangleMesh):
    """Compute single-value global descriptors with robust error handling"""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    # 1. Surface area
    try:
        surface_area = mesh.get_surface_area()
    except:
        surface_area = np.nan

    # 2. Volume
    try:
        volume = abs(mesh.get_volume())
    except:
        try:
            volume = compute_signed_volume(v, f)
        except:
            volume = np.nan

    # 3. Compactness
    try:
        if surface_area > 0 and not np.isnan(volume) and volume > 0:
            compactness = (surface_area ** 3) / (36 * np.pi * (volume ** 2))
        else:
            compactness = np.nan
    except:
        compactness = np.nan

    # 4. Rectangularity
    try:
        obb = mesh.get_oriented_bounding_box()
        obb_volume = obb.volume()
        rectangularity = volume / obb_volume if obb_volume > 0 and not np.isnan(volume) else np.nan
    except:
        rectangularity = np.nan

    # 5. Diameter
    try:
        hull = ConvexHull(v)
        hull_points = v[hull.vertices]
        diameter = np.max(pdist(hull_points))
    except:
        try:
            min_bound = v.min(axis=0)
            max_bound = v.max(axis=0)
            diameter = np.linalg.norm(max_bound - min_bound)
        except:
            diameter = np.nan

    # 6. Convexity
    try:
        hull = ConvexHull(v)
        convex_hull_volume = hull.volume
        convexity = volume / convex_hull_volume if convex_hull_volume > 0 and not np.isnan(volume) else np.nan
    except:
        convexity = np.nan

    # 7. Eccentricity
    try:
        cov_matrix = np.cov(v.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eccentricity = eigenvalues[0] / eigenvalues[2] if eigenvalues[2] > 1e-12 else np.nan
    except:
        eccentricity = np.nan

    return {
        'surface_area': float(surface_area) if not np.isnan(surface_area) else np.nan,
        'volume': float(volume) if not np.isnan(volume) else np.nan,
        'compactness': float(compactness) if not np.isnan(compactness) else np.nan,
        'rectangularity': float(rectangularity) if not np.isnan(rectangularity) else np.nan,
        'diameter': float(diameter) if not np.isnan(diameter) else np.nan,
        'convexity': float(convexity) if not np.isnan(convexity) else np.nan,
        'eccentricity': float(eccentricity) if not np.isnan(eccentricity) else np.nan,
    }


def compute_signed_volume(vertices, triangles):
    """Compute volume using the tetrahedron decomposition method."""
    origin = np.array([0.0, 0.0, 0.0])
    total_volume = 0.0

    for tri in triangles:
        x1 = vertices[tri[0]]
        x2 = vertices[tri[1]]
        x3 = vertices[tri[2]]

        v1 = x1 - origin
        v2 = x2 - origin
        v3 = x3 - origin

        cross_product = np.cross(v1, v2)
        scalar_triple = np.dot(cross_product, v3)
        tetrahedron_volume = scalar_triple / 6.0
        total_volume += tetrahedron_volume

    return abs(total_volume)


def calculate_A3(vertices, n):
    """A3: Angle between 3 random vertices (from uniformly sampled points)."""
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

                A = np.array(vertices[vi])
                B = np.array(vertices[vj])
                C = np.array(vertices[vl])

                AB = B - A
                AC = C - A

                norm_AB = np.linalg.norm(AB)
                norm_AC = np.linalg.norm(AC)
                if norm_AB < 1e-12 or norm_AC < 1e-12:
                    continue

                cos_theta = np.dot(AB, AC) / (norm_AB * norm_AC)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angles.append(np.degrees(angle))

    return np.array(angles)


def calculate_D1(vertices, n):
    """D1: Distance between barycenter and random vertices (from uniformly sampled points)."""
    barycenter = np.mean(vertices, axis=0)
    distances = []
    N = len(vertices)

    # Sample n random vertices from the uniformly sampled points
    for i in range(n):
        vi = random.randint(0, N - 1)
        vertex = vertices[vi]
        distance = np.linalg.norm(vertex - barycenter)
        distances.append(distance)

    return np.array(distances)


def calculate_D2(vertices, n):
    """D2: Distance between 2 random vertices (from uniformly sampled points)."""
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
    """D3: Square root of area of triangle (from uniformly sampled points)."""
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

                AB = B - A
                AC = C - A
                cross_product = np.cross(AB, AC)

                area = np.linalg.norm(cross_product) / 2.0
                sqrt_areas.append(np.sqrt(area))

    return np.array(sqrt_areas)


def calculate_D4(vertices, n):
    """D4: Cube root of volume of tetrahedron (from uniformly sampled points)."""
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

                    volume = np.abs(np.dot(A - D, np.cross(B - D, C - D))) / 6.0
                    cbrt_volumes.append(np.cbrt(volume))

    return np.array(cbrt_volumes)


def compute_distribution_descriptors(mesh: o3d.geometry.TriangleMesh, n_samples=100000, n_bins=64):
    """
    Compute distribution descriptors using UNIFORM surface sampling.
    """
    sampled_points = uniform_sample_mesh(mesh, n_samples)

    # Calculate descriptors on uniformly sampled points
    A3 = calculate_A3(sampled_points, n_samples)
    D1 = calculate_D1(sampled_points, n_samples)
    D2 = calculate_D2(sampled_points, n_samples)
    D3 = calculate_D3(sampled_points, n_samples)
    D4 = calculate_D4(sampled_points, n_samples)

    def hist_to_features(data, name, n_bins):
        if len(data) == 0:
            return {f'{name}_bin{i}': 0.0 for i in range(n_bins)}

        fixed_ranges = {
            'A3': (0, 180),
            'D1': (0, 1.0),
            'D2': (0, 2.0),
            'D3': (0, 1.0),
            'D4': (0, 1.0),
        }

        hist_range = fixed_ranges.get(name, (data.min(), data.max()))
        hist, _ = np.histogram(data, bins=n_bins, range=hist_range)

        hist = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist.astype(float)

        return {f'{name}_bin{i}': float(hist[i]) for i in range(n_bins)}

    features = {}
    features.update(hist_to_features(A3, 'A3', n_bins))
    features.update(hist_to_features(D1, 'D1', n_bins))
    features.update(hist_to_features(D2, 'D2', n_bins))
    features.update(hist_to_features(D3, 'D3', n_bins))
    features.update(hist_to_features(D4, 'D4', n_bins))

    return features


def process_single_class(class_folder, output_csv, n_samples=100000, n_bins=100):
    """
    Process all meshes in a single class folder
    """
    class_path = Path(class_folder)

    if not class_path.exists() or not class_path.is_dir():
        print(f"Error: '{class_folder}' does not exist or is not a directory")
        return

    # Get class name from folder
    class_name = class_path.name

    # Collect all mesh files
    mesh_files = list(class_path.glob('*.obj'))

    if not mesh_files:
        print(f"Error: No .obj files found in '{class_folder}'")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing Class: {class_name} (UNIFORM SAMPLING)")
    print(f"{'=' * 60}")
    print(f"Found {len(mesh_files)} meshes")
    print(f"Parameters: n_samples={n_samples:,}, n_bins={n_bins}")
    print(f"Output: {output_csv}\n")

    # Process all meshes
    all_data = []
    failed_meshes = []

    for mesh_path in tqdm(mesh_files, desc=f"Processing {class_name}"):
        try:
            # Read mesh
            mesh = safe_read_mesh(mesh_path)
            if mesh is None:
                failed_meshes.append((mesh_path.name, "Failed to read"))
                continue

            # Check for degenerate mesh
            vertices = np.asarray(mesh.vertices)
            if len(vertices) < 4:
                failed_meshes.append((mesh_path.name, "Too few vertices"))
                continue

            # Compute features
            single_features = compute_single_descriptors(mesh)
            dist_features = compute_distribution_descriptors(mesh, n_samples, n_bins)

            # Combine all features
            result = {
                'class_name': class_name,
                'file_name': mesh_path.name,
            }
            result.update(single_features)
            result.update(dist_features)

            all_data.append(result)

        except Exception as e:
            failed_meshes.append((mesh_path.name, str(e)))
            tqdm.write(f"Error processing {mesh_path.name}: {str(e)[:50]}")

    # Save to CSV
    if all_data:
        # Create output directory if it doesn't exist
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)

        print(f"\n{'=' * 60}")
        print(f"RESULTS FOR {class_name}")
        print(f"{'=' * 60}")
        print(f"Successfully processed: {len(all_data)}/{len(mesh_files)} meshes")
        print(f"Saved to: {output_csv}")
        print(f"\nFeature Statistics:")
        print(f"   - Total columns: {len(df.columns)}")
        print(f"   - Metadata: 2 (class_name, file_name)")
        print(f"   - Single-value features: 7")
        print(f"   - Distribution features: {5 * n_bins} (5 × {n_bins} bins)")

        # Show single-value feature statistics
        print(f"\nSingle-Value Feature Summary:")
        single_cols = ['surface_area', 'volume', 'compactness', 'rectangularity',
                       'diameter', 'convexity', 'eccentricity']
        stats = df[single_cols].describe().loc[['mean', 'std', 'min', 'max']]
        print(stats.to_string())

    else:
        print(f"\nNo meshes were successfully processed for {class_name}")

    # Report failures
    if failed_meshes:
        print(f"\nFailed to process {len(failed_meshes)} meshes:")
        for fname, reason in failed_meshes[:10]:
            print(f"   - {fname}: {reason[:50]}")
        if len(failed_meshes) > 10:
            print(f"   ... and {len(failed_meshes) - 10} more")

        # Save failed meshes list
        failed_csv = output_csv.replace('.csv', '_failed.csv')
        failed_df = pd.DataFrame(failed_meshes, columns=['file_name', 'reason'])
        failed_df.to_csv(failed_csv, index=False)
        print(f"   Failed meshes saved to: {failed_csv}")

    print(f"\n{'=' * 60}\n")


def main():
    # root that contains all class folders
    ROOT_NORMALIZED = Path("./step3_1_results/normalized")

    N_SAMPLES = 150000
    N_BINS = 100

    # make sure output dir exists
    out_dir = Path("./MR System/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # loop over every subfolder (each is a class)
    for class_folder in ROOT_NORMALIZED.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            output_csv = out_dir / f"{class_name}_descriptors.csv"

            print(f"[INFO] Processing class: {class_name}")
            process_single_class(
                class_folder=str(class_folder),
                output_csv=str(output_csv),
                n_samples=N_SAMPLES,
                n_bins=N_BINS
            )

    print("[DONE] All classes processed. ✅✅✅")


if __name__ == "__main__":
    main()