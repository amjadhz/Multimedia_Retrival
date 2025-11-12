import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import math


def triangleCenter(v1, v2, v3):
    """Calculate the center of a triangle."""
    return np.mean([v1, v2, v3], axis=0)


def sign(n):
    """Return the sign of a number."""
    if n < 0:
        return -1
    else:
        return 1


def nbins(data):
    """Calculate number of bins using square root rule."""
    return int(math.sqrt(len(data)))


def print_stats(variable, data):
    """Print mean and standard deviation."""
    print(f"\n{variable}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std:  {np.std(data):.6f}")
    print(f"  Min:  {np.min(data):.6f}")
    print(f"  Max:  {np.max(data):.6f}")


def make_histogram(title, xlabel, data, bins=None, output_dir="figures", version="normalized"):
    """Create and save a histogram."""
    b = nbins(data) if bins is None else bins

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=b, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_path = Path(output_dir) / f"{filename}_{version}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def analyze_translation(mesh_paths, output_dir, version):
    """
    Analyze translation: distance from barycenter to origin.
    """
    print("\n" + "=" * 70)
    print("ANALYZING TRANSLATION (Center to Origin)")
    print("=" * 70)

    distances = []
    outliers = []

    for path in tqdm(mesh_paths, desc="Analyzing translation"):
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh = mesh.remove_duplicated_vertices()

            centroid = mesh.get_center()
            dist = np.linalg.norm(centroid - np.array([0, 0, 0]))

            if dist < 5:  # Filter extreme outliers
                distances.append(dist)
            else:
                outliers.append((path.name, dist))
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    print_stats("Distance to Origin", distances)

    if outliers:
        print(f"\n  Outliers (dist > 5): {len(outliers)}")
        for name, dist in outliers[:5]:  # Show first 5
            print(f"    {name}: {dist:.3f}")

    make_histogram(
        "Distance from Barycenter to Origin",
        "Distance",
        distances,
        bins=np.arange(0, max(distances) + 0.05, 0.05),
        output_dir=output_dir,
        version=version
    )

    return {
        'translation_mean': np.mean(distances),
        'translation_std': np.std(distances),
        'translation_outliers': len(outliers)
    }


def analyze_rotation(mesh_paths, output_dir, version):
    """
    Analyze rotation: alignment of principal axes with coordinate axes.
    """
    print("\n" + "=" * 70)
    print("ANALYZING ROTATION (PCA Alignment)")
    print("=" * 70)

    dotxs = []  # e1 · (1,0,0)
    dotys = []  # e2 · (0,1,0)
    dotzs = []  # e3 · (0,0,1)

    invalid = 0
    misaligned = 0
    threshold = 0.9

    for path in tqdm(mesh_paths, desc="Analyzing rotation"):
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh = mesh.remove_duplicated_vertices()
            vertices = np.asarray(mesh.vertices)

            # Compute covariance matrix
            covariance = np.cov(vertices.T)

            # PCA
            eigenvalues, eigenvectors = np.linalg.eig(covariance)

            # Sort by eigenvalues (descending)
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]

            # Get principal axes
            l1 = eigenvectors[:, 0]
            l2 = eigenvectors[:, 1]
            l3 = np.cross(l1, l2)
            l3 = l3 / np.linalg.norm(l3)

            # Compute dot products (absolute value)
            dotx = abs(np.dot(l1, np.array([1, 0, 0])))
            doty = abs(np.dot(l2, np.array([0, 1, 0])))
            dotz = abs(np.dot(l3, np.array([0, 0, 1])))

            dotxs.append(dotx)
            dotys.append(doty)
            dotzs.append(dotz)

            # Check alignment quality
            if dotx < threshold or doty < threshold or dotz < threshold:
                misaligned += 1

        except Exception as e:
            invalid += 1

    print_stats("Dot Product e1·x", dotxs)
    print_stats("Dot Product e2·y", dotys)
    print_stats("Dot Product e3·z", dotzs)
    print(f"\n  Invalid meshes: {invalid}")
    print(f"  Misaligned (< {threshold}): {misaligned}/{len(mesh_paths)} ({100 * misaligned / len(mesh_paths):.1f}%)")

    # Create histograms
    bins = np.arange(0, 1.02, 0.02)

    make_histogram(
        "Dot Product e1·x (First Principal Axis)",
        "e1 · (1,0,0)",
        dotxs,
        bins=bins,
        output_dir=output_dir,
        version=version
    )

    make_histogram(
        "Dot Product e2·y (Second Principal Axis)",
        "e2 · (0,1,0)",
        dotys,
        bins=bins,
        output_dir=output_dir,
        version=version
    )

    make_histogram(
        "Dot Product e3·z (Third Principal Axis)",
        "e3 · (0,0,1)",
        dotzs,
        bins=bins,
        output_dir=output_dir,
        version=version
    )

    return {
        'rotation_e1x_mean': np.mean(dotxs),
        'rotation_e2y_mean': np.mean(dotys),
        'rotation_e3z_mean': np.mean(dotzs),
        'rotation_misaligned': misaligned,
        'rotation_invalid': invalid
    }


def analyze_scale(mesh_paths, output_dir, version):
    """
    Analyze scale: AABB maximum extent.
    """
    print("\n" + "=" * 70)
    print("ANALYZING SCALE (AABB Max Extent)")
    print("=" * 70)

    max_extents = []
    max_distances = []  # Distance from origin to furthest vertex
    outliers = []

    for path in tqdm(mesh_paths, desc="Analyzing scale"):
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh = mesh.remove_duplicated_vertices()
            vertices = np.asarray(mesh.vertices)

            # Method 1: AABB max extent
            bbox = mesh.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            max_extent = np.max(extent)

            # Method 2: Max distance from origin
            max_dist = np.max(np.linalg.norm(vertices, axis=1))

            if max_extent < 5:  # Filter outliers
                max_extents.append(max_extent)
                max_distances.append(max_dist)
            else:
                outliers.append((path.name, max_extent))

        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    print_stats("AABB Max Extent", max_extents)
    print_stats("Max Distance from Origin", max_distances)

    if outliers:
        print(f"\n  Outliers (extent > 5): {len(outliers)}")

    make_histogram(
        "AABB Maximum Extent",
        "Max Extent",
        max_extents,
        bins=np.arange(0, max(max_extents) + 0.05, 0.05),
        output_dir=output_dir,
        version=version
    )

    make_histogram(
        "Maximum Distance from Origin",
        "Max Distance",
        max_distances,
        bins=np.arange(0, max(max_distances) + 0.05, 0.05),
        output_dir=output_dir,
        version=version
    )

    return {
        'scale_aabb_mean': np.mean(max_extents),
        'scale_aabb_std': np.std(max_extents),
        'scale_maxdist_mean': np.mean(max_distances),
        'scale_outliers': len(outliers)
    }


def analyze_flipping(mesh_paths, output_dir, version):
    """
    Analyze flipping: mass distribution on axes.
    """
    print("\n" + "=" * 70)
    print("ANALYZING FLIPPING (Mass Distribution)")
    print("=" * 70)

    xpos = xneg = 0
    ypos = yneg = 0
    zpos = zneg = 0

    for path in tqdm(mesh_paths, desc="Analyzing flipping"):
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))
            mesh = mesh.remove_duplicated_vertices()
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            fx = fy = fz = 0

            # Calculate moment test
            for a, b, c in triangles:
                tricenter = triangleCenter(vertices[a], vertices[b], vertices[c])
                fx += sign(tricenter[0]) * (tricenter[0] ** 2)
                fy += sign(tricenter[1]) * (tricenter[1] ** 2)
                fz += sign(tricenter[2]) * (tricenter[2] ** 2)

            # Count direction
            if sign(fx) > 0:
                xpos += 1
            else:
                xneg += 1

            if sign(fy) > 0:
                ypos += 1
            else:
                yneg += 1

            if sign(fz) > 0:
                zpos += 1
            else:
                zneg += 1

        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    total = len(mesh_paths)

    print(f"\n  X-axis: {xpos} positive ({100 * xpos / total:.1f}%), {xneg} negative ({100 * xneg / total:.1f}%)")
    print(f"  Y-axis: {ypos} positive ({100 * ypos / total:.1f}%), {yneg} negative ({100 * yneg / total:.1f}%)")
    print(f"  Z-axis: {zpos} positive ({100 * zpos / total:.1f}%), {zneg} negative ({100 * zneg / total:.1f}%)")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.array([-1, 1])
    width = 0.25

    rects1 = ax.bar(x_positions - width, [xneg, xpos], width, label='X-axis', color='steelblue')
    rects2 = ax.bar(x_positions, [yneg, ypos], width, label='Y-axis', color='orange')
    rects3 = ax.bar(x_positions + width, [zneg, zpos], width, label='Z-axis', color='green')

    # Add value labels
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    ax.set_xlabel('Mass Distribution Direction', fontsize=12)
    ax.set_ylabel('Number of Meshes', fontsize=12)
    ax.set_title('Mass Distribution on Each Axis', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    output_path = Path(output_dir) / f"mass_distribution_{version}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_path}")
    plt.close()

    return {
        'flip_x_positive_pct': 100 * xpos / total,
        'flip_y_positive_pct': 100 * ypos / total,
        'flip_z_positive_pct': 100 * zpos / total
    }


def analyze_mesh_complexity(mesh_paths, output_dir, version):
    """
    Analyze mesh complexity: number of vertices and faces.
    """
    print("\n" + "=" * 70)
    print("ANALYZING MESH COMPLEXITY")
    print("=" * 70)

    vertex_counts = []
    face_counts = []

    for path in tqdm(mesh_paths, desc="Analyzing complexity"):
        try:
            mesh = o3d.io.read_triangle_mesh(str(path))

            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.triangles)

            vertex_counts.append(num_vertices)
            face_counts.append(num_faces)

        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    print_stats("Number of Vertices", vertex_counts)
    print_stats("Number of Faces", face_counts)

    make_histogram(
        "Number of Vertices per Mesh",
        "Vertex Count",
        vertex_counts,
        bins=np.arange(0, max(vertex_counts) + 1000, 1000),
        output_dir=output_dir,
        version=version
    )

    make_histogram(
        "Number of Faces per Mesh",
        "Face Count",
        face_counts,
        bins=np.arange(0, max(face_counts) + 1000, 1000),
        output_dir=output_dir,
        version=version
    )

    return {
        'vertices_mean': np.mean(vertex_counts),
        'vertices_std': np.std(vertex_counts),
        'faces_mean': np.mean(face_counts),
        'faces_std': np.std(face_counts)
    }


def generate_summary_report(results, output_dir, version):
    """Generate a summary report with all statistics."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    # Combine all results
    summary = {}
    for result_dict in results:
        summary.update(result_dict)

    # Save to CSV
    df = pd.DataFrame([summary])
    csv_path = Path(output_dir) / f"analysis_summary_{version}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV: {csv_path}")

    # Create a text report
    report_path = Path(output_dir) / f"analysis_report_{version}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"MESH NORMALIZATION QUALITY ANALYSIS REPORT\n")
        f.write(f"Version: {version}\n")
        f.write("=" * 70 + "\n\n")

        f.write("TRANSLATION (Center to Origin)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean distance:     {summary.get('translation_mean', 0):.6f}\n")
        f.write(f"  Std deviation:     {summary.get('translation_std', 0):.6f}\n")
        f.write(f"  Outliers:          {summary.get('translation_outliers', 0)}\n")
        f.write(f"  Status: {'✓ PASS' if summary.get('translation_mean', 1) < 0.01 else '✗ FAIL'}\n\n")

        f.write("ROTATION (PCA Alignment)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  e1·x mean:         {summary.get('rotation_e1x_mean', 0):.6f}\n")
        f.write(f"  e2·y mean:         {summary.get('rotation_e2y_mean', 0):.6f}\n")
        f.write(f"  e3·z mean:         {summary.get('rotation_e3z_mean', 0):.6f}\n")
        f.write(f"  Misaligned:        {summary.get('rotation_misaligned', 0)}\n")
        f.write(f"  Invalid:           {summary.get('rotation_invalid', 0)}\n")
        avg_alignment = (summary.get('rotation_e1x_mean', 0) +
                         summary.get('rotation_e2y_mean', 0) +
                         summary.get('rotation_e3z_mean', 0)) / 3
        f.write(f"  Status: {'✓ PASS' if avg_alignment > 0.9 else '✗ FAIL'}\n\n")

        f.write("SCALE (AABB Max Extent)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  AABB mean:         {summary.get('scale_aabb_mean', 0):.6f}\n")
        f.write(f"  AABB std:          {summary.get('scale_aabb_std', 0):.6f}\n")
        f.write(f"  Max dist mean:     {summary.get('scale_maxdist_mean', 0):.6f}\n")
        f.write(f"  Outliers:          {summary.get('scale_outliers', 0)}\n")
        f.write(f"  Status: {'✓ PASS' if abs(summary.get('scale_aabb_mean', 0) - 1.0) < 0.1 else '✗ FAIL'}\n\n")

        f.write("FLIPPING (Mass Distribution)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  X positive:        {summary.get('flip_x_positive_pct', 0):.1f}%\n")
        f.write(f"  Y positive:        {summary.get('flip_y_positive_pct', 0):.1f}%\n")
        f.write(f"  Z positive:        {summary.get('flip_z_positive_pct', 0):.1f}%\n")
        avg_positive = (summary.get('flip_x_positive_pct', 0) +
                        summary.get('flip_y_positive_pct', 0) +
                        summary.get('flip_z_positive_pct', 0)) / 3
        f.write(f"  Status: {'✓ PASS' if avg_positive > 90 else '✗ FAIL'}\n\n")

        f.write("MESH COMPLEXITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Vertices mean:     {summary.get('vertices_mean', 0):.0f}\n")
        f.write(f"  Vertices std:      {summary.get('vertices_std', 0):.0f}\n")
        f.write(f"  Faces mean:        {summary.get('faces_mean', 0):.0f}\n")
        f.write(f"  Faces std:         {summary.get('faces_std', 0):.0f}\n\n")

        f.write("=" * 70 + "\n")

    print(f"  Saved report: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(
        f"Translation:  {'✓ PASS' if summary.get('translation_mean', 1) < 0.01 else '✗ FAIL'} (mean={summary.get('translation_mean', 0):.6f})")
    print(f"Rotation:     {'✓ PASS' if avg_alignment > 0.9 else '✗ FAIL'} (avg={avg_alignment:.6f})")
    print(
        f"Scale:        {'✓ PASS' if abs(summary.get('scale_aabb_mean', 0) - 1.0) < 0.1 else '✗ FAIL'} (mean={summary.get('scale_aabb_mean', 0):.6f})")
    print(f"Flipping:     {'✓ PASS' if avg_positive > 90 else '✗ FAIL'} (avg={avg_positive:.1f}% positive)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze normalized mesh quality')
    parser.add_argument('--input', default='step2_results/normalized',
                        help='Path to normalized mesh database (default: step2_results/normalized)')
    parser.add_argument('--output', default='analysis_results',
                        help='Output directory for analysis results (default: analysis_results)')
    parser.add_argument('--version', default='normalized',
                        help='Version label for output files (default: normalized)')
    parser.add_argument('--analyses', default='all',
                        help='Comma-separated list of analyses: translation,rotation,scale,flipping,complexity,all (default: all)')
    args = parser.parse_args()

    # Find all .obj files
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    mesh_paths = list(input_path.rglob("*.obj"))

    if len(mesh_paths) == 0:
        print(f"Error: No .obj files found in {input_path}")
        return

    print("\n" + "=" * 70)
    print("MESH NORMALIZATION QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Input directory:  {input_path}")
    print(f"Output directory: {args.output}")
    print(f"Version:          {args.version}")
    print(f"Found meshes:     {len(mesh_paths)}")
    print("=" * 70)

    analyses = args.analyses.lower().split(',')
    run_all = 'all' in analyses

    results = []

    if run_all or 'translation' in analyses:
        results.append(analyze_translation(mesh_paths, args.output, args.version))

    if run_all or 'rotation' in analyses:
        results.append(analyze_rotation(mesh_paths, args.output, args.version))

    if run_all or 'scale' in analyses:
        results.append(analyze_scale(mesh_paths, args.output, args.version))

    if run_all or 'flipping' in analyses:
        results.append(analyze_flipping(mesh_paths, args.output, args.version))

    if run_all or 'complexity' in analyses:
        results.append(analyze_mesh_complexity(mesh_paths, args.output, args.version))

    # Generate summary report
    if results:
        generate_summary_report(results, args.output, args.version)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"All results saved to: {args.output}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()