# Step 2 — Preprocessing & Cleaning (MR assignment)
# Simple, single-file script (Open3D + NumPy + Pandas + Matplotlib)
#
# Outputs in step2_results/ :
#   graphs/
#     raw/overall/*.png
#     raw/<Class>/*.{png,txt}
#     resampled/overall/*.png
#     resampled/<Class>/*.{png,txt}
#     normalized/overall/*.png
#     normalized/<Class>/*.{png,txt}
#   stats_raw.csv,         objects_raw.csv,         outliers_raw.txt
#   resampled/   (ALL meshes: *_refined / *_simplified / *_kept)
#   stats_resampled.csv,   objects_resampled.csv,   outliers_resampled.txt
#   normalized/  (ALL normalized)
#   stats_normalized.csv,  objects_normalized.csv,  outliers_normalized.txt
#
# -------------------------------------------
# MR formulas used (names match variables):
#   V  = vertex array (N x 3), v = V[i]
#   AABB (axis-aligned bbox):
#       vmin = V.min(axis=0)        # (bbox_min_x, bbox_min_y, bbox_min_z)
#       vmax = V.max(axis=0)        # (bbox_max_x, bbox_max_y, bbox_max_z)
#       extent = vmax - vmin        # (extent_x, extent_y, extent_z)
#       bbox_diag = ||extent||_2    # np.linalg.norm(extent)
#   Barycenter (centroid):
#       c = V.mean(axis=0)
#   Normalization (uniform to unit cube):
#       s = 1 / max(extent)         # scale so the largest bbox side becomes 1
#       V' = (V - c) * s            # translate to origin and scale (fits unit cube)
# These are exactly the cell/grid & resampling ideas from the MR lectures.


import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # safe for headless runs

SUPPORTED = {".obj", ".off", ".ply", ".stl", ".glb", ".gltf"}

# Small helpers (simple & clear)
def get_class(path: Path, root: Path) -> str:
    """Infer class from first subfolder under root."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) >= 2 else "Unknown"

def face_type_obj(path: Path, tri_count=None) -> str:
    """Very quick face-type detector for .obj (triangles/quads/ngons/mixed)."""
    if path.suffix.lower() != ".obj":
        return "triangles" if (tri_count is not None and tri_count > 0) else "unknown"
    tri = quad = ngon = 0
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("f "):
                    n = len(line.split()) - 1
                    if n == 3: tri += 1
                    elif n == 4: quad += 1
                    elif n > 4: ngon += 1
    except Exception:
        return "unknown"
    if ngon > 0: return "ngons"
    if tri > 0 and quad > 0: return "triangles+quads"
    if tri > 0: return "triangles"
    if quad > 0: return "quads"
    return "unknown"

def safe_read_mesh(p: Path):
    """Read a mesh and do light cleanup; return None if unreadable."""
    try:
        m = o3d.io.read_triangle_mesh(str(p), enable_post_processing=True)
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

# 2.1 Analyze one shape
def analyze_mesh_record(mesh, path: Path, cls: str, force_face_type=None):
    """Compute stats for a mesh (used in all phases)."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    vmin, vmax = v.min(0), v.max(0)      # AABB
    extent = vmax - vmin
    return dict(
        cls=cls, path=str(path), ext=path.suffix.lower(),
        n_vertices=int(len(v)), n_faces=int(len(t)), n_triangles=int(len(t)),
        face_type=force_face_type if force_face_type else face_type_obj(path, len(t)),
        bbox_min_x=float(vmin[0]), bbox_min_y=float(vmin[1]), bbox_min_z=float(vmin[2]),
        bbox_max_x=float(vmax[0]), bbox_max_y=float(vmax[1]), bbox_max_z=float(vmax[2]),
        extent_x=float(extent[0]), extent_y=float(extent[1]), extent_z=float(extent[2]),
        bbox_diag=float(np.linalg.norm(extent)),
    )

# 2.3 Resampling (simple policy)
#   Goal: keep vertex count in [5000, 10000],
#         with a bias toward ~5500–7000 for efficiency & quality.
def refine_until(mesh, target_vertices):
    """Loop subdivision until we reach ~target (stop at ~0.8*target to avoid overshoot)."""
    m = mesh
    while len(m.vertices) < max(200, int(0.8 * target_vertices)):
        m = m.subdivide_loop(number_of_iterations=1)
        m.remove_degenerate_triangles()
        m.remove_duplicated_vertices()
        m.compute_vertex_normals()
        if len(m.vertices) > 5 * target_vertices:  # very defensive
            break
    return m

def simplify_to(mesh, target_vertices):
    """Quadric decimation to ~target (Open3D uses triangle count; it's OK as proxy)."""
    target_tris = max(1000, int(target_vertices))  # basic floor
    m = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    m.remove_degenerate_triangles()
    m.remove_duplicated_vertices()
    m.remove_non_manifold_edges()
    m.compute_vertex_normals()
    return m

def resample_smart(mesh, n, prefer_low=True):
    """
    SIMPLE, range-aware rule:
      - n < 5000         -> refine to ~5500  (suffix: _refined)
      - 5000 ≤ n ≤ 7000  -> keep             (_kept)
      - 7000 < n < 8500  -> small simplify:
                              prefer_low  -> ~6800
                              prefer_high -> ~8800
                           (_simplified)
      - n ≥ 8500 & ≤10000-> keep             (_kept)
      - n > 10000        -> simplify in one go to ~9000
                           (_simplified)
    """
    if n < 5000:
        m2 = refine_until(mesh, 5500)
        # if overshoot >6000, nudge down a bit
        if len(m2.vertices) > 6000:
            m2 = simplify_to(m2, 5500)
        return m2, "_refined"

    if 5000 <= n <= 7000:
        return mesh, "_kept"

    if 7000 < n < 8500:
        target = 6800 if prefer_low else 8800
        m2 = simplify_to(mesh, target)
        if len(m2.vertices) < 5000:
            m2 = refine_until(m2, 5500)
        return m2, "_simplified"

    if n <= 10000:
        return mesh, "_kept"

    # n > 10000
    m2 = simplify_to(mesh, 9000)
    if len(m2.vertices) < 5000:
        m2 = refine_until(m2, 5500)
    return m2, "_simplified"

# 2.5 Normalization (center + scale to unit cube)
def normalize_mesh(mesh):
    """
    2.5 — Normalization (uniform):
        c = mean(V)           # barycenter
        extent = max(V) - min(V)
        s = 1 / max(extent)
        V' = (V - c) * s
    """
    v = np.asarray(mesh.vertices)
    c = v.mean(axis=0)
    v_centered = v - c
    vmin = v_centered.min(0); vmax = v_centered.max(0)
    extent = vmax - vmin
    s = 1.0 / max(extent.max(), 1e-12)
    v_norm = v_centered * s
    m2 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v_norm),
        triangles=mesh.triangles
    )
    m2.compute_vertex_normals()
    return m2

# Plotting (overall + per-class)
def _ensure(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    return d

def _band_lines():
    for x in (5000, 10000):
        plt.axvline(x, linestyle="--", linewidth=1)

def save_overall_plots(df: pd.DataFrame, out_root: Path, stage: str):
    """Overall graphs for a stage -> graphs/<stage>/0 overall/*.png"""
    figdir = _ensure(out_root / "graphs" / stage / "0 overall")

    plt.figure(figsize=(8,6))
    df["n_vertices"].hist(bins=40); _band_lines()
    plt.xlabel("n_vertices"); plt.ylabel("count"); plt.title(f"{stage.upper()} — Histogram of n_vertices")
    plt.tight_layout(); plt.savefig(figdir / f"{stage}_hist_n_vertices.png", dpi=150); plt.close()

    plt.figure(figsize=(8,6))
    df["n_faces"].hist(bins=40)
    plt.xlabel("n_faces"); plt.ylabel("count"); plt.title(f"{stage.upper()} — Histogram of n_faces")
    plt.tight_layout(); plt.savefig(figdir / f"{stage}_hist_n_faces.png", dpi=150); plt.close()

    plt.figure(figsize=(10,6))
    df["cls"].value_counts().plot(kind="bar")
    plt.xlabel("class"); plt.ylabel("count"); plt.title(f"{stage.upper()} — Shapes per class")
    plt.tight_layout(); plt.savefig(figdir / f"{stage}_bar_classes.png", dpi=150); plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(df["n_vertices"], df["n_faces"], s=10); _band_lines()
    plt.xlabel("n_vertices"); plt.ylabel("n_faces"); plt.title(f"{stage.upper()} — Vertices vs Faces")
    plt.tight_layout(); plt.savefig(figdir / f"{stage}_scatter_vertices_vs_faces.png", dpi=150); plt.close()

    # small text summary
    in_band = ((df["n_vertices"] >= 5000) & (df["n_vertices"] <= 10000)).sum()
    total = len(df)
    with open(figdir / f"{stage}_band_compliance.txt", "w") as f:
        f.write(f"Within [5000, 10000]: {in_band} / {total} ({(in_band/total*100):.1f}%)\n")

    # for normalized, add a tiny “closer to 5k or 10k” pie
    if stage == "normalized" and total > 0:
        dist5 = (df["n_vertices"] - 5000).abs()
        dist10 = (df["n_vertices"] - 10000).abs()
        closer5 = (dist5 <= dist10).sum(); closer10 = total - closer5
        plt.figure(figsize=(6,6))
        pd.Series({"closer_to_5k": closer5, "closer_to_10k": closer10}).plot(kind="pie", autopct="%1.0f%%")
        plt.ylabel(""); plt.title("Normalized — Closer to 5k vs 10k")
        plt.tight_layout(); plt.savefig(figdir / "normalized_closer_5k_vs_10k.png", dpi=150); plt.close()

def save_class_plots(df: pd.DataFrame, out_root: Path, stage: str, k_outliers: int = 5):
    """
    Per-class graphs & summaries:
      - histograms (n_vertices with band lines, n_faces)
      - boxplots (n_vertices, n_faces)
      - scatter (n_vertices vs n_faces)
      - pies (face_type, ext)
      - outliers_<Class>.txt (smallest/largest)
      - class_report.txt (count, means/medians, bbox stats, common face type/ext)
    """
    base = out_root / "graphs" / stage
    for cls, g in df.groupby("cls"):
        cdir = _ensure(base / cls)

        # Histogram: vertices
        plt.figure(figsize=(8,6))
        g["n_vertices"].hist(bins=30); _band_lines()
        plt.xlabel("n_vertices"); plt.ylabel("count")
        plt.title(f"{stage.upper()} — {cls} — Histogram of n_vertices")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_hist_n_vertices.png", dpi=150); plt.close()

        # Histogram: faces
        plt.figure(figsize=(8,6))
        g["n_faces"].hist(bins=30)
        plt.xlabel("n_faces"); plt.ylabel("count")
        plt.title(f"{stage.upper()} — {cls} — Histogram of n_faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_hist_n_faces.png", dpi=150); plt.close()

        # Boxplots
        plt.figure(figsize=(7,5))
        plt.boxplot(g["n_vertices"].values, vert=True, labels=["n_vertices"])
        plt.title(f"{stage.upper()} — {cls} — Boxplot n_vertices")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_box_vertices.png", dpi=150); plt.close()

        plt.figure(figsize=(7,5))
        plt.boxplot(g["n_faces"].values, vert=True, labels=["n_faces"])
        plt.title(f"{stage.upper()} — {cls} — Boxplot n_faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_box_faces.png", dpi=150); plt.close()

        # Scatter
        plt.figure(figsize=(8,6))
        plt.scatter(g["n_vertices"], g["n_faces"], s=12); _band_lines()
        plt.xlabel("n_vertices"); plt.ylabel("n_faces")
        plt.title(f"{stage.upper()} — {cls} — Vertices vs Faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_scatter_vertices_vs_faces.png", dpi=150); plt.close()

        # Face-type pie
        ft_counts = g["face_type"].value_counts()
        if len(ft_counts) > 0:
            plt.figure(figsize=(6,6))
            ft_counts.plot(kind="pie", autopct="%1.0f%%")
            plt.ylabel(""); plt.title(f"{stage.upper()} — {cls} — Face types")
            plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_pie_face_types.png", dpi=150); plt.close()

        # Extension pie
        ext_counts = g["ext"].value_counts()
        if len(ext_counts) > 0:
            plt.figure(figsize=(6,6))
            ext_counts.plot(kind="pie", autopct="%1.0f%%")
            plt.ylabel(""); plt.title(f"{stage.upper()} — {cls} — File extensions")
            plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_pie_extensions.png", dpi=150); plt.close()

        # Outliers per class
        k = min(k_outliers, len(g))
        small = g.sort_values(["n_faces","n_vertices"]).head(k)
        large = g.sort_values(["n_faces","n_vertices"], ascending=[False,False]).head(k)
        with open(cdir / f"outliers_{cls}.txt", "w") as f:
            f.write(f"== {cls} — Smallest ({stage}) ==\n")
            f.write(small[["n_vertices","n_faces","path"]].to_string(index=False))
            f.write("\n\n== {cls} — Largest ({stage}) ==\n")
            f.write(large[["n_vertices","n_faces","path"]].to_string(index=False))
            f.write("\n")

        # Mini class report
        report = {
            "count": int(len(g)),
            "verts_mean": float(g["n_vertices"].mean()),
            "verts_median": float(g["n_vertices"].median()),
            "faces_mean": float(g["n_faces"].mean()),
            "faces_median": float(g["n_faces"].median()),
            "bbox_diag_mean": float(g["bbox_diag"].mean()),
            "bbox_diag_median": float(g["bbox_diag"].median()),
            "common_face_type": ft_counts.idxmax() if len(ft_counts) else "unknown",
            "common_ext": ext_counts.idxmax() if len(ext_counts) else "unknown",
            "in_band_5k_10k": int(((g["n_vertices"] >= 5000) & (g["n_vertices"] <= 10000)).sum()),
        }
        with open(cdir / "class_report.txt", "w") as f:
            for k_, v_ in report.items():
                f.write(f"{k_}: {v_}\n")

# Main (simple, linear pipeline)
def main():
    parser = argparse.ArgumentParser("Step 2 — Preprocess & Clean 3D meshes (simple)")
    parser.add_argument("--root", type=str, default="data", help="dataset root folder")
    parser.add_argument("--out",  type=str, default="step2_results", help="output folder")
    parser.add_argument("--prefer_high", action="store_true",
                        help="bias in-band meshes toward ~9k (default bias is ~6–7k)")
    parser.add_argument("--limit", type=int, default=0, help="process only first N files (speed check)")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    resampled_dir = outdir / "resampled"
    normalized_dir = outdir / "normalized"
    for d in (outdir, resampled_dir, normalized_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Scan all supported files
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print("No meshes found. Supported:", ", ".join(sorted(SUPPORTED))); sys.exit(1)

    # 2.1 Analyze RAW
    raw_rows = []
    for p in files:
        m = safe_read_mesh(p)
        if m is None: 
            continue
        rec = analyze_mesh_record(m, p, get_class(p, root))
        raw_rows.append(rec)

    if not raw_rows:
        print("Meshes could not be read / were empty."); sys.exit(1)

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(outdir / "stats_raw.csv", index=False)
    df_raw[["cls","path","n_vertices","n_faces","n_triangles"]].to_csv(outdir / "objects_raw.csv", index=False)
    save_overall_plots(df_raw, outdir, "raw")
    save_class_plots(df_raw, outdir, "raw")

    # outliers
    k = min(5, len(df_raw))
    small = df_raw.sort_values(["n_faces","n_vertices"]).head(k)
    large = df_raw.sort_values(["n_faces","n_vertices"], ascending=[False,False]).head(k)
    median_nv = df_raw.n_vertices.median()
    avg_row = df_raw.iloc[(df_raw.n_vertices - median_nv).abs().argsort()[:1]]
    with open(outdir / "outliers_raw.txt", "w") as f:
        f.write("== Smallest (raw) ==\n"); f.write(small[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Largest (raw) ==\n"); f.write(large[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Average (raw) ==\n"); f.write(avg_row[["cls","n_vertices","n_faces","path"]].to_string(index=False)); f.write("\n")

    # 2.3 Resample ALL (keep, refine, or simplify)
    resampled_records = []
    for _, r in df_raw.iterrows():
        p = Path(r["path"])
        cls = r["cls"]
        m = safe_read_mesh(p)
        if m is None:
            continue
        n = int(r["n_vertices"])
        m2, suffix = resample_smart(m, n, prefer_low=(not args.prefer_high))

        out_cls = resampled_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        out_path = out_cls / f"{p.stem}{suffix}.obj"
        o3d.io.write_triangle_mesh(str(out_path), m2, write_ascii=True)

        resampled_records.append(analyze_mesh_record(m2, out_path, cls, force_face_type="triangles"))

    df_res = pd.DataFrame(resampled_records)
    df_res.to_csv(outdir / "stats_resampled.csv", index=False)
    df_res[["cls","path","n_vertices","n_faces","n_triangles"]].to_csv(outdir / "objects_resampled.csv", index=False)
    save_overall_plots(df_res, outdir, "resampled")
    save_class_plots(df_res, outdir, "resampled")

    small_r = df_res.sort_values(["n_faces","n_vertices"]).head(min(5, len(df_res)))
    large_r = df_res.sort_values(["n_faces","n_vertices"], ascending=[False,False]).head(min(5, len(df_res)))
    med_r = df_res.n_vertices.median()
    avg_r = df_res.iloc[(df_res.n_vertices - med_r).abs().argsort()[:1]]
    with open(outdir / "outliers_resampled.txt", "w") as f:
        f.write("== Smallest (resampled) ==\n"); f.write(small_r[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Largest (resampled) ==\n"); f.write(large_r[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Average (resampled) ==\n"); f.write(avg_r[["cls","n_vertices","n_faces","path"]].to_string(index=False)); f.write("\n")

    # 2.5 Normalize ALL resampled (counts unchanged; only center+scale)
    norm_records = []
    for _, r in df_res.iterrows():
        p = Path(r["path"])
        m = safe_read_mesh(p)
        if m is None: 
            continue
        m_norm = normalize_mesh(m)

        out_cls = normalized_dir / r["cls"]
        out_cls.mkdir(parents=True, exist_ok=True)
        out_path = out_cls / f"{p.stem}_norm.obj"
        o3d.io.write_triangle_mesh(str(out_path), m_norm, write_ascii=True)

        norm_records.append(analyze_mesh_record(m_norm, out_path, r["cls"], force_face_type="triangles"))

    df_norm = pd.DataFrame(norm_records)
    df_norm.to_csv(outdir / "stats_normalized.csv", index=False)
    df_norm[["cls","path","n_vertices","n_faces","n_triangles"]].to_csv(outdir / "objects_normalized.csv", index=False)
    save_overall_plots(df_norm, outdir, "normalized")
    save_class_plots(df_norm, outdir, "normalized")

    small_n = df_norm.sort_values(["n_faces","n_vertices"]).head(min(5, len(df_norm)))
    large_n = df_norm.sort_values(["n_faces","n_vertices"], ascending=[False,False]).head(min(5, len(df_norm)))
    med_n = df_norm.n_vertices.median()
    avg_n = df_norm.iloc[(df_norm.n_vertices - med_n).abs().argsort()[:1]]
    with open(outdir / "outliers_normalized.txt", "w") as f:
        f.write("== Smallest (normalized) ==\n"); f.write(small_n[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Largest (normalized) ==\n"); f.write(large_n[["cls","n_vertices","n_faces","path"]].to_string(index=False))
        f.write("\n\n== Average (normalized) ==\n"); f.write(avg_n[["cls","n_vertices","n_faces","path"]].to_string(index=False)); f.write("\n")

    print("[DONE] Step 2 complete.")
    print("Graphs: step2_results/graphs/{raw,resampled,normalized}/{overall,<Class>}/")
    print("Meshes: resampled/ and normalized/ contain ALL objects (kept/refined/simplified).")

if __name__ == "__main__":
    main()
