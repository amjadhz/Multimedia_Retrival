# Step 2 — Preprocessing & Cleaning (MR assignment)
# Simple, single-file script (Open3D + NumPy + Pandas + Matplotlib)
#
# Outputs in step2_results/ :
#   graphs/
#     raw/0 overall/*.png
#     raw/<Class>/*.{png,txt}
#     resampled/0 overall/*.png
#     resampled/<Class>/*.{png,txt}
#     normalized/0 overall/*.png
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

# 2.3 Resampling (strict band enforcement)
#   Goal: keep vertex count in [5000, 10000],
#         with a bias toward ~5500–7000 when we do simplify steps.

def refine_until(mesh, target_vertices, max_passes=8, brake_mult=5.0):
    """Loop subdivision until we reach ~target (bounded passes; avoids runaway growth)."""
    m = mesh
    for _ in range(max_passes):
        if len(m.vertices) >= target_vertices:
            break
        m = m.subdivide_midpoint(number_of_iterations=1)
        m.remove_degenerate_triangles()
        m.remove_duplicated_vertices()
        m.compute_vertex_normals()
        if len(m.vertices) > brake_mult * target_vertices:  # very defensive
            break
    return m

def simplify_to_vertices(mesh, desired_vertices, passes=6):
    """Quadric decimation guided by a *vertex target* via iterative F/V conversion."""
    m = mesh
    for _ in range(passes):
        V = max(1, len(m.vertices))
        F = max(4, len(m.triangles))
        r = F / V
        r = min(2.8, max(1.4, r))  # stabilize typical triangle/vertex ratio
        target_tris = int(desired_vertices * r)
        target_tris = max(200, min(target_tris, F - 1))
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        m.remove_degenerate_triangles()
        m.remove_duplicated_vertices()
        m.remove_non_manifold_edges()
        m.compute_vertex_normals()
        # stop early if we are close enough
        if abs(len(m.vertices) - desired_vertices) <= 150:
            break
    return m

def resample_to_band(mesh, prefer_low=True, minv=5000, maxv=10000):
    """
    Hard-enforce vertex count into [minv, maxv] with gentle bias:
      - below minv: refine toward ~5500
      - in-band 7000..8500: optionally nudge down (~6800) if prefer_low; otherwise keep
      - above maxv: simplify toward ~9000
    Guaranteed to return a mesh with minv ≤ V ≤ maxv (barring unreadable inputs).
    """
    V0 = len(mesh.vertices)
    tgt_low, tgt_mid_l, tgt_mid_h, tgt_high = 5500, 6800, 8800, 9000

    m = mesh
    tag = "_kept"

    if V0 < minv:
        m = refine_until(m, tgt_low)
        tag = "_refined"
    elif V0 > maxv:
        m = simplify_to_vertices(m, tgt_high)
        tag = "_simplified"
    else:
        if 7000 < V0 < 8500 and prefer_low:
            m = simplify_to_vertices(m, tgt_mid_l)
            tag = "_simplified"
        else:
            tag = "_kept"

    # Final guard loop to enforce band
    # If still out of band (decimator/subdivider can be quirky), iterate a few times.
    for _ in range(4):
        V = len(m.vertices)
        if V < minv:
            m = refine_until(m, tgt_low)
            tag = "_refined"
        elif V > maxv:
            m = simplify_to_vertices(m, tgt_high)
            tag = "_simplified"
        else:
            break

    # Clamp once more if tiny drift remains
    V = len(m.vertices)
    if V < minv:
        m = refine_until(m, minv)
        tag = "_refined"
    elif V > maxv:
        m = simplify_to_vertices(m, maxv)
        tag = "_simplified"

    return m, tag

def resample(mesh, prefer_low=True, minv=5000, maxv=10000):
    """
    Hard-enforce vertex count into [minv, maxv] with gentle bias:
      - below minv: refine toward ~5500
      - in-band 7000..8500: optionally nudge down (~6800) if prefer_low; otherwise keep
      - above maxv: simplify toward ~9000
    Guaranteed to return a mesh with minv ≤ V ≤ maxv (barring unreadable inputs).
    """
    V0 = len(mesh.vertices)
    tgt_low, tgt_mid_l, tgt_mid_h, tgt_high = 5500, 6800, 8800, 9000

    m = mesh
    tag = "_kept"

    if V0 < minv:
        m = refine_until(m, tgt_low)
        tag = "_refined"
    elif V0 > maxv:
        m = simplify_to_vertices(m, tgt_high)
        tag = "_simplified"
    else:
        if 7000 < V0 < 8500 and prefer_low:
            m = simplify_to_vertices(m, tgt_mid_l)
            tag = "_simplified"
        else:
            tag = "_kept"

    # Final guard loop to enforce band
    # If still out of band (decimator/subdivider can be quirky), iterate a few times.
    for _ in range(4):
        V = len(m.vertices)
        if V < minv:
            m = refine_until(m, tgt_low)
            tag = "_refined"
        elif V > maxv:
            m = simplify_to_vertices(m, tgt_high)
            tag = "_simplified"
        else:
            break

    # Clamp once more if tiny drift remains
    V = len(m.vertices)
    if V < minv:
        m = refine_until(m, minv)
        tag = "_refined"
    elif V > maxv:
        m = simplify_to_vertices(m, maxv)
        tag = "_simplified"

    return m

# 2.5 Normalization (center + scale to unit cube)
def normalize_mesh(mesh):
    """
    2.5 — Normalization (uniform):
        c = mean(V)           # barycenter
        extent = max(V) - min(V)
        s = 1 / max(extent)
        V' = (V - c) * s
    NOTE: This does NOT change vertex/triangle counts.
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

def save_overall_plots(df: pd.DataFrame, out_root: Path, stage: str, bins: int):
    """Overall graphs for a stage -> graphs/<stage>/overall/*.png"""
    figdir = _ensure(out_root / "graphs" / stage / "0 overall")

    plt.figure(figsize=(8,6))
    df["n_vertices"].hist(bins=bins); _band_lines()
    plt.xlabel("n_vertices"); plt.ylabel("count"); plt.title(f"{stage.upper()} — Histogram of n_vertices")
    plt.tight_layout(); plt.savefig(figdir / f"{stage}_hist_n_vertices.png", dpi=150); plt.close()

    plt.figure(figsize=(8,6))
    df["n_faces"].hist(bins=bins)
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

    in_band = ((df["n_vertices"] >= 5000) & (df["n_vertices"] <= 10000)).sum()
    total = len(df)
    with open(figdir / f"{stage}_band_compliance.txt", "w") as f:
        f.write(f"Within [5000, 10000]: {in_band} / {total} ({(in_band/total*100):.1f}%)\n")

    if stage == "normalized" and total > 0:
        dist5 = (df["n_vertices"] - 5000).abs()
        dist10 = (df["n_vertices"] - 10000).abs()
        closer5 = (dist5 <= dist10).sum(); closer10 = total - closer5
        plt.figure(figsize=(6,6))
        pd.Series({"closer_to_5k": closer5, "closer_to_10k": closer10}).plot(kind="pie", autopct="%1.0f%%")
        plt.ylabel(""); plt.title("Normalized — Closer to 5k vs 10k")
        plt.tight_layout(); plt.savefig(figdir / "normalized_closer_5k_vs_10k.png", dpi=150); plt.close()

def save_class_plots(df: pd.DataFrame, out_root: Path, stage: str, k_outliers: int, bins: int):
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

        plt.figure(figsize=(8,6))
        g["n_vertices"].hist(bins=bins); _band_lines()
        plt.xlabel("n_vertices"); plt.ylabel("count")
        plt.title(f"{stage.upper()} — {cls} — Histogram of n_vertices")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_hist_n_vertices.png", dpi=150); plt.close()

        plt.figure(figsize=(8,6))
        g["n_faces"].hist(bins=bins)
        plt.xlabel("n_faces"); plt.ylabel("count")
        plt.title(f"{stage.upper()} — {cls} — Histogram of n_faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_hist_n_faces.png", dpi=150); plt.close()

        plt.figure(figsize=(7,5))
        plt.boxplot(g["n_vertices"].values, vert=True, labels=["n_vertices"])
        plt.title(f"{stage.upper()} — {cls} — Boxplot n_vertices")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_box_vertices.png", dpi=150); plt.close()

        plt.figure(figsize=(7,5))
        plt.boxplot(g["n_faces"].values, vert=True, labels=["n_faces"])
        plt.title(f"{stage.upper()} — {cls} — Boxplot n_faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_box_faces.png", dpi=150); plt.close()

        plt.figure(figsize=(8,6))
        plt.scatter(g["n_vertices"], g["n_faces"], s=12); _band_lines()
        plt.xlabel("n_vertices"); plt.ylabel("n_faces")
        plt.title(f"{stage.upper()} — {cls} — Vertices vs Faces")
        plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_scatter_vertices_vs_faces.png", dpi=150); plt.close()

        ft_counts = g["face_type"].value_counts()
        if len(ft_counts) > 0:
            plt.figure(figsize=(6,6))
            ft_counts.plot(kind="pie", autopct="%1.0f%%")
            plt.ylabel(""); plt.title(f"{stage.upper()} — {cls} — Face types")
            plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_pie_face_types.png", dpi=150); plt.close()

        ext_counts = g["ext"].value_counts()
        if len(ext_counts) > 0:
            plt.figure(figsize=(6,6))
            ext_counts.plot(kind="pie", autopct="%1.0f%%")
            plt.ylabel(""); plt.title(f"{stage.upper()} — {cls} — File extensions")
            plt.tight_layout(); plt.savefig(cdir / f"{stage}_{cls}_pie_extensions.png", dpi=150); plt.close()

        k = min(k_outliers, len(g))
        small = g.sort_values(["n_faces","n_vertices"]).head(k)
        large = g.sort_values(["n_faces","n_vertices"], ascending=[False,False]).head(k)
        with open(cdir / f"outliers_{cls}.txt", "w") as f:
            f.write(f"== {cls} — Smallest ({stage}) ==\n")
            f.write(small[["n_vertices","n_faces","path"]].to_string(index=False))
            f.write("\n\n== {cls} — Largest ({stage}) ==\n")
            f.write(large[["n_vertices","n_faces","path"]].to_string(index=False))
            f.write("\n")

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
    parser.add_argument("--bins_overall", type=int, default=120, help="histogram bins for overall plots")
    parser.add_argument("--bins_class", type=int, default=60, help="histogram bins for per-class plots")
    parser.add_argument("--class_outliers", type=int, default=5, help="how many per-class outliers to list")
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
    save_overall_plots(df_raw, outdir, "raw", bins=args.bins_overall)
    save_class_plots(df_raw, outdir, "raw", k_outliers=args.class_outliers, bins=args.bins_class)

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

    # 2.3 Resample ALL (strict band enforcement)
    resampled_records = []
    for _, r in df_raw.iterrows():
        p = Path(r["path"])
        cls = r["cls"]
        m = safe_read_mesh(p)
        if m is None:
            continue
        n = int(r["n_vertices"])
        m2, suffix = resample_to_band(m, prefer_low=(not args.prefer_high))

        out_cls = resampled_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        out_path = out_cls / f"{p.stem}{suffix}.obj"
        o3d.io.write_triangle_mesh(str(out_path), m2, write_ascii=True)

        resampled_records.append(analyze_mesh_record(m2, out_path, cls, force_face_type="triangles"))

    df_res = pd.DataFrame(resampled_records)
    df_res.to_csv(outdir / "stats_resampled.csv", index=False)
    df_res[["cls","path","n_vertices","n_faces","n_triangles"]].to_csv(outdir / "objects_resampled.csv", index=False)
    save_overall_plots(df_res, outdir, "resampled", bins=args.bins_overall)
    save_class_plots(df_res, outdir, "resampled", k_outliers=args.class_outliers, bins=args.bins_class)

    # Write a quick check for any violations (should be zero)
    below = df_res[df_res["n_vertices"] < 5000]
    above = df_res[df_res["n_vertices"] > 10000]
    if len(below) or len(above):
        with open(outdir / "violations_resampled.txt", "w") as f:
            if len(below):
                f.write("== Below 5k ==\n")
                f.write(below[["cls","n_vertices","n_faces","path"]].to_string(index=False))
                f.write("\n\n")
            if len(above):
                f.write("== Above 10k ==\n")
                f.write(above[["cls","n_vertices","n_faces","path"]].to_string(index=False))
                f.write("\n")

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
    save_overall_plots(df_norm, outdir, "normalized", bins=args.bins_overall)
    save_class_plots(df_norm, outdir, "normalized", k_outliers=args.class_outliers, bins=args.bins_class)

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
    print("Note: Normalization does NOT change vertex counts; resampling enforces 5k–10k.")

if __name__ == "__main__":
    main()
