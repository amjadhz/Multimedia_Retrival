import streamlit as st
import sys
import tempfile
from pathlib import Path
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# basic setup
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MR_DIR = BASE_DIR / "MR system"
sys.path.append(str(MR_DIR))

from step4 import mesh_querying   # your existing function


# ---------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------
st.set_page_config(page_title="3D Mesh Similarity Viewer", layout="wide")

# custom css
st.markdown(
    """
    <style>
    /* overall page */
    .main {
        background: #0f1116;
    }
    /* header box */
    .app-header {
        background: radial-gradient(circle at top, #1f2937, #0f1116 70%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 1.2rem 1.5rem 0.8rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.55);
    }
    /* author chips */
    .author-row {
        margin-top: 0.7rem;
        display: flex;
        gap: 0.5rem;
    }
    .author-pill {
        background: rgba(103,232,249,0.12);
        border: 1px solid rgba(103,232,249,0.3);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        color: #e2f5ff;
        display: flex;
        gap: 0.3rem;
        align-items: center;
    }
    .author-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #67e8f9;
    }
    /* section titles */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        margin: 0.8rem 0 0.5rem 0;
        color: #ffffff;
    }
    .badge-icon {
        width: 20px;
        height: 20px;
    }
    /* result cards (when missing) */
    .missing-card {
        background: rgba(255, 55, 95, 0.12);
        border: 1px solid rgba(255, 55, 95, 0.28);
        border-radius: 14px;
        padding: 0.5rem 0.7rem;
        font-size: 0.7rem;
        color: #ffe6ea;
        min-height: 70px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is None or len(mesh.triangles) == 0:
        raise ValueError(f"Could not load a valid triangle mesh from: {path}")
    return mesh


def mesh_to_plotly(mesh: o3d.geometry.TriangleMesh,
                   color="lightgrey",
                   width=900,
                   height=500) -> go.Figure:
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    fig = go.Figure()
    if verts.size == 0 or tris.size == 0:
        fig.update_layout(title="Empty mesh")
        return fig

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]

    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=1.0,
            color=color,
        )
    )
    fig.update_layout(
        scene_aspectmode="data",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        margin=dict(r=0, l=0, b=0, t=30),
    )
    return fig


def clean_filename(name: str) -> str:
    """Keep only the file name (CSV may contain full path)."""
    name = name.replace("\\", "/")
    return name.split("/")[-1]


def resolve_in_class_folder(base: Path, cls_name: str, filename: str) -> Path | None:
    """
    Your layout: MR system/data/<class>/<file>.obj
    query gives: class, 'D00078_simplified_norm.obj'
    actual file might be: 'D00078_refined_simplified_norm.obj'
    → so match by prefix in that class folder.
    """
    class_dir = base / cls_name
    if not class_dir.exists():
        return None

    # 1) exact
    cand = class_dir / filename
    if cand.exists():
        return cand

    # 2) prefix match
    prefix = filename.split("_")[0]   # e.g. D00078
    if prefix:
        for p in class_dir.glob(f"{prefix}*.obj"):
            return p
        for p in class_dir.glob(f"{prefix}*"):
            return p

    return None


# ---------------------------------------------------------------------
# layout top
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">3D Mesh Similarity Viewer</div>
        <div class="app-subtitle">Upload a 3D mesh and browse the most similar shapes from your dataset.</div>
        <div class="author-row">
            <div class="author-pill"><div class="author-dot"></div>Amjad Hwidy</div>
            <div class="author-pill"><div class="author-dot"></div>Binxu Jiang</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# sidebar controls
# ---------------------------------------------------------------------
uploaded = st.file_uploader("Upload OBJ / STL / PLY", type=["obj", "stl", "ply"])

default_csv = MR_DIR / "database" / "all_features.csv"
default_norm = MR_DIR / "database" / "normalization_params.csv"
default_mesh_base = "data"

with st.sidebar:
    st.header("Settings")
    K = st.number_input("Top-K results", 1, 30, 10)
    csv_path = st.text_input("Features CSV", str(default_csv))
    norm_path = st.text_input("Normalization params", str(default_norm))
    mesh_base = st.text_input("Meshes folder", str(default_mesh_base))
    st.markdown("---")
    st.caption("3D Mesh Similarity Viewer\nby **Amjad Hwidy** & **Binxu Jiang**")

# ---------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------
if uploaded:
    # save temp
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # uploaded mesh section
    st.markdown('<div class="section-title">📦 Uploaded mesh</div>', unsafe_allow_html=True)
    try:
        qmesh = load_mesh(tmp_path)
        qmesh.paint_uniform_color([0.82, 0.82, 0.82])
        fig = mesh_to_plotly(qmesh, width=950, height=520)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render uploaded mesh: {e}")

    # similar meshes section
    st.markdown('<div class="section-title">🧩 Similar meshes</div>', unsafe_allow_html=True)
    try:
        classes, results = mesh_querying(
            model_file_name=tmp_path,
            csv_path=csv_path,
            normalization_params_path=norm_path,
            K=K,
        )
    except Exception as e:
        st.error(f"Error running mesh_querying: {e}")
        st.stop()

    cols_per_row = 4
    cols = st.columns(cols_per_row)

    for i, ((cls_name, rel_name), dist) in enumerate(results):
        col = cols[i % cols_per_row]

        filename = clean_filename(rel_name)
        mesh_path = resolve_in_class_folder(Path(mesh_base), cls_name, filename)

        if mesh_path is None or not mesh_path.exists():
            col.markdown(
                f"<div class='missing-card'>Missing:<br>{Path(mesh_base) / cls_name / filename}</div>",
                unsafe_allow_html=True,
            )
            continue

        try:
            m = load_mesh(str(mesh_path))
            m.paint_uniform_color([0.9, 0.9, 0.9])
            fig_sim = mesh_to_plotly(m, width=360, height=270)
            col.plotly_chart(fig_sim, use_container_width=True)
            col.caption(f"{i+1}. {cls_name} • {mesh_path.name} • Dist: {dist:.4f}")
        except Exception as e:
            col.error(f"Render error: {e}")

    with st.expander("Classes / scores"):
        st.write(classes)

else:
    st.info("👆 Upload a mesh to get similar results.")
