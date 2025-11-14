#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, time, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# optional libs
try:
    import mplcursors
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except Exception:
    HAS_ANNOY = False


# ------------------------------------------------------------------
# small utilities
# ------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_guess_class_from_row(row: pd.Series) -> str:
    """
    Heuristic in case class labels are missing:
    Try obvious class columns first; otherwise guess from filename stem.
    """
    cand_cols = [
        "class", "Class", "class_name", "Class_Name", "label", "Label", "category",
        "name", "Name", "model", "Model", "object", "Object",
        "file", "File", "filepath", "path", "Path"
    ]
    for col in cand_cols:
        if col in row and isinstance(row[col], str):
            # direct class column
            if col.lower() in ("class", "class_name"):
                return row[col]
            # guess from filename
            stem = Path(row[col]).stem
            m = re.match(r"([A-Za-z]+)", stem)
            if m:
                return m.group(1)
    return "Unknown"


def list_csvs(root: Path):
    if not root or not root.exists():
        return []
    return list(root.rglob("*.csv"))


def explain_csv(path: Path):
    """
    Quick probe: can we read it, and does it look numeric enough to be feature data?
    """
    try:
        df = pd.read_csv(path, nrows=3)
    except Exception as e:
        return False, f"unreadable ({e})"
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numerics) >= 5:
        return True, f"OK ({len(numerics)} numeric)"
    return False, "too few numeric columns (<5)"


def pick_biggest(csvs):
    """
    Among candidate CSVs, pick the one that likely contains the full feature DB:
    score = (#rows * #numeric_columns)
    """
    best, best_score = None, -1
    for p in csvs:
        try:
            df = pd.read_csv(p)
            numeric_cnt = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
            score = len(df) * max(1, numeric_cnt)
            if score > best_score:
                best, best_score = p, score
        except Exception:
            continue
    return best


def load_any_csv(csv_path: Path, root_label: str):
    """
    Load a features CSV and return:
      df_out: DataFrame with ['file','class',<feature columns...>]
      numeric_cols: list of feature column names
      source_str: human-readable description
    """
    df = pd.read_csv(csv_path)

    # numeric feature columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise SystemExit(f"[ERROR] {csv_path} has no numeric feature columns.")

    # find a file/id column or synthesize one
    file_col = None
    for c in ["file", "File", "filepath", "path", "Path", "model", "name", "Name", "Model"]:
        if c in df.columns:
            file_col = c
            break
    if file_col is None:
        df["file"] = [f"{root_label}/row_{i}" for i in range(len(df))]
        file_col = "file"

    # find a class column or synthesize one
    if "class" in df.columns:
        class_series = df["class"].astype(str)
    elif "class_name" in df.columns:
        class_series = df["class_name"].astype(str)
    else:
        class_series = df.apply(safe_guess_class_from_row, axis=1).astype(str)
    df["class"] = class_series

    keep_cols = ["file", "class"] + numeric_cols
    df_out = df[keep_cols].copy()
    df_out["file"] = df_out["file"].astype(str)
    df_out["class"] = df_out["class"].astype(str)

    return df_out, numeric_cols, f"{root_label}: {csv_path}"


# ------------------------------------------------------------------
# nearest neighbor engines
# ------------------------------------------------------------------

class ExactEngine:
    """
    Wrapper over sklearn NearestNeighbors for exact euclidean search.
    """
    def __init__(self, X):
        self.nn = NearestNeighbors(metric="euclidean", n_jobs=-1).fit(X)
        self.X = X

    def knn(self, x, k):
        d, i = self.nn.kneighbors(x.reshape(1, -1), n_neighbors=k, return_distance=True)
        return d[0], i[0]

    def knn_all(self, k):
        """
        Vectorized KNN for *all* items.
        Returns:
            dists: (N, k)
            idxs:  (N, k)
        """
        d, i = self.nn.kneighbors(n_neighbors=k, return_distance=True)
        return d, i

    def rnn(self, x, r):
        d, i = self.nn.radius_neighbors(x.reshape(1, -1), radius=r, return_distance=True)
        return d[0], i[0]


class AnnEngine:
    """
    Wrapper around Annoy for approximate NN.
    """
    def __init__(self, X, n_trees=50):
        if not HAS_ANNOY:
            raise RuntimeError("Annoy missing. pip install annoy")
        self.X = X.astype(np.float32)
        self.idx = AnnoyIndex(self.X.shape[1], "euclidean")
        for j, v in enumerate(self.X):
            self.idx.add_item(j, v)
        self.idx.build(n_trees)

    def knn(self, x, k, search_k=-1):
        i, d = self.idx.get_nns_by_vector(
            x.astype(np.float32), k, search_k=search_k, include_distances=True
        )
        return np.array(d), np.array(i, dtype=int)

    def knn_all(self, k):
        """
        Annoy has no vectorized kneighbors, so we loop.
        Returns arrays shaped (N,k) just like ExactEngine.knn_all.
        """
        all_d = []
        all_i = []
        for v in self.X:
            i, d = self.idx.get_nns_by_vector(
                v.astype(np.float32), k, search_k=-1, include_distances=True
            )
            all_d.append(d)
            all_i.append(i)
        return np.array(all_d), np.array(all_i)


# ------------------------------------------------------------------
# plotting helpers
# ------------------------------------------------------------------

def plot_benchmark(k_vals, exact_us, ann_us, out_png: Path):
    plt.figure(figsize=(9, 6))
    plt.yscale("log")

    plt.plot(k_vals, exact_us, label="Exact k-NN query time", linewidth=3)
    if not np.all(np.isnan(ann_us)):
        plt.plot(k_vals, ann_us, label="ANN query time", linewidth=3)

    plt.xlabel("Query size (K)", fontsize=14)
    plt.ylabel("Average query time [μs] (log scale)", fontsize=14)
    plt.title("Average top-K query time over all shapes", fontsize=18, weight="bold")
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(alpha=0.3, which="both", linestyle="--")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_tsne(Y, classes, out_png: Path):
    plt.figure(figsize=(16, 8))
    unique_classes = pd.unique(classes)

    for cls in unique_classes:
        mask = (classes == cls)
        plt.scatter(
            Y[mask, 0],
            Y[mask, 1],
            s=20,
            alpha=0.8,
            label=str(cls),
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    # Limit legend to max 3 columns
    n_items = len(labels)
    ncol = min(n_items, 3)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=1.0,
        fontsize=9,
        frameon=False,
        title="Class",
        title_fontsize=11,
        ncol=ncol,
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1.0])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def interactive_tsne(Y, meta):
    if not HAS_MPL:
        print("[INFO] mplcursors not installed; skipping interactive viewer.")
        return

    import mplcursors as _mplcursors  # runtime import

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Y[:, 0], Y[:, 1], s=15, alpha=0.7)
    ax.set_title("t-SNE (hover for details)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    cur = _mplcursors.cursor(sc, hover=True)

    @cur.connect("add")
    def on_add(sel):
        i = int(sel.index)
        row = meta.iloc[i]
        sel.annotation.set(
            text=f"class: {row['class']}\nfile: {Path(row['file']).name}"
        )

    plt.show()


# ------------------------------------------------------------------
# extended evaluation utilities (Step 6)
# ------------------------------------------------------------------

def compute_metrics_for_engine_at_Ks(df, engine, K_list, label_for_engine):
    """
    For a given engine (ExactEngine or AnnEngine), compute retrieval metrics for multiple K.
    Treat each query's own class as "relevant".
    Returns a list of dicts with precision, recall, TP, FP, FN, TN, sensitivity, specificity.
    """
    classes = df["class"].to_numpy()
    N = len(df)

    Kmax = max(K_list)
    d_all, idx_all = engine.knn_all(Kmax + 1)  # +1 to skip self

    results = []

    for K in K_list:
        TP_all = np.zeros(N, dtype=np.float32)
        FP_all = np.zeros(N, dtype=np.float32)
        FN_all = np.zeros(N, dtype=np.float32)
        TN_all = np.zeros(N, dtype=np.float32)
        prec_all = np.zeros(N, dtype=np.float32)
        rec_all  = np.zeros(N, dtype=np.float32)

        for i in range(N):
            c_true = classes[i]

            # total relevant shapes = same class (except self)
            same_class_mask = (classes == c_true)
            total_same_class = np.sum(same_class_mask) - 1  # don't count self

            # the top-K neighbors for this query
            neigh_idx = idx_all[i, 1:K+1]  # skip self at [0]
            neigh_cls = classes[neigh_idx]

            TP = np.sum(neigh_cls == c_true)
            FP = np.sum(neigh_cls != c_true)
            FN = max(total_same_class, 0) - TP
            TN = (N - 1) - TP - FP - FN  # all remaining others

            TP_all[i] = TP
            FP_all[i] = FP
            FN_all[i] = FN
            TN_all[i] = TN

            # precision@K
            prec_all[i] = TP / float(K) if K > 0 else np.nan

            # recall@K / sensitivity
            if total_same_class > 0:
                rec_all[i] = TP / float(total_same_class)
            else:
                rec_all[i] = np.nan

        # aggregate across queries
        TP_sum = np.nansum(TP_all)
        FP_sum = np.nansum(FP_all)
        FN_sum = np.nansum(FN_all)
        TN_sum = np.nansum(TN_all)

        precision_global = np.nanmean(prec_all)
        recall_global    = np.nanmean(rec_all)

        # Sensitivity (TPR) = TP / (TP + FN)
        sensitivity = TP_sum / float(TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else np.nan
        # Specificity (TNR) = TN / (TN + FP)
        specificity = TN_sum / float(TN_sum + FP_sum) if (TN_sum + FP_sum) > 0 else np.nan

        results.append({
            "engine": label_for_engine,
            "K": K,
            "precision": precision_global,
            "recall": recall_global,
            "TP": TP_sum,
            "FP": FP_sum,
            "FN": FN_sum,
            "TN": TN_sum,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })

    return results


def plot_confusion_bars(results_for_one_K, out_png: Path):
    """
    Bar plot for TP / FP / FN / TN for a single K (use e.g. last K in list).
    results_for_one_K should be a dict with TP/FP/FN/TN and 'engine' and 'K'.
    """
    labels = ["TP", "FP", "FN", "TN"]
    vals = [results_for_one_K["TP"],
            results_for_one_K["FP"],
            results_for_one_K["FN"],
            results_for_one_K["TN"]]

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, vals)
    plt.title(f"Retrieval outcomes at K={results_for_one_K['K']} ({results_for_one_K['engine']})",
              fontsize=14, weight="bold")
    plt.ylabel("Total count over all queries", fontsize=12)
    plt.xlabel("Outcome type", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for b in bars:
        y = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, y, f"{int(y)}",
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_precision_recall_vs_K(results_exact, results_ann, out_png: Path):
    """
    Line plot: Precision@K and Recall@K as K increases.
    Shows trade-off (recall goes up, precision down).
    """
    ex_sorted = sorted(results_exact, key=lambda r: r["K"])
    Ks = [r["K"] for r in ex_sorted]
    prec_e = [r["precision"] for r in ex_sorted]
    rec_e  = [r["recall"]    for r in ex_sorted]

    plt.figure(figsize=(8,5))

    plt.plot(Ks, prec_e, "-o", label="Precision (Exact)")
    plt.plot(Ks, rec_e,  "-o", label="Recall (Exact)")

    if results_ann:
        ann_sorted = sorted(results_ann, key=lambda r: r["K"])
        prec_a = [r["precision"] for r in ann_sorted]
        rec_a  = [r["recall"]    for r in ann_sorted]
        plt.plot(Ks, prec_a, "--o", label="Precision (ANN)")
        plt.plot(Ks, rec_a,  "--o", label="Recall (ANN)")

    plt.title("Precision@K and Recall@K vs K", fontsize=16, weight="bold")
    plt.xlabel("K (top-K neighbors returned)", fontsize=12)
    plt.ylabel("Score (0 to 1)", fontsize=12)
    plt.ylim(0,1)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roc_curves(results_exact, results_ann, out_png: Path):
    """
    ROC-style plot: Sensitivity vs Specificity for several K.
    We'll also draw a dashed diagonal "random baseline".
    """
    plt.figure(figsize=(6,6))

    ex_sorted = sorted(results_exact, key=lambda r: r["K"])
    sens_e = [r["sensitivity"] for r in ex_sorted]
    spec_e = [r["specificity"] for r in ex_sorted]
    plt.plot(sens_e, spec_e, "-o", label="Exact k-NN")

    if results_ann:
        ann_sorted = sorted(results_ann, key=lambda r: r["K"])
        sens_a = [r["sensitivity"] for r in ann_sorted]
        spec_a = [r["specificity"] for r in ann_sorted]
        plt.plot(sens_a, spec_a, "-o", label="ANN (Approx)")

    # random baseline: specificity ~ 1 - sensitivity
    xs = np.linspace(0,1,50)
    ys = 1 - xs
    plt.plot(xs, ys, "--", color="gray", label="Random baseline")

    plt.title("Sensitivity vs Specificity", fontsize=16, weight="bold")
    plt.xlabel("Sensitivity (True Positive Rate / Recall)", fontsize=12)
    plt.ylabel("Specificity (True Negative Rate)", fontsize=12)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def evaluate_precision_recall_at_k(df, Xs, engine, k_eval: int, out_dir: Path):
    """
    Legacy single-K eval: per-shape precision and recall,
    per-class averages, silhouette score.
    """
    print(f"[EVAL] Computing precision@{k_eval} / recall@{k_eval} for all shapes...")

    d_all, idx_all = engine.knn_all(k_eval + 1)
    classes = df["class"].to_numpy()
    N = len(df)

    precisions = np.zeros(N, dtype=np.float32)
    recalls    = np.zeros(N, dtype=np.float32)

    for i in range(N):
        c_true = classes[i]
        neigh_idx = idx_all[i, 1:k_eval+1]
        neigh_cls = classes[neigh_idx]

        correct = np.sum(neigh_cls == c_true)

        precisions[i] = correct / float(k_eval)

        total_same_class = np.sum(classes == c_true) - 1
        if total_same_class > 0:
            recalls[i] = correct / float(total_same_class)
        else:
            recalls[i] = np.nan

    df_eval = pd.DataFrame({
        "file": df["file"],
        "class": df["class"],
        f"precision_at_{k_eval}": precisions,
        f"recall_at_{k_eval}":    recalls,
    })

    per_class = df_eval.groupby("class")[
        [f"precision_at_{k_eval}", f"recall_at_{k_eval}"]
    ].agg(["mean", "count"])
    per_class.columns = [f"{m}_{stat}" for m, stat in per_class.columns]
    per_class = per_class.sort_values(f"precision_at_{k_eval}_mean", ascending=False)

    global_precision = np.nanmean(precisions)
    global_recall    = np.nanmean(recalls)

    print(f"[EVAL] Global mean precision@{k_eval}: {global_precision:.3f}")
    print(f"[EVAL] Global mean recall@{k_eval}:    {global_recall:.3f}")
    print("[EVAL] Per-class (sorted by precision):")
    print(per_class)

    try:
        sil = silhouette_score(Xs, classes)
        print(f"[EVAL] Silhouette score (feature separability): {sil:.3f}")
    except Exception as e:
        sil = None
        print(f"[EVAL] Silhouette score skipped ({e})")

    ensure_dir(out_dir)
    df_eval.to_csv(out_dir / f"eval_per_shape_k{k_eval}.csv", index=False)
    per_class.to_csv(out_dir / f"eval_per_class_k{k_eval}.csv")

    with open(out_dir / f"eval_summary_k{k_eval}.txt", "w") as fh:
        fh.write(f"global_precision_at_{k_eval}: {global_precision:.6f}\n")
        fh.write(f"global_recall_at_{k_eval}:    {global_recall:.6f}\n")
        if sil is not None:
            fh.write(f"silhouette_score:           {sil:.6f}\n")

    return global_precision, global_recall, per_class


# ------------------------------------------------------------------
# main pipeline
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--features_csv", type=str, default=None,
                    help='Direct path to your features CSV (quote it if it has spaces).')
    ap.add_argument("--mr_root", type=str, default="MR System")
    ap.add_argument("--step3_root", type=str, default="step3_results")

    # demo query params
    ap.add_argument("--k", type=int, default=10,
                    help="k for demo single-query neighbor printout")
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--query_idx", type=int, default=0)

    # benchmarking params
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench_kmax", type=int, default=30)

    # tsne params
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=750)
    ap.add_argument("--tsne_perplexities", type=str, default=None,
                    help="Comma-separated list of perplexities for t-SNE, e.g. '30,40,50'. If set, overrides --perplexity.")

    # evaluation params (Step 6)
    ap.add_argument("--eval", action="store_true",
                    help="Run evaluation/plots on retrieval quality")
    ap.add_argument("--eval_k", type=int, default=10,
                    help="K for per-class precision/recall table (legacy)")
    ap.add_argument("--eval_k_list", type=str, default=None,
                    help="Comma-separated list of K values for per-class precision/recall tables, e.g. '5,10,20'")
    ap.add_argument("--k_sweep", type=str, default="1,3,5,10,20,30",
                    help="Comma-separated list of K values for curves/ROC, e.g. '1,3,5,10,20,30'")

    ap.add_argument("--out", type=str, default="step5_results")

    args = ap.parse_args()

    out_dir = Path(args.out)
    graphs_dir = out_dir / "graphs"
    ensure_dir(graphs_dir)

    # --------------------------------------------------
    # load data
    # --------------------------------------------------
    if args.features_csv:
        csv_path = Path(args.features_csv)
        if not csv_path.exists():
            sys.exit(f"[ERROR] Not found: {csv_path}")
        df, feat_cols, source = load_any_csv(csv_path, "manual")
    else:
        mr_root = Path(args.mr_root)
        step3_root = Path(args.step3_root)

        def scan_and_choose(root):
            cands = []
            for p in list_csvs(root):
                ok, why = explain_csv(p)
                print(f"  - {p}: {why}")
                if ok:
                    cands.append(p)
            return cands

        print(f"[SCAN] Searching CSVs under: {mr_root} …")
        mr_cands = scan_and_choose(mr_root)

        print(f"[SCAN] Searching CSVs under: {step3_root} …")
        s3_cands = scan_and_choose(step3_root)

        all_cands = mr_cands + s3_cands
        if not all_cands:
            sys.exit('No usable features CSV found. Pass one explicitly with --features_csv "<path>"')

        best_csv = pick_biggest(all_cands)
        root_label = "MR System" if best_csv and (mr_root in best_csv.parents) else "step3_results"
        df, feat_cols, source = load_any_csv(best_csv, root_label)

    print(f"[INFO] Using: {source}")
    print(f"[INFO] Shapes: {len(df)} | Feature dims: {len(feat_cols)}")

    # print class distribution
    class_counts = df["class"].value_counts()
    print("[INFO] Class distribution:")
    for cname, cnt in class_counts.items():
        print(f"   {cname}: {cnt}")

    # --------------------------------------------------
    # prepare feature matrix
    # --------------------------------------------------
    # 1) fill missing values in the feature columns
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].mean())

    # 2) now convert to numpy
    X = df[feat_cols].to_numpy().astype(np.float32)

    # 3) scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # --------------------------------------------------
    # build engines
    # --------------------------------------------------
    exact_engine = ExactEngine(Xs)
    ann_engine = AnnEngine(Xs, n_trees=50) if HAS_ANNOY else None
    if not ann_engine:
        print("[WARN] Annoy not available; skipping ANN timings/queries.")

    # --------------------------------------------------
    # example single query
    # --------------------------------------------------
    qidx = max(0, min(len(df) - 1, int(args.query_idx)))
    qvec = Xs[qidx]

    d_exact, i_exact = exact_engine.knn(qvec, args.k)
    print(f"[INFO] Exact KNN (k={args.k}):")
    for rank, (dist, idx_neighbor) in enumerate(zip(d_exact, i_exact), 1):
        print(f"  {rank:>2}. d={dist:.4f}  cls={df.iloc[idx_neighbor]['class']}  file={df.iloc[idx_neighbor]['file']}")

    if ann_engine is not None:
        d_ann, i_ann = ann_engine.knn(qvec, args.k)
        print(f"[INFO] ANN  KNN (k={args.k}):")
        for rank, (dist, idx_neighbor) in enumerate(zip(d_ann, i_ann), 1):
            print(f"  {rank:>2}. d~={dist:.4f} cls={df.iloc[idx_neighbor]['class']}  file={df.iloc[idx_neighbor]['file']}")

    if args.radius is not None:
        d_rad, i_rad = exact_engine.rnn(qvec, args.radius)
        print(f"[INFO] Exact R-NN (R={args.radius}) → {len(i_rad)} neighbors")
        for rank, (dist, idx_neighbor) in enumerate(sorted(zip(d_rad, i_rad), key=lambda t: t[0])[:20], 1):
            print(f"  {rank:>2}. d={dist:.4f}  cls={df.iloc[idx_neighbor]['class']}  file={df.iloc[idx_neighbor]['file']}")

    # --------------------------------------------------
    # benchmark scalability (Step 5 timing plot)
    # --------------------------------------------------
    if args.bench:
        ks = list(range(3, args.bench_kmax + 1))
        exact_us, ann_us = [], []
        sample_idx = np.linspace(0, len(df) - 1, num=min(200, len(df)), dtype=int)
        reps = max(1, min(5, len(df) // 1000 or 1))

        for k in ks:
            # exact timing
            t0 = time.perf_counter()
            for _ in range(reps):
                for s in sample_idx:
                    exact_engine.knn(Xs[s], k)
            t1 = time.perf_counter()
            exact_us.append(((t1 - t0) / (len(sample_idx) * reps)) * 1e6)

            # approx timing
            if ann_engine is not None:
                t0 = time.perf_counter()
                for _ in range(reps):
                    for s in sample_idx:
                        ann_engine.knn(Xs[s], k)
                t1 = time.perf_counter()
                ann_us.append(((t1 - t0) / (len(sample_idx) * reps)) * 1e6)
            else:
                ann_us.append(np.nan)

        bench_png = graphs_dir / "benchmark_knn_time.png"
        plot_benchmark(ks, exact_us, np.array(ann_us, float), bench_png)
        print(f"[OK] Saved benchmark → {bench_png}")

    # --------------------------------------------------
    # t-SNE visualization (Step 5 visual clustering)
    # --------------------------------------------------
    if args.tsne:
        # decide which perplexities to run
        if args.tsne_perplexities:
            perp_vals = [float(p.strip()) for p in args.tsne_perplexities.split(",") if p.strip() != ""]
        else:
            perp_vals = [float(args.perplexity)]

        first_tsne_done = False

        for perp in perp_vals:
            print(f"[INFO] Running t-SNE… (perplexity={perp})")
            tsne_kwargs = dict(
                n_components=2,
                perplexity=perp,
                init="pca",
                random_state=42,
            )

            # sklearn version compatibility for n_iter / learning_rate
            try:
                tsne = TSNE(**tsne_kwargs, learning_rate="auto", n_iter=args.iters)
                Y = tsne.fit_transform(Xs)
            except TypeError:
                try:
                    tsne = TSNE(**tsne_kwargs, learning_rate=200.0, n_iter=args.iters)
                    Y = tsne.fit_transform(Xs)
                except TypeError:
                    tsne = TSNE(**tsne_kwargs, learning_rate=200.0)
                    Y = tsne.fit_transform(Xs)

            tsne_png = graphs_dir / f"tsne_perp{int(perp)}.png"
            plot_tsne(Y, df["class"].values, tsne_png)
            print(f"[OK] Saved t-SNE → {tsne_png}")

            # show interactive only once (optional)
            if HAS_MPL and not first_tsne_done:
                print("[INFO] Showing interactive viewer… Close the window to finish.")
                interactive_tsne(Y, df[["file", "class"]])
                first_tsne_done = True

    # --------------------------------------------------
    # Evaluation (Step 6)
    # --------------------------------------------------
    if args.eval:
        print("[INFO] Running retrieval quality evaluation...")

        # 6A. run per-class tables for one or more eval K's
        if args.eval_k_list:
            eval_ks = [int(x.strip()) for x in args.eval_k_list.split(",") if x.strip() != ""]
        else:
            eval_ks = [int(args.eval_k)]

        for ek in eval_ks:
            gp, gr, pc = evaluate_precision_recall_at_k(
                df=df,
                Xs=Xs,
                engine=exact_engine,
                k_eval=ek,
                out_dir=out_dir,
            )
            print(f"[INFO] Global precision@{ek}: {gp:.3f}")
            print(f"[INFO] Global recall@{ek}:    {gr:.3f}")

        # 6B. Sweep K values to build curves and ROC-style plot (do once)
        K_list = [int(x) for x in args.k_sweep.split(",") if x.strip() != ""]
        exact_results = compute_metrics_for_engine_at_Ks(
            df=df,
            engine=exact_engine,
            K_list=K_list,
            label_for_engine="Exact k-NN"
        )
        if ann_engine is not None:
            ann_results = compute_metrics_for_engine_at_Ks(
                df=df,
                engine=ann_engine,
                K_list=K_list,
                label_for_engine="ANN"
            )
        else:
            ann_results = []

        # save numeric summary for report
        summary_rows = exact_results + ann_results
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "eval_summary_allK.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Saved per-K summary → {summary_csv}")

        # plot Precision/Recall vs K
        pr_png = graphs_dir / "precision_recall_vs_K.png"
        plot_precision_recall_vs_K(exact_results, ann_results, pr_png)
        print(f"[OK] Saved Precision/Recall vs K plot → {pr_png}")

        # plot ROC-style (Sensitivity vs Specificity)
        roc_png = graphs_dir / "roc_curve.png"
        plot_roc_curves(exact_results, ann_results, roc_png)
        print(f"[OK] Saved ROC-style curve → {roc_png}")

        # plot TP/FP/FN/TN bars for the largest K in sweep
        best_result = sorted(exact_results, key=lambda r: r["K"])[-1]
        conf_png = graphs_dir / "confusion_bars.png"
        plot_confusion_bars(best_result, conf_png)
        print(f"[OK] Saved confusion totals plot → {conf_png}")

    # --------------------------------------------------
    # Save run configuration
    # --------------------------------------------------
    ensure_dir(out_dir)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[DONE] Artifacts → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
