import os
import pandas as pd
import numpy as np
from step4 import fast_query
from ANN import construct_kd_tree, save_ann_index, load_ann_index

INDEX_PATH = "database/ann_index.pkl"

# load index
if os.path.exists(INDEX_PATH):
    print("[INFO] Loading ANN index from disk...")
    ann_index = load_ann_index(INDEX_PATH)
else:
    print("[INFO] ANN index not found. Building new index...")

    # create index
    df = pd.read_csv("database/all_features.csv")
    feature_matrix = df.iloc[:, 2:].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(0.0)
    feature_matrix = feature_matrix.values.astype(np.float32)

    ann_index = construct_kd_tree(
        data=feature_matrix,
        distance_function="euclidean"
    )

    # save index
    save_ann_index(ann_index, INDEX_PATH)

# 2. run fast_query
classes, results = fast_query(
    input_mesh_path="../data/Apartment/D00156.obj",
    descriptors_path="database/all_features.csv",
    normalization_params_path="database/normalization_params.csv",
    ann_index=ann_index,
    K=10
)

print("\nFast-query results:")
print("=" * 70)
for i, ((cls, fname), dist) in enumerate(results, 1):
    print(f"{i:2}. {cls:15} | {fname:30} | distance: {dist:8.4f}")

print(f"\nTop-10 classes: {classes}")
