from __future__ import annotations
import pandas as pd
import numpy as np
import open3d as o3d
from mesh_normalization import mesh_normalize_for_new
from f_e import compute_single_descriptors, compute_distribution_descriptors
from ANN import ann
import trimesh
from stopwatch import Stopwatch
from data_analysis import resample
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os


def mesh_querying(model_file_name, csv_path, distance_stats_path, K,
                  stopwatch: Stopwatch | None = None):
    """
    Query K most similar models using distance normalization

    Parameters:
        model_file_name: Query model path
        csv_path: Feature database path (all_features.csv)
        distance_stats_path: Distance statistics CSV path
        K: Number of most similar models to return
        stopwatch: Performance monitoring object
    """
    # Load distance statistics
    distance_stats_df = pd.read_csv(distance_stats_path)

    # Convert to dictionary for easy lookup
    distance_stats = {}
    for _, row in distance_stats_df.iterrows():
        distance_stats[row['feature']] = {
            'mean': row['mean'],
            'std': row['std']
        }

    data = pd.read_csv(csv_path)

    # Process query model
    descriptors = process_new_model(model_file_name)

    # Single-value feature names (7 features)
    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    if stopwatch is not None:
        stopwatch.start()

    # Extract database features
    single_value_features = data.iloc[:, 2:9].values
    histogram_features = [
        data.iloc[:, 9:73].values,
        data.iloc[:, 73:137].values,
        data.iloc[:, 137:201].values,
        data.iloc[:, 201:265].values,
        data.iloc[:, 265:329].values
    ]

    # Separate query features
    query_single_values = np.array(descriptors[:7])
    query_histograms = [
        np.array(descriptors[7:71]),
        np.array(descriptors[71:135]),
        np.array(descriptors[135:199]),
        np.array(descriptors[199:263]),
        np.array(descriptors[263:327])
    ]

    # Initialize total distances
    n_samples = len(data)
    total_distances = np.zeros(n_samples)

    # 1. Calculate and normalize single-value feature distances
    for i, feature_name in enumerate(single_value_names):
        raw_distances = np.abs(single_value_features[:, i] - query_single_values[i])
        mean_dist = distance_stats[feature_name]['mean']
        std_dist = distance_stats[feature_name]['std']

        if std_dist > 0:
            z_distances = (raw_distances - mean_dist) / std_dist
        else:
            z_distances = raw_distances

        total_distances += z_distances

    # 2. Calculate and normalize histogram distances
    histogram_names = ['A3', 'D1', 'D2', 'D3', 'D4']

    for i, hist_name in enumerate(histogram_names):
        raw_distances = []
        for j in range(n_samples):
            similarity = cosine_similarity(
                [query_histograms[i]],
                [histogram_features[i][j]]
            )[0][0]
            distance = 1 - similarity
            raw_distances.append(distance)

        raw_distances = np.array(raw_distances)
        mean_dist = distance_stats[hist_name]['mean']
        std_dist = distance_stats[hist_name]['std']

        if std_dist > 0:
            z_distances = (raw_distances - mean_dist) / std_dist
        else:
            z_distances = raw_distances

        total_distances += z_distances

    # 3. Find K most similar models
    closest_indices = np.argsort(total_distances)[:K]
    closest_models = data.iloc[closest_indices][['class_name', 'file_name']].values
    closest_distances = total_distances[closest_indices]

    if stopwatch is not None:
        stopwatch.stop()
        stopwatch.record_time()

    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))


def process_new_model(input_mesh_path):
    """
    Process new model and extract features

    Note: Returned feature order must match database:
    - First 7: Single-value features (unnormalized raw values)
    - Last 320: 5 histogram features (64 bins each, normalized)
    """
    try:
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            temp_path = tmp.name

        # First read mesh
        input_mesh = o3d.io.read_triangle_mesh(input_mesh_path)

        # Resample - pass mesh object instead of path
        resampled_mesh = resample(input_mesh)

        # Save resampled mesh
        o3d.io.write_triangle_mesh(temp_path, resampled_mesh)

        # Normalize
        normalized_mesh = mesh_normalize_for_new(resampled_mesh)
        normalized_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(temp_path, normalized_mesh)

        # Extract features
        normalized_mesh = trimesh.load(temp_path)

        # Single-value features (returns dictionary)
        single_features = compute_single_descriptors(normalized_mesh)

        # Histogram features (returns dictionary)
        dist_features = compute_distribution_descriptors(
            normalized_mesh,
            n_samples=150000,
            n_bins=64
        )

        # Clean up temporary file
        os.unlink(temp_path)

        # Combine into list (in CSV column order)
        descriptors = []

        # Add single-value features (7 features)
        single_value_names = [
            'surface_area', 'volume', 'compactness',
            'rectangularity', 'diameter', 'convexity', 'eccentricity'
        ]
        for name in single_value_names:
            value = single_features[name]
            descriptors.append(0.0 if np.isnan(value) else value)

        # Add histogram features (5 × 64 = 320 features)
        for hist_name in ['A3', 'D1', 'D2', 'D3', 'D4']:
            for i in range(64):
                bin_name = f'{hist_name}_bin{i}'
                descriptors.append(dist_features[bin_name])

        return descriptors

    except Exception as e:
        print(f"Error processing model {input_mesh_path}: {e}")
        raise


def fast_query(input_mesh_path, descriptors_path,
               distance_stats_path, ann_index, K,
               stopwatch: Stopwatch | None = None):
    """
    Fast query using ANN (with distance normalization)
    """
    distance_stats_df = pd.read_csv(distance_stats_path)
    distance_stats = {}
    for _, row in distance_stats_df.iterrows():
        distance_stats[row['feature']] = {
            'mean': row['mean'],
            'std': row['std']
        }

    descriptors = process_new_model(input_mesh_path)
    db_descriptors = pd.read_csv(descriptors_path)

    if stopwatch is not None:
        stopwatch.start()

    K_candidates = min(K * 3, len(db_descriptors))
    indices, raw_distances = ann(ann_index, descriptors, K_candidates)

    candidate_indices = indices[0, :]
    candidate_features = db_descriptors.iloc[candidate_indices]

    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    single_value_features = candidate_features.iloc[:, 2:9].values
    histogram_features = [
        candidate_features.iloc[:, 9:73].values,
        candidate_features.iloc[:, 73:137].values,
        candidate_features.iloc[:, 137:201].values,
        candidate_features.iloc[:, 201:265].values,
        candidate_features.iloc[:, 265:329].values
    ]

    query_single_values = np.array(descriptors[:7])
    query_histograms = [
        np.array(descriptors[7:71]),
        np.array(descriptors[71:135]),
        np.array(descriptors[135:199]),
        np.array(descriptors[199:263]),
        np.array(descriptors[263:327])
    ]

    n_candidates = len(candidate_features)
    total_distances = np.zeros(n_candidates)

    for i, feature_name in enumerate(single_value_names):
        raw_distances = np.abs(single_value_features[:, i] - query_single_values[i])
        mean_dist = distance_stats[feature_name]['mean']
        std_dist = distance_stats[feature_name]['std']
        z_distances = (raw_distances - mean_dist) / std_dist if std_dist > 0 else raw_distances
        total_distances += z_distances

    histogram_names = ['A3', 'D1', 'D2', 'D3', 'D4']
    for i, hist_name in enumerate(histogram_names):
        raw_distances = []
        for j in range(n_candidates):
            similarity = cosine_similarity(
                [query_histograms[i]],
                [histogram_features[i][j]]
            )[0][0]
            distance = 1 - similarity
            raw_distances.append(distance)
        raw_distances = np.array(raw_distances)

        mean_dist = distance_stats[hist_name]['mean']
        std_dist = distance_stats[hist_name]['std']
        z_distances = (raw_distances - mean_dist) / std_dist if std_dist > 0 else raw_distances
        total_distances += z_distances

    top_k_indices = np.argsort(total_distances)[:K]
    final_indices = candidate_indices[top_k_indices]
    final_distances = total_distances[top_k_indices]

    closest_models = db_descriptors.iloc[final_indices][['class_name', 'file_name']].values

    if stopwatch is not None:
        stopwatch.stop()
        stopwatch.record_time()

    return [model[0] for model in closest_models], list(zip(closest_models, final_distances))