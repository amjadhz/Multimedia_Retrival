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
from data_analysis import normalize_mesh
from scipy.stats import wasserstein_distance

# Weight configuration
WEIGHTS = {
    'single_value': {
        'surface_area': 0.015,  # 3.5%
        'volume': 0.005,  # 3.5%
        'compactness': 0.010,  # 2.0%
        'rectangularity': 0.025,  # 3.5%
        'diameter': 0.025,  # 3.5%
        'convexity': 0.010,  # 2.0%
        'eccentricity': 0.010,  # 2.0%
        # Total single-value: 0.20 (20%)
    },
    'histogram': {
        'A3': 0.3,  # 30%
        'D1': 0.05,  # 5%
        'D2': 0.21,  # 15%
        'D3': 0.22,  # 15%
        'D4': 0.22,  # 15%
        # Total histogram: 0.80 (80%)
    }
}


def mesh_querying(model_file_name, csv_path, normalization_params_path, K,
                  stopwatch: Stopwatch | None = None,
                  weights=None):
    """
    Query K most similar models using normalized features

    Distance metrics:
    - Single-value features: Euclidean distance
    - Histogram features: EMD

    Parameters:
        model_file_name: Query model path
        csv_path: Feature database path (normalized all_features.csv)
        normalization_params_path: Normalization parameters CSV path
        K: Number of most similar models to return
        stopwatch: Performance monitoring object
    """
    # Use default weights
    if weights is None:
        weights = WEIGHTS

    # Load normalization parameters
    norm_params_df = pd.read_csv(normalization_params_path, index_col=0)

    # Load database
    data = pd.read_csv(csv_path)

    # Process query model
    descriptors = process_new_model(model_file_name, norm_params_df)

    # Single-value feature names
    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    if stopwatch is not None:
        stopwatch.start()

    # Extract database features
    single_value_features = data.iloc[:, 2:9].values
    histogram_features = [
        data.iloc[:, 9:109].values,  # A3 (100 bins)
        data.iloc[:, 109:209].values,  # D1 (100 bins)
        data.iloc[:, 209:309].values,  # D2 (100 bins)
        data.iloc[:, 309:409].values,  # D3 (100 bins)
        data.iloc[:, 409:509].values  # D4 (100 bins)
    ]

    # Separate query features
    query_single_values = np.array(descriptors[:7])
    query_histograms = [
        np.array(descriptors[7:107]),  # A3 (100 bins)
        np.array(descriptors[107:207]),  # D1 (100 bins)
        np.array(descriptors[207:307]),  # D2 (100 bins)
        np.array(descriptors[307:407]),  # D3 (100 bins)
        np.array(descriptors[407:507])  # D4 (100 bins)
    ]

    n_samples = len(data)
    total_distances = np.zeros(n_samples)

    # 1. Calculate single-value feature distances using EUCLIDEAN distance
    print("\nComputing single-value distances (Euclidean)...")

    # Compute Euclidean distance for each feature separately with weights
    for i, feature_name in enumerate(single_value_names):
        # Euclidean distance: |query - database|
        feature_distances = np.abs(single_value_features[:, i] - query_single_values[i])

        # Apply weight
        weight = weights['single_value'][feature_name]
        weighted_distances = weight * feature_distances

        total_distances += weighted_distances

        print(f"  {feature_name:20} : weight={weight:.4f}, "
              f"dist range=[{feature_distances.min():.4f}, {feature_distances.max():.4f}]")

    # 2. Calculate histogram distances using EMD
    print("\nComputing histogram distances (EMD)...")
    histogram_names = ['A3', 'D1', 'D2', 'D3', 'D4']

    for i, hist_name in enumerate(histogram_names):
        query_hist = query_histograms[i]
        db_hist = histogram_features[i]

        # EMD
        emd_distances = np.array([
            wasserstein_distance(query_hist, db_hist[j])
            for j in range(len(db_hist))
        ])

        # apply weighted
        weight = weights['histogram'][hist_name]
        total_distances += weight * emd_distances

        print(f"  {hist_name:20} : weight={weight:.4f}, "
              f"dist range=[{emd_distances.min():.4f}, {emd_distances.max():.4f}]")

    # 3. Find K most similar models
    closest_indices = np.argsort(total_distances)[:K]
    closest_models = data.iloc[closest_indices][['class_name', 'file_name']].values
    closest_distances = total_distances[closest_indices]

    if stopwatch is not None:
        stopwatch.stop()
        stopwatch.record_time()

    # Print summary
    print(f"\nQuery complete:")
    print(f"  Total weight: {sum(weights['single_value'].values()) + sum(weights['histogram'].values()):.2f}")
    print(f"  Single-value weight: {sum(weights['single_value'].values()):.2f} (20%)")
    print(f"  Histogram weight: {sum(weights['histogram'].values()):.2f} (80%)")

    return [model[0] for model in closest_models], list(zip(closest_models, closest_distances))


def process_new_model(input_mesh_path, norm_params_df=None):
    import copy

    try:
        # Step 1: Read mesh
        input_mesh = o3d.io.read_triangle_mesh(input_mesh_path)

        # Step 2: Resample
        resampled_mesh = resample(input_mesh)

        # Step 3: Simple normalization
        simple_normalized = normalize_mesh(resampled_mesh)

        # Step 4: Pose normalization
        normalized_mesh = mesh_normalize_for_new(copy.deepcopy(simple_normalized))

        # Step 5 & 6: Extract features
        single_features = compute_single_descriptors(normalized_mesh)
        dist_features = compute_distribution_descriptors(
            normalized_mesh,
            n_samples=150000,
            n_bins=100
        )

        # Combine into list
        descriptors = []

        # Add single-value features (7 features)
        single_value_names = [
            'surface_area', 'volume', 'compactness',
            'rectangularity', 'diameter', 'convexity', 'eccentricity'
        ]

        for name in single_value_names:
            value = single_features[name]
            value = 0.0 if np.isnan(value) else value

            # Normalize using database statistics
            if norm_params_df is not None and name in norm_params_df.index:
                mean = norm_params_df.loc[name, 'mean']
                std = norm_params_df.loc[name, 'std']
                if std > 0:
                    value = (value - mean) / std

            descriptors.append(value)

        # Add histogram features
        for hist_name in ['A3', 'D1', 'D2', 'D3', 'D4']:
            hist_values = []
            for i in range(100):
                bin_name = f'{hist_name}_bin{i}'
                hist_values.append(dist_features[bin_name])

            # Normalize histogram to sum=1
            hist_array = np.array(hist_values)
            hist_sum = hist_array.sum()
            if hist_sum > 0:
                hist_array = hist_array / hist_sum

            descriptors.extend(hist_array)

        return descriptors

    except Exception as e:
        print(f"Error processing model {input_mesh_path}: {e}")
        raise


def fast_query(input_mesh_path, descriptors_path, normalization_params_path,
               ann_index, K, stopwatch: Stopwatch | None = None,
               weights=None):
    """
    Fast query using ANN

    Parameters:
        input_mesh_path: Query model path
        descriptors_path: Feature database path (normalized)
        normalization_params_path: Normalization parameters CSV path
        ann_index: ANN index
        K: Number of results
        stopwatch: Performance monitoring object
    """
    # Use default weights
    if weights is None:
        weights = WEIGHTS

    # Load normalization parameters
    norm_params_df = pd.read_csv(normalization_params_path, index_col=0)

    # Process query
    descriptors = process_new_model(input_mesh_path, norm_params_df)
    db_descriptors = pd.read_csv(descriptors_path)

    if stopwatch is not None:
        stopwatch.start()

    # Get candidates from ANN
    K_candidates = min(K * 3, len(db_descriptors))
    indices, raw_distances = ann(ann_index, descriptors, K_candidates)

    candidate_indices = indices[0, :]
    candidate_features = db_descriptors.iloc[candidate_indices]

    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    # Extract candidate features
    single_value_features = candidate_features.iloc[:, 2:9].values
    histogram_features = [
        candidate_features.iloc[:, 9:109].values,  # A3 (100 bins)
        candidate_features.iloc[:, 109:209].values,  # D1 (100 bins)
        candidate_features.iloc[:, 209:309].values,  # D2 (100 bins)
        candidate_features.iloc[:, 309:409].values,  # D3 (100 bins)
        candidate_features.iloc[:, 409:509].values  # D4 (100 bins)
    ]

    query_single_values = np.array(descriptors[:7])
    query_histograms = [
        np.array(descriptors[7:107]),  # A3 (100 bins)
        np.array(descriptors[107:207]),  # D1 (100 bins)
        np.array(descriptors[207:307]),  # D2 (100 bins)
        np.array(descriptors[307:407]),  # D3 (100 bins)
        np.array(descriptors[407:507])  # D4 (100 bins)
    ]

    n_candidates = len(candidate_features)
    total_distances = np.zeros(n_candidates)

    # 1. Single-value distances (Euclidean)
    for i, feature_name in enumerate(single_value_names):
        feature_distances = np.abs(single_value_features[:, i] - query_single_values[i])
        weight = weights['single_value'][feature_name]
        total_distances += weight * feature_distances

    # 2. Histogram distances (EMD)
    histogram_names = ['A3', 'D1', 'D2', 'D3', 'D4']
    for i, hist_name in enumerate(histogram_names):
        query_hist = query_histograms[i]
        db_hists = histogram_features[i]

        # Compute EMD for all candidates
        emd_distances = np.array([
            wasserstein_distance(query_hist, db_hists[j])
            for j in range(len(db_hists))
        ])

        weight = weights['histogram'][hist_name]
        total_distances += weight * emd_distances

    # 3. Find top K
    top_k_indices = np.argsort(total_distances)[:K]
    final_indices = candidate_indices[top_k_indices]
    final_distances = total_distances[top_k_indices]

    closest_models = db_descriptors.iloc[final_indices][['class_name', 'file_name']].values

    if stopwatch is not None:
        stopwatch.stop()
        stopwatch.record_time()

    return [model[0] for model in closest_models], list(zip(closest_models, final_distances))