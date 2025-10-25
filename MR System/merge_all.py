import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def merge_all_descriptors(features_folder='features', output_csv='database/all_features.csv'):
    """
    Merge feature files from all categories into one large CSV

    Parameters:
        features_folder: Folder containing all *_descriptors.csv files
        output_csv: Output merged CSV file
    """
    print("=" * 60)
    print("Merging feature files from all categories...")
    print("=" * 60)

    features_path = Path(features_folder)
    all_files = list(features_path.glob('*_descriptors.csv'))

    print(f"Found {len(all_files)} feature files\n")

    all_dataframes = []

    for file in all_files:
        class_name = file.stem.replace('_descriptors', '')
        print(f"  Loading: {class_name:20} ({file.name})")

        df = pd.read_csv(file)
        all_dataframes.append(df)

    # Merge all data
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"\n✓ Total {len(merged_df)} models")
    print(f"✓ {len(all_files)} categories")
    print(f"✓ Total columns: {len(merged_df.columns)}")

    # Create output directory (if it doesn't exist)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    merged_df.to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")
    print("=" * 60)

    return merged_df


def compute_distance_statistics(csv_path, output_csv='distance_stats.csv',
                                sample_size=500):
    """
    Pre-compute distance statistics for each feature type (based on all categories)

    CSV structure:
    - Column 0: class_name
    - Column 1: file_name
    - Columns 2-8: 7 single-value features (unnormalized raw values)
    - Columns 9-72: A3 histogram (64 bins)
    - Columns 73-136: D1 histogram (64 bins)
    - Columns 137-200: D2 histogram (64 bins)
    - Columns 201-264: D3 histogram (64 bins)
    - Columns 265-328: D4 histogram (64 bins)

    Parameters:
        csv_path: Feature database CSV path (containing all categories)
        output_csv: Output statistics CSV file path
        sample_size: Sample size (to avoid excessive computation)

    Returns:
        Distance statistics DataFrame
    """
    print("=" * 60)
    print("Pre-computing distance statistics (based on all categories)...")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    n = len(df)

    # Statistics on category distribution
    if 'class_name' in df.columns:
        class_counts = df['class_name'].value_counts()
        print(f"Database size: {n} models")
        print(f"Number of categories: {len(class_counts)} categories")
        print(f"\nCategory distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls:20} : {count:4} models")
        print()

    # Single-value feature names
    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    # Extract features
    features = {}

    # Extract and standardize single-value features
    print("Standardizing single-value features...")
    for i, name in enumerate(single_value_names):
        raw_values = df.iloc[:, 2 + i].values

        # Standardization: (x - mean) / std
        mean_val = np.mean(raw_values)
        std_val = np.std(raw_values)

        if std_val > 0:
            standardized = (raw_values - mean_val) / std_val
        else:
            standardized = raw_values

        features[name] = standardized.reshape(-1, 1)
        print(
            f"  {name:20} : raw range [{raw_values.min():.2f}, {raw_values.max():.2f}] → standardized range [{standardized.min():.2f}, {standardized.max():.2f}]")

    print()

    # Extract histogram features (already normalized)
    features['A3'] = df.iloc[:, 9:73].values  # 64 bins
    features['D1'] = df.iloc[:, 73:137].values  # 64 bins
    features['D2'] = df.iloc[:, 137:201].values  # 64 bins
    features['D3'] = df.iloc[:, 201:265].values  # 64 bins
    features['D4'] = df.iloc[:, 265:329].values  # 64 bins

    print("Feature extraction:")
    for name, data in features.items():
        print(f"  {name:20} : shape {data.shape}")
    print()

    stats_list = []

    # Sampling (to avoid computing all pairwise distances)
    actual_sample_size = min(sample_size, n)
    indices = np.random.choice(n, actual_sample_size, replace=False)
    print(f"Sample size: {actual_sample_size} models")
    print("(Mixed sampling including all categories)\n")

    for feature_name, feature_data in features.items():
        print(f"Processing {feature_name:20} ...", end=' ', flush=True)
        distances = []

        # Determine if single-value feature or histogram feature
        is_single_value = feature_name in single_value_names

        # Calculate pairwise distances for sampled data (including cross-category)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]

                try:
                    if is_single_value:
                        # Single-value features use absolute difference (already standardized)
                        dist = abs(feature_data[idx_i][0] - feature_data[idx_j][0])
                        distances.append(dist)
                    else:
                        # Histograms use cosine distance
                        vec_i = feature_data[idx_i].reshape(1, -1)
                        vec_j = feature_data[idx_j].reshape(1, -1)

                        # Check if vectors are all zeros
                        if np.all(vec_i == 0) and np.all(vec_j == 0):
                            dist = 0.0
                        elif np.all(vec_i == 0) or np.all(vec_j == 0):
                            dist = 1.0
                        else:
                            similarity = cosine_similarity(vec_i, vec_j)[0][0]
                            dist = 1 - similarity

                        distances.append(dist)

                except Exception as e:
                    print(f"\n  Warning: Error computing distance (i={idx_i}, j={idx_j}): {e}")
                    continue

        if len(distances) == 0:
            print(f"Skipped (no valid distance data)")
            continue

        # Compute statistics
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))

        stats_list.append({
            'feature': feature_name,
            'mean': mean_dist,
            'std': std_dist,
            'min': min_dist,
            'max': max_dist
        })

        print(f"mean={mean_dist:.6f}, std={std_dist:.6f}")

    if len(stats_list) == 0:
        print("\n✗ Error: Failed to compute statistics for any features")
        return None

    # Create output directory (if it doesn't exist)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame and save
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_csv, index=False)

    print(f"\n✓ Distance statistics saved to: {output_csv}")
    print(f"✓ Successfully computed statistics for {len(stats_list)} features")
    print("  - 7 single-value features (standardized)")
    print("  - 5 histogram features")
    print("✓ Statistics based on all categories (including intra- and inter-class distances)")
    print("=" * 60)

    return stats_df


# ==================== Complete Pipeline ====================

if __name__ == "__main__":
    # Step 1: Merge feature files from all categories
    print("\nStep 1: Merge feature files from all categories\n")
    merged_df = merge_all_descriptors(
        features_folder='features',
        output_csv='database/all_features.csv'
    )

    # Step 2: Compute distance statistics based on merged data
    print("\nStep 2: Compute distance statistics\n")
    distance_stats = compute_distance_statistics(
        csv_path='database/all_features.csv',
        output_csv='database/distance_stats.csv',
        sample_size=500
    )

    if distance_stats is not None:
        print("\n" + "=" * 60)
        print("✓ Complete!")
        print("=" * 60)
        print("Generated files:")
        print("  1. database/all_features.csv     - Features from all categories")
        print("  2. database/distance_stats.csv   - Distance statistics")
        print("\nDistance statistics summary:")
        print(distance_stats.to_string(index=False))
        print("\nNext step:")
        print("  Use mesh_querying() for queries")
        print("=" * 60)
    else:
        print("\n✗ Processing failed, please check the data")