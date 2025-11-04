import pandas as pd
import numpy as np
from pathlib import Path


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

    print(f"\nTotal {len(merged_df)} models")
    print(f"{len(all_files)} categories")
    print(f"Total columns: {len(merged_df.columns)}")

    # Create output directory (if it doesn't exist)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    print("=" * 60)

    return merged_df


def normalize_features(csv_path, output_csv='database/normalized_features.csv',
                       output_params_csv='database/normalization_params.csv'):
    """
    Normalize features in the database

    CSV structure:
    - Column 0: class_name
    - Column 1: file_name
    - Columns 2-8: 7 single-value features (will be standardized)
    - Columns 9-108: A3 histogram (100 bins, will be normalized to sum=1)
    - Columns 109-208: D1 histogram (100 bins)
    - Columns 209-308: D2 histogram (100 bins)
    - Columns 309-408: D3 histogram (100 bins)
    - Columns 409-508: D4 histogram (100 bins)

    Parameters:
        csv_path: Input feature database CSV path
        output_csv: Output normalized CSV file path
        output_params_csv: Output normalization parameters CSV file path

    Returns:
        Normalized DataFrame and normalization parameters
    """
    print("=" * 60)
    print("Normalizing features...")
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

    # Create normalized dataframe
    df_normalized = df.copy()

    # Single-value feature names
    single_value_names = [
        'surface_area', 'volume', 'compactness',
        'rectangularity', 'diameter', 'convexity', 'eccentricity'
    ]

    # Dictionary to store normalization parameters
    normalization_params = {}

    # Standardize single-value features
    print("Standardizing single-value features (Z-score)...")
    for i, name in enumerate(single_value_names):
        raw_values = df.iloc[:, 2 + i].values

        # Compute mean and std
        mean_val = np.mean(raw_values)
        std_val = np.std(raw_values)

        # Store parameters
        normalization_params[name] = {
            'mean': mean_val,
            'std': std_val,
            'raw_min': float(raw_values.min()),
            'raw_max': float(raw_values.max())
        }

        # Standardization: (x - mean) / std
        if std_val > 0:
            standardized = (raw_values - mean_val) / std_val
        else:
            standardized = raw_values

        # Save to normalized dataframe
        df_normalized.iloc[:, 2 + i] = standardized

        print(f"  {name:20} : "
              f"raw [{raw_values.min():8.2f}, {raw_values.max():8.2f}] -> "
              f"std [{standardized.min():7.2f}, {standardized.max():7.2f}]")

    print()

    # Normalize histogram features
    histogram_specs = [
        ('A3', 9, 109),
        ('D1', 109, 209),
        ('D2', 209, 309),
        ('D3', 309, 409),
        ('D4', 409, 509)
    ]

    print("Normalizing histogram features (sum = 1)...")
    for hist_name, start_col, end_col in histogram_specs:
        hist_data = df.iloc[:, start_col:end_col].values
        hist_sums = hist_data.sum(axis=1)

        # Check if already normalized
        all_normalized = np.allclose(hist_sums, 1.0, rtol=1e-5)

        if all_normalized:
            print(f"  {hist_name:20} : Already normalized (sum = 1.0)")
        else:
            print(f"  {hist_name:20} : Normalizing... ", end='')

            # Avoid division by zero
            hist_sums[hist_sums == 0] = 1.0

            # Normalize: each histogram sums to 1
            hist_normalized = hist_data / hist_sums[:, np.newaxis]
            df_normalized.iloc[:, start_col:end_col] = hist_normalized

            # Verify
            new_sums = hist_normalized.sum(axis=1)
            print(f"Done (sum range: [{new_sums.min():.6f}, {new_sums.max():.6f}])")

        # Store histogram info
        normalization_params[hist_name] = {
            'type': 'histogram',
            'bins': end_col - start_col,
            'normalized': 'sum_to_one'
        }

    print()

    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save normalized database
    df_normalized.to_csv(output_csv, index=False)
    print(f"Normalized database saved to: {output_csv}")

    # Save normalization parameters
    params_df = pd.DataFrame(normalization_params).T
    params_path = Path(output_params_csv)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(output_params_csv)
    print(f"Normalization parameters saved to: {output_params_csv}")

    print("=" * 60)

    return df_normalized, normalization_params


def print_normalization_summary(df_normalized, normalization_params):
    """
    Print summary of normalized features
    """
    print("\n" + "=" * 60)
    print("Normalization Summary")
    print("=" * 60)

    print("\nSingle-value features (should have mean~0, std~1):")
    for i, name in enumerate(['surface_area', 'volume', 'compactness',
                              'rectangularity', 'diameter', 'convexity', 'eccentricity']):
        col = df_normalized.iloc[:, 2 + i]
        print(f"  {name:20} : mean={col.mean():7.4f}, std={col.std():7.4f}")

    print("\nHistogram features (should sum to 1):")
    histogram_ranges = [
        ('A3', 9, 109),
        ('D1', 109, 209),
        ('D2', 209, 309),
        ('D3', 309, 409),
        ('D4', 409, 509)
    ]
    for name, start, end in histogram_ranges:
        hist_data = df_normalized.iloc[:, start:end].values
        sums = hist_data.sum(axis=1)
        print(f"  {name:20} : sum range [{sums.min():.6f}, {sums.max():.6f}]")

    print("\n" + "=" * 60)


# ==================== Complete Pipeline ====================

if __name__ == "__main__":
    # Step 1: Merge feature files from all categories
    print("\nStep 1: Merge feature files from all categories\n")
    merged_df = merge_all_descriptors(
        features_folder='features',
        output_csv='database/all_features_raw.csv'
    )

    # Step 2: Normalize features
    print("\nStep 2: Normalize features\n")
    normalized_df, norm_params = normalize_features(
        csv_path='database/all_features_raw.csv',
        output_csv='database/all_features.csv',
        output_params_csv='database/normalization_params.csv'
    )

    # Step 3: Print summary
    print_normalization_summary(normalized_df, norm_params)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("Generated files:")
    print("  1. database/all_features_raw.csv        - Raw features (original values)")
    print("  2. database/all_features.csv            - Normalized features (use this for queries)")
    print("  3. database/normalization_params.csv    - Normalization parameters")
    print("\nNext step:")
    print("  Use normalized database (all_features.csv) for queries")
    print("=" * 60)