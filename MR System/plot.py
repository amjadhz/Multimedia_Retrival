import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_single_descriptor(csv_file, descriptor_name, output_folder='plots'):
    """
    Plot histogram for a single descriptor
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Get class name
    class_name = df['class_name'].iloc[0]
    n_meshes = len(df)

    # Get columns for this descriptor
    desc_cols = [col for col in df.columns if col.startswith(f'{descriptor_name}_bin')]
    n_bins = len(desc_cols)

    if n_bins == 0:
        print(f"WARNING: No {descriptor_name} data found in {csv_file}")
        return

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Define range and labels for each descriptor
    descriptor_config = {
        'A3': {
            'range': (0, 180),
            'xlabel': 'Angle (degrees)',
            'title': 'Angle Distribution (A3)'
        },
        'D1': {
            'range': (0, 1.0),
            'xlabel': 'Distance to Barycenter',
            'title': 'Barycenter Distance Distribution (D1)'
        },
        'D2': {
            'range': (0, 2.0),
            'xlabel': 'Distance Between Vertices',
            'title': 'Pairwise Distance Distribution (D2)'
        },
        'D3': {
            'range': (0, 1.0),
            'xlabel': 'Square Root of Triangle Area',
            'title': 'Triangle Area Distribution (D3)'
        },
        'D4': {
            'range': (0, 1.0),
            'xlabel': 'Cube Root of Tetrahedron Volume',
            'title': 'Tetrahedron Volume Distribution (D4)'
        }
    }

    config = descriptor_config.get(descriptor_name)
    if not config:
        print(f"WARNING: Unknown descriptor: {descriptor_name}")
        return

    # Check data range
    data = df[desc_cols].values
    print(f"\n{descriptor_name} - Class: {class_name}")
    print(f"  Meshes: {n_meshes}, Bins: {n_bins}")
    print(f"  Data range: max bin value = {data.max():.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # x-axis values
    x_values = np.linspace(config['range'][0], config['range'][1], n_bins)

    # Plot line for each mesh
    for idx, row in df.iterrows():
        values = [row[col] for col in desc_cols]
        ax.plot(x_values, values, alpha=0.6, linewidth=1.5)

    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel('Normalized Frequency', fontsize=12)
    ax.set_title(f"{config['title']} - {class_name}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(config['range'][0], config['range'][1])
    ax.set_ylim(0, None)

    # Save plot
    output_file = f"{output_folder}/{class_name}_{descriptor_name}_histogram.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  SAVED: {output_file}")
    plt.close()


def plot_all_descriptors(csv_file, output_folder='plots'):
    """
    Plot histograms for all descriptors (A3, D1, D2, D3, D4)
    """
    print("=" * 60)
    print(f"Processing: {csv_file}")
    print("=" * 60)

    descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']

    for desc in descriptors:
        plot_single_descriptor(csv_file, desc, output_folder)

    print(f"\nAll plots saved to folder: {output_folder}/")

def main():

    print("=" * 60)
    print("Shape Descriptor Histogram Visualization Tool")
    print("=" * 60)

    csv_file = "MR System/features/Insect_descriptors.csv"
    #csv_file = "single1_mesh_features.csv"
    output_folder = "plots"

    if not Path(csv_file).exists():
        print(f"ERROR: File not found: {csv_file}")
        return

    # Plot each descriptor separately
    print("\nPlotting individual descriptor charts")
    plot_all_descriptors(csv_file, output_folder)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()