import os
import geopandas as gpd
from geodataset.utils import get_utm_crs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for KDE plotting and aesthetics
from matplotlib.ticker import ScalarFormatter


def get_avg_side_length(geom):
    """
    Given a geometry, compute its bounding box dimensions and return the average side length.
    Assumes the geometry is a box (i.e. rectangle).
    """
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    return (width + height) / 2

def process_gpkg(filepath, stats):
    """
    Process one GeoPackage file and update stats dictionary.
    Reprojects the entire GeoDataFrame using a UTM CRS determined from its total bounds.
    """
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
    
    left, bottom, right, top = gdf.total_bounds
    centroid_lon = (left + right) / 2.0
    centroid_lat = (top + bottom) / 2.0

    # Determine target UTM CRS based on the GeoDataFrame's overall bounds
    target_crs = get_utm_crs(centroid_lon, centroid_lat)

    # Reproject to a CRS in meters
    gdf = gdf.to_crs(target_crs)

    for idx, row in gdf.iterrows():
        if row.geometry is None:
            continue
        avg_side = get_avg_side_length(row.geometry)
        stats['total_objects'] += 1
        stats['side_lengths'].append(avg_side)
        
        # Categorize by average side length (for reporting purposes)
        if avg_side < 2:
            stats['categories']['<2'] += 1
        elif 2 <= avg_side < 5:
            stats['categories']['2-5'] += 1
        elif 5 <= avg_side < 10:
            stats['categories']['5-10'] += 1
        elif 10 <= avg_side < 20:
            stats['categories']['10-20'] += 1
        elif 20 <= avg_side < 30:
            stats['categories']['20-30'] += 1
        elif 30 <= avg_side < 40:
            stats['categories']['30-40'] += 1
        elif 40 <= avg_side < 50:
            stats['categories']['40-50'] += 1
        else:  # avg_side >= 50
            stats['categories']['>50'] += 1

        # Update smallest and largest objects tracking
        if stats['min_side'] is None or avg_side < stats['min_side']:
            stats['min_side'] = avg_side
            stats['min_obj'] = (filepath, idx, avg_side)
        if avg_side > stats['max_side']:
            stats['max_side'] = avg_side
            stats['max_obj'] = (filepath, idx, avg_side)

def traverse_directory(root_dir):
    """
    Traverse the directory recursively, process all *_boxes.gpkg files,
    and return aggregated statistics.
    """
    stats = {
        'total_objects': 0,
        'side_lengths': [],
        'categories': {
            '<2': 0,
            '2-5': 0,
            '5-10': 0,
            '10-20': 0,
            '20-30': 0,
            '30-40': 0,
            '40-50': 0,
            '>50': 0
        },
        'min_side': None,
        'max_side': 0,
        'min_obj': None,  # (filepath, row index, side length)
        'max_obj': None
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if file is a GeoPackage ending with _boxes.gpkg
            if filename.endswith("_boxes.gpkg"):
                filepath = os.path.join(dirpath, filename)
                print(f"Processing: {filepath}")
                process_gpkg(filepath, stats)
                
    return stats

def print_statistics(name, stats):
    print(f"=== Statistics Summary for {name} ===")
    print(f"Total number of objects: {stats['total_objects']}")
    print("Counts by average side length:")
    for category, count in stats['categories'].items():
        print(f"  {category} m: {count}")
    
    if stats['side_lengths']:
        overall_avg = sum(stats['side_lengths']) / len(stats['side_lengths'])
        print(f"Overall average side length: {overall_avg:.2f} m")
    
    if stats['min_obj']:
        filepath, idx, min_side = stats['min_obj']
        print(f"Smallest object: {min_side:.2f} m (File: {filepath}, Row: {idx})")
    if stats['max_obj']:
        filepath, idx, max_side = stats['max_obj']
        print(f"Largest object: {max_side:.2f} m (File: {filepath}, Row: {idx})")
    print()

if __name__ == "__main__":
    # Define parent folders for each dataset
    output_plot_path = './experiments/dataset_statistics/box_sizes_histogram.png'
    directories = {
        'Brazil ZF2': '/network/scratch/h/hugo.baudchon/data/raw/brazil_zf2',
        'Ecuador Tiputini': '/network/scratch/h/hugo.baudchon/data/raw/ecuador_tiputini',
        'Panama Aguasalud': '/network/scratch/h/hugo.baudchon/data/raw/panama_aguasalud'
    }

    # Dictionary to hold stats for each dataset
    dataset_stats = {}

    for name, folder in directories.items():
        print(f"\nScanning dataset: {name}")
        stats = traverse_directory(folder)
        dataset_stats[name] = stats
        print_statistics(name, stats)
    
    # Compute the global maximum side length to have consistent bins
    global_max = 0
    for stats in dataset_stats.values():
        if stats['side_lengths']:
            global_max = max(global_max, max(stats['side_lengths']))
    
    # Create and save two separate histogram plots

    # Set Seaborn style for prettier plots
    sns.set(style="whitegrid", font_scale=1.2)

    # Define output paths for the two plots
    categorical_plot_path = './experiments/dataset_statistics/box_sizes_categorical.png'
    histogram_5m_plot_path = './experiments/dataset_statistics/box_sizes_histogram_5m.png'

    # ---------------------------------------------------------------------------------
    # FIGURE 1: Categorical grouped bar plot
    # ---------------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))

    # Set up the categorical bins
    categories = ['<2', '2-5', '5-10', '10-20', '20-30', '30-40', '40-50', '>50']
    x = np.arange(len(categories))
    width = 0.25  # width of the bars
    dataset_names = list(dataset_stats.keys())

    # Plot grouped bars for each dataset
    for i, (name, stats) in enumerate(dataset_stats.items()):
        # Extract category counts from stats
        counts = [stats['categories'][cat] for cat in categories]
        
        # Position bars for this dataset
        offset = (i - len(dataset_names)/2 + 0.5) * width
        bars = plt.bar(
            x + offset, 
            counts, 
            width, 
            label=name,
            color=plt.cm.tab10(i % 10),
            edgecolor='black',
            linewidth=1
        )
        
        # Add count labels above bars
        for bar, count in zip(bars, counts):
            if count > 0:  # Only label non-zero bars
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(1, bar.get_height() * 0.05),
                    str(count),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90 if count > 1000 else 0
                )

    plt.xlabel("Average Box Side Length (m)")
    plt.ylabel("Number of Objects (log scale)")
    plt.title("Distribution of Box Sizes by Dataset (Categorical Bins)")
    plt.xticks(x, categories)
    plt.ylim(bottom=0.5)  # Start y-axis slightly above zero for log scale

    # Use logarithmic y-axis with plain tick labels
    plt.yscale("log")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Add legend
    plt.legend(title="Dataset", loc='upper right')

    # Add grid lines on the y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save categorical plot
    plt.savefig(categorical_plot_path)
    print(f"Categorical bar plot saved to {categorical_plot_path}")
    plt.close()  # Close the first figure

    # ---------------------------------------------------------------------------------
    # FIGURE 2: Histogram with 5m bins as bars (Updated with wider bars and bin range labels)
    # ---------------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))

    # Compute the global maximum side length to have consistent bins
    global_max = 0
    for stats in dataset_stats.values():
        if stats['side_lengths']:
            global_max = max(global_max, max(stats['side_lengths']))

    # Create bins from 0 to global_max (5m bins)
    bins = np.arange(0, global_max + 5, 5)  # Using 5m bins

    # Get all bin counts for each dataset
    dataset_counts = {}
    bin_centers = []
    max_count = 0

    for name, stats in dataset_stats.items():
        if stats['side_lengths']:
            counts, bin_edges = np.histogram(stats['side_lengths'], bins=bins)
            dataset_counts[name] = counts
            if len(bin_centers) == 0:
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            max_count = max(max_count, max(counts)) if len(counts) > 0 else max_count

    # Generate x-axis labels with the range for each bin
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}m" for i in range(len(bins)-1)]

    # Plot bars for each dataset with a grouped bar chart approach
    n_datasets = len(dataset_counts)
    total_width = 3.0  # increased total width for each bin to make bars 3 times wider
    single_width = total_width / n_datasets  # width for each individual bar

    for i, (name, counts) in enumerate(dataset_counts.items()):
        # Calculate the shift from the bin center for this dataset's bars
        offset = (i - n_datasets/2 + 0.5) * single_width
        
        # Only plot bins with non-zero counts
        nonzero_idx = np.where(counts > 0)[0]
        
        plt.bar(
            bin_centers[nonzero_idx] + offset,
            counts[nonzero_idx],
            width=single_width,  # using the updated single_width for wider bars
            label=name,
            color=plt.cm.tab10(i % 10),
            edgecolor='black',
            linewidth=0.5
        )

    plt.xlabel("Average Box Side Length (m) [5m bins]")
    plt.ylabel("Number of Objects (log scale)")
    plt.title("Histogram of Box Sizes with 5m Bins")
    plt.xlim(left=-2.5, right=bin_centers[-1] + 5)  # Set proper x limits

    # Set x-axis ticks with the range labels for each bin
    plt.xticks(bin_centers, bin_labels)

    # Use logarithmic y-axis with plain tick labels
    plt.yscale("log")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Add legend
    plt.legend(title="Dataset", loc='upper right')

    # Add grid lines on the y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save 5m bin histogram plot
    plt.savefig(histogram_5m_plot_path)
    print(f"5m bin histogram plot saved to {histogram_5m_plot_path}")
    plt.close()

