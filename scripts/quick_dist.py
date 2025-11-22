import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

# Set style
sns.set_style("whitegrid")

# Ground resolution (meters per pixel)
GROUND_RESOLUTION = 0.045

def load_coco_annotations(coco_path: Path) -> list:
    """Load annotations from COCO JSON file."""
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data.get('annotations', [])


def extract_areas(coco_path: Path) -> np.ndarray:
    """Extract areas (in pixels²) from COCO annotations."""
    annotations = load_coco_annotations(coco_path)
    areas = [ann['area'] for ann in annotations if ann.get('area', 0) > 0]
    return np.array(areas)


def find_coco_files(base_dir: Path) -> List[Path]:
    """Find all JSON files containing 'coco' in their name."""
    coco_files = []
    for json_file in base_dir.rglob('*.json'):
        if 'coco' in json_file.name.lower():
            coco_files.append(json_file)
    return sorted(coco_files)



def analyze_category_splits(all_areas: np.ndarray, site_name: str = "All Sites"):
    """Analyze different ways to split into small/medium/large categories."""
    
    print(f"\n{'='*80}")
    print(f"CATEGORY SPLIT ANALYSIS - {site_name}")
    print(f"{'='*80}")
    print(f"Total segmentations: {len(all_areas):,}\n")
    
    # Method 1: Equal count (terciles - 33/67 percentiles)
    p33 = np.percentile(all_areas, 33.33)
    p67 = np.percentile(all_areas, 66.67)
    
    small_count_m1 = np.sum(all_areas < p33)
    medium_count_m1 = np.sum((all_areas >= p33) & (all_areas < p67))
    large_count_m1 = np.sum(all_areas >= p67)
    
    print("METHOD 1: Equal Count Split (Terciles)")
    print("-" * 80)
    print(f"Small:  < {p33:.2f} px² (< {p33 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {small_count_m1:,} ({small_count_m1/len(all_areas)*100:.1f}%)")
    print(f"Medium: {p33:.2f} - {p67:.2f} px² ({p33 * GROUND_RESOLUTION**2:.5f} - {p67 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {medium_count_m1:,} ({medium_count_m1/len(all_areas)*100:.1f}%)")
    print(f"Large:  > {p67:.2f} px² (> {p67 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {large_count_m1:,} ({large_count_m1/len(all_areas)*100:.1f}%)")
    
    # Method 2: Quartile-based (25/75 percentiles)
    p25 = np.percentile(all_areas, 25)
    p75 = np.percentile(all_areas, 75)
    
    small_count_m2 = np.sum(all_areas < p25)
    medium_count_m2 = np.sum((all_areas >= p25) & (all_areas < p75))
    large_count_m2 = np.sum(all_areas >= p75)
    
    print(f"\nMETHOD 2: Quartile Split (25th/75th percentiles)")
    print("-" * 80)
    print(f"Small:  < {p25:.2f} px² (< {p25 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {small_count_m2:,} ({small_count_m2/len(all_areas)*100:.1f}%)")
    print(f"Medium: {p25:.2f} - {p75:.2f} px² ({p25 * GROUND_RESOLUTION**2:.5f} - {p75 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {medium_count_m2:,} ({medium_count_m2/len(all_areas)*100:.1f}%)")
    print(f"Large:  > {p75:.2f} px² (> {p75 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {large_count_m2:,} ({large_count_m2/len(all_areas)*100:.1f}%)")
    
    # Method 3: COCO-style (standard area thresholds in m²)
    small_thresh_m2 = 32 * 32 * (GROUND_RESOLUTION ** 2)
    medium_thresh_m2 = 96 * 96 * (GROUND_RESOLUTION ** 2)
    
    small_thresh_px = small_thresh_m2 / (GROUND_RESOLUTION ** 2)
    medium_thresh_px = medium_thresh_m2 / (GROUND_RESOLUTION ** 2)
    
    small_count_m3 = np.sum(all_areas < small_thresh_px)
    medium_count_m3 = np.sum((all_areas >= small_thresh_px) & (all_areas < medium_thresh_px))
    large_count_m3 = np.sum(all_areas >= medium_thresh_px)
    
    print(f"\nMETHOD 3: COCO-Style Fixed Thresholds")
    print("-" * 80)
    print(f"Small:  < {small_thresh_px:.2f} px² (< {small_thresh_m2:.5f} m²)")
    print(f"        Count: {small_count_m3:,} ({small_count_m3/len(all_areas)*100:.1f}%)")
    print(f"Medium: {small_thresh_px:.2f} - {medium_thresh_px:.2f} px² ({small_thresh_m2:.5f} - {medium_thresh_m2:.5f} m²)")
    print(f"        Count: {medium_count_m3:,} ({medium_count_m3/len(all_areas)*100:.1f}%)")
    print(f"Large:  > {medium_thresh_px:.2f} px² (> {medium_thresh_m2:.5f} m²)")
    print(f"        Count: {large_count_m3:,} ({large_count_m3/len(all_areas)*100:.1f}%)")
    
    # Method 4: Custom based on median and IQR
    median = np.median(all_areas)
    q1 = np.percentile(all_areas, 25)
    q3 = np.percentile(all_areas, 75)
    iqr = q3 - q1
    
    small_thresh_m4 = median - 0.5 * iqr
    large_thresh_m4 = median + 0.5 * iqr
    
    small_count_m4 = np.sum(all_areas < small_thresh_m4)
    medium_count_m4 = np.sum((all_areas >= small_thresh_m4) & (all_areas < large_thresh_m4))
    large_count_m4 = np.sum(all_areas >= large_thresh_m4)
    
    print(f"\nMETHOD 4: Median ± 0.5×IQR")
    print("-" * 80)
    print(f"Small:  < {small_thresh_m4:.2f} px² (< {small_thresh_m4 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {small_count_m4:,} ({small_count_m4/len(all_areas)*100:.1f}%)")
    print(f"Medium: {small_thresh_m4:.2f} - {large_thresh_m4:.2f} px² ({small_thresh_m4 * GROUND_RESOLUTION**2:.5f} - {large_thresh_m4 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {medium_count_m4:,} ({medium_count_m4/len(all_areas)*100:.1f}%)")
    print(f"Large:  > {large_thresh_m4:.2f} px² (> {large_thresh_m4 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {large_count_m4:,} ({large_count_m4/len(all_areas)*100:.1f}%)")
    
    # Method 5: Peak-based split (keep main peak, then split rest 50/50)
    # Use median as threshold for main peak, then split remaining at midpoint
    median = np.median(all_areas)
    
    # For the remaining distribution above median, find the midpoint
    above_median = all_areas[all_areas >= median]
    if len(above_median) > 0:
        # Split remaining 50/50
        midpoint_remaining = np.median(above_median)
    else:
        midpoint_remaining = median
    
    small_count_m5 = np.sum(all_areas < median)
    medium_count_m5 = np.sum((all_areas >= median) & (all_areas < midpoint_remaining))
    large_count_m5 = np.sum(all_areas >= midpoint_remaining)
    
    print(f"\nMETHOD 5: Peak Focus (Median, then split rest 50/50)")
    print("-" * 80)
    print(f"📌 Keeps main distribution peak intact, splits tail evenly")
    print(f"Small:  < {median:.2f} px² (< {median * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {small_count_m5:,} ({small_count_m5/len(all_areas)*100:.1f}%)")
    print(f"Medium: {median:.2f} - {midpoint_remaining:.2f} px² ({median * GROUND_RESOLUTION**2:.5f} - {midpoint_remaining * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {medium_count_m5:,} ({medium_count_m5/len(all_areas)*100:.1f}%)")
    print(f"Large:  > {midpoint_remaining:.2f} px² (> {midpoint_remaining * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"        Count: {large_count_m5:,} ({large_count_m5/len(all_areas)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print("For balanced evaluation across size categories, use METHOD 1 (Equal Count).")
    print("For peak-focused split with tail division, use METHOD 5 (Peak Focus). ⭐")
    print("For COCO-compatible metrics, use METHOD 3 (COCO-Style).")
    print("For data-driven adaptive split, use METHOD 2 (Quartiles).")
    print(f"{'='*80}\n")
    
    return {
        'terciles': (p33, p67),
        'quartiles': (p25, p75),
        'coco_style': (small_thresh_px, medium_thresh_px),
        'median_iqr': (small_thresh_m4, large_thresh_m4),
        'peak_focus': (median, midpoint_remaining),  # NEW!
    }


def plot_histogram(all_areas: np.ndarray, site_name: str = "All Sites", remove_outliers: bool = False):
    """Create a histogram of segmentation sizes."""
    
    # Keep ALL data (no outlier removal)
    filtered_areas = all_areas
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot histogram
    n, bins, patches = ax.hist(filtered_areas, bins=100, color='steelblue', 
                                edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Labels
    ax.set_xlabel('Size (pixels²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Segmentations)', fontsize=14, fontweight='bold')
    ax.set_title(f'Segmentation Size Distribution - {site_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Statistics
    mean_size = np.mean(filtered_areas)
    median_size = np.median(filtered_areas)
    std_size = np.std(filtered_areas)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_size:.2f} px²')
    ax.axvline(median_size, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_size:.2f} px²')
    
    # Add percentile lines for RECOMMENDED split (Peak Focus - Method 5)
    above_median = filtered_areas[filtered_areas >= median_size]
    if len(above_median) > 0:
        midpoint_remaining = np.median(above_median)
    else:
        midpoint_remaining = median_size
        
    ax.axvline(median_size, color='green', linestyle=':', linewidth=2.5, alpha=0.8,
               label=f'Median: {median_size:.2f} px² (Small/Med) ⭐')
    ax.axvline(midpoint_remaining, color='purple', linestyle=':', linewidth=2.5, alpha=0.8,
               label=f'75th %ile: {midpoint_remaining:.2f} px² (Med/Large) ⭐')
    
    # Also show tercile split for comparison (lighter)
    p33 = np.percentile(filtered_areas, 33.33)
    p67 = np.percentile(filtered_areas, 66.67)
    ax.axvline(p33, color='lightgreen', linestyle='-.', linewidth=1.5, alpha=0.5,
               label=f'33rd %ile: {p33:.2f} px² (tercile)')
    ax.axvline(p67, color='plum', linestyle='-.', linewidth=1.5, alpha=0.5,
               label=f'67th %ile: {p67:.2f} px² (tercile)')
    
    ax.legend(fontsize=9, loc='upper right')
    
    # Statistics box
    p25 = np.percentile(filtered_areas, 25)
    p75 = np.percentile(filtered_areas, 75)
    p99 = np.percentile(filtered_areas, 99)
    
    stats_text = (
        f'Total: {len(filtered_areas):,} segmentations\n'
        f'Mean: {mean_size:.2f} px² ({mean_size * GROUND_RESOLUTION**2:.5f} m²)\n'
        f'Median: {median_size:.2f} px² ({median_size * GROUND_RESOLUTION**2:.5f} m²)\n'
        f'Std Dev: {std_size:.2f} px²\n'
        f'Min: {np.min(filtered_areas):.2f} px²\n'
        f'Max: {np.max(filtered_areas):.2f} px²\n'
        f'---\n'
        f'RECOMMENDED (Peak Focus):\n'
        f'S/M: {median_size:.2f} px² (50th %ile)\n'
        f'M/L: {midpoint_remaining:.2f} px² (75th %ile)\n'
        f'---\n'
        f'Alternative (Tercile):\n'
        f'S/M: {p33:.2f} px²\n'
        f'M/L: {p67:.2f} px²'
    )
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = Path('quick_dist_output')
    output_path.mkdir(exist_ok=True)
    safe_name = site_name.replace(' ', '_').replace('/', '_')
    plt.savefig(output_path / f'histogram_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / f'histogram_{safe_name}.png'}")
    plt.close()
    
    # Print detailed stats
    print(f"\n{'='*70}")
    print(f"STATISTICS - {site_name}")
    print(f"{'='*70}")
    print(f"Total count: {len(filtered_areas):,}")
    print(f"\nSize statistics (pixels²):")
    print(f"  Mean:   {mean_size:.2f} px² ({mean_size * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"  Median: {median_size:.2f} px² ({median_size * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"  Std:    {std_size:.2f} px²")
    print(f"  Min:    {np.min(filtered_areas):.2f} px²")
    print(f"  Max:    {np.max(filtered_areas):.2f} px²")
    print(f"\nPercentiles (pixels²):")
    print(f"  25th: {p25:.2f} px² ({p25 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"  50th: {median_size:.2f} px² ({median_size * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"  75th: {p75:.2f} px² ({p75 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"  99th: {p99:.2f} px² ({p99 * GROUND_RESOLUTION**2:.5f} m²)")
    print(f"{'='*70}")
    
    return filtered_areas
def plot_dual_scale_histogram(all_areas: np.ndarray, site_name: str = "All Sites"):
    """
    Create side-by-side histograms: linear scale and log scale.
    X-axis in m² (linear), Y-axis as frequency (right plot has log Y-axis).
    """
    
    # Convert areas from pixels² to m²
    areas_m2 = all_areas * (GROUND_RESOLUTION ** 2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Calculate statistics
    mean_m2 = np.mean(areas_m2)
    median_m2 = np.median(areas_m2)
    
    # ==================== LINEAR SCALE ====================
    n1, bins1, patches1 = ax1.hist(areas_m2, bins=100, color='steelblue', 
                                     edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax1.set_xlabel('Area (m²)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=18, fontweight='bold')
    ax1.set_title('Distribution - Linear Scale', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='both', linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add ONLY median and mean lines
    ax1.axvline(median_m2, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.axvline(mean_m2, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Simple legend
    ax1.legend([f'Median: {median_m2:.2f}', f'Mean: {mean_m2:.2f}'], 
               fontsize=14, loc='upper right', framealpha=0.95)
    
    # ==================== LOG Y-AXIS SCALE ====================
    # Use regular bins (not log-spaced) since X-axis is linear
    n2, bins2, patches2 = ax2.hist(areas_m2, bins=100, color='steelblue', 
                                     edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax2.set_xlabel('Area (m²)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Frequency (log scale)', fontsize=18, fontweight='bold')
    ax2.set_title('Distribution - Log Scale', fontsize=20, fontweight='bold')
    
    # ONLY log scale on Y-axis, X-axis stays linear
    ax2.set_yscale('log')
    
    # X-axis starts at 0
    ax2.set_xlim(left=0)
    
    ax2.grid(True, alpha=0.3, axis='both', linestyle='--', linewidth=0.5, which='both')
    ax2.set_axisbelow(True)
    
    # Add ONLY median and mean lines
    ax2.axvline(median_m2, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.axvline(mean_m2, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Simple legend
    ax2.legend([f'Median: {median_m2:.2f}', f'Mean: {mean_m2:.2f}'], 
               fontsize=14, loc='upper right', framealpha=0.95)
    
    # Main title
    fig.suptitle(f'Segmentation Area Distribution - {site_name}', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = Path('quick_dist_output')
    output_path.mkdir(exist_ok=True)
    safe_name = site_name.replace(' ', '_').replace('/', '_')
    plt.savefig(output_path / f'dual_scale_histogram_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path / f'dual_scale_histogram_{safe_name}.png'}")
    plt.close()
def main():
    """Main execution function."""
    print("="*80)
    print("COCO SEGMENTATION SIZE HISTOGRAM & CATEGORY ANALYSIS")
    print("="*80)
    
    base_dir = Path('../selvamask')
    print(f"\nSearching for COCO files in: {base_dir.resolve()}")
    
    coco_files = find_coco_files(base_dir)
    
    if not coco_files:
        print(f"\n⚠️ No COCO files found!")
        return
    
    print(f"\nFound {len(coco_files)} COCO files:")
    for f in coco_files:
        print(f"  → {f.relative_to(base_dir)}")
    
    # Process each site
    all_sites_areas = []
    
    for coco_file in coco_files:
        site_name = coco_file.parent.name
        areas = extract_areas(coco_file)
        if len(areas) > 0:
            print(f"\nProcessing {site_name}: {len(areas):,} segmentations")
            # Keep all outliers - no filtering
            filtered = plot_histogram(areas, site_name, remove_outliers=False)
            all_sites_areas.append(areas)  # Keep original areas
            
            # NEW: Create dual-scale histogram
            plot_dual_scale_histogram(areas, site_name)
            
            # Analyze category splits for this site
            analyze_category_splits(filtered, site_name)
    
    # Combined histogram and analysis
    if all_sites_areas:
        all_areas = np.concatenate(all_sites_areas)
        print(f"\nProcessing combined data: {len(all_areas):,} segmentations")
        # Keep all outliers - no filtering
        filtered_combined = plot_histogram(all_areas, "All Sites Combined", remove_outliers=False)
        
        # NEW: Create dual-scale histogram for combined data
        plot_dual_scale_histogram(all_areas, "All Sites Combined")
        
        # Final category analysis for all sites combined
        thresholds = analyze_category_splits(filtered_combined, "All Sites Combined")
        
        # Print final recommendation
        print("\n" + "="*80)
        print("FINAL RECOMMENDATIONS FOR YOUR CODE")
        print("="*80)
        
        # RECOMMENDED: Peak Focus method (Method 5)
        peak_small, peak_large = thresholds['peak_focus']
        print(f"\n⭐ RECOMMENDED (Peak Focus - Median + 50/50 split):")
        print(f"  small_max_sq_meters = {peak_small * GROUND_RESOLUTION**2:.6f}")
        print(f"  medium_max_sq_meters = {peak_large * GROUND_RESOLUTION**2:.6f}")
        print(f"\n  Or in pixels²:")
        print(f"  small_max_pixels = {peak_small:.2f}")
        print(f"  medium_max_pixels = {peak_large:.2f}")
        
        # Alternative: Terciles
        tercile_small, tercile_large = thresholds['terciles']
        print(f"\n💡 Alternative (Equal Count - Terciles):")
        print(f"  small_max_sq_meters = {tercile_small * GROUND_RESOLUTION**2:.6f}")
        print(f"  medium_max_sq_meters = {tercile_large * GROUND_RESOLUTION**2:.6f}")
        
        print("="*80)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == '__main__':
    main()