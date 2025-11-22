#!/usr/bin/env python3
# filepath: /home/soduguay/selvamask/CanopyRS/quick_viz.py
"""
Compare evaluation metrics across different models and sites.
Generates comparison graphs and summary tables.
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
RESULTS_BASE = Path('results_quantile')
GRAPHS_OUTPUT = RESULTS_BASE / 'graphs'
GRAPHS_OUTPUT.mkdir(parents=True, exist_ok=True)

# Valid model prefixes to process
VALID_MODEL_PREFIXES = ['deepforest', 'detectree2', 'dino']

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_metrics(results_dir: Path, metric_type: str):
    """
    Load metrics from JSON file.
    
    Args:
        results_dir: Directory containing the metrics file
        metric_type: 'tile' or 'raster'
    
    Returns:
        dict or None if file doesn't exist
    """
    metric_file = results_dir / f'{metric_type}_metrics.json'
    if metric_file.exists():
        with open(metric_file, 'r') as f:
            return json.load(f)
    return None


def collect_all_metrics():
    """
    Collect all metrics from results folders.
    Only processes folders starting with valid model prefixes.
    
    Returns:
        tuple: (tile_metrics_df, raster_metrics_df)
    """
    tile_data = []
    raster_data = []
    
    # Iterate through all result directories
    for result_dir in RESULTS_BASE.iterdir():
        if not result_dir.is_dir() or result_dir.name == 'graphs':
            continue
        
        # Check if directory name starts with a valid model prefix
        if not any(result_dir.name.startswith(prefix) for prefix in VALID_MODEL_PREFIXES):
            continue
        
        # Parse directory name: {model}_{site}_{fold}
        parts = result_dir.name.split('_')
        if len(parts) < 3:
            continue
        
        # Handle multi-part model names (e.g., deepforest_sam2)
        if len(parts) == 4:
            model = f"{parts[0]}_{parts[1]}"
            site = parts[2]
            fold = parts[3]
        else:
            model = parts[0]
            site = parts[1]
            fold = parts[2]
        
        # Load tile metrics
        tile_metrics = load_metrics(result_dir, 'tile')
        if tile_metrics:
            tile_metrics['model'] = model
            tile_metrics['site'] = site.upper()
            tile_metrics['fold'] = fold
            tile_data.append(tile_metrics)
        
        # Load raster metrics
        raster_metrics = load_metrics(result_dir, 'raster')
        if raster_metrics:
            raster_metrics['model'] = model
            raster_metrics['site'] = site.upper()
            raster_metrics['fold'] = fold
            raster_data.append(raster_metrics)
    
    tile_df = pd.DataFrame(tile_data) if tile_data else pd.DataFrame()
    raster_df = pd.DataFrame(raster_data) if raster_data else pd.DataFrame()
    
    return tile_df, raster_df


def compute_mean_by_model(df: pd.DataFrame, metric_cols: list):
    """Compute mean metrics grouped by model across all sites."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by model and compute mean for specified columns
    numeric_cols = [col for col in metric_cols if col in df.columns]
    mean_df = df.groupby('model')[numeric_cols].mean().reset_index()
    
    return mean_df

def plot_ap_ar_by_size_and_threshold(tile_df: pd.DataFrame):
    """
    Create detailed AP/AR comparison across thresholds and object sizes.
    
    Shows:
    - AP at different IoU thresholds (50, 75, overall)
    - AR at different detection limits (1, 10, 100)
    - Breakdown by object size (small, medium, large)
    """
    if tile_df.empty:
        print("No tile metrics to plot")
        return
    
    # Compute mean by model
    ap_metrics = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
    ar_metrics = ['AR', 'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
    all_metrics = ap_metrics + ar_metrics
    
    mean_df = compute_mean_by_model(tile_df, all_metrics)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    models = mean_df['model'].values
    colors = sns.color_palette("husl", len(models))
    
    # 1. AP by IoU Threshold
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, mean_df['AP50'], width, label='AP@50', color='#FF6B6B')
    bars2 = ax1.bar(x, mean_df['AP'], width, label='AP@[50:95]', color='#4ECDC4')
    bars3 = ax1.bar(x + width, mean_df['AP75'], width, label='AP@75', color='#45B7D1')
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Average Precision', fontweight='bold')
    ax1.set_title('AP Across IoU Thresholds', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    # 2. AR by Detection Limit
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(models))
    
    bars1 = ax2.bar(x - width, mean_df['AR_1'], width, label='AR@1', color='#95E1D3')
    bars2 = ax2.bar(x, mean_df['AR_10'], width, label='AR@10', color='#F38181')
    bars3 = ax2.bar(x + width, mean_df['AR_100'], width, label='AR@100', color='#AA96DA')
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Average Recall', fontweight='bold')
    ax2.set_title('AR by Max Detections', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    # 3. AP by Object Size
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(models))
    
    bars1 = ax3.bar(x - width, mean_df['AP_small'], width, label='Small', color='#FFD93D')
    bars2 = ax3.bar(x, mean_df['AP_medium'], width, label='Medium', color='#6BCB77')
    bars3 = ax3.bar(x + width, mean_df['AP_large'], width, label='Large', color='#4D96FF')
    
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Average Precision', fontweight='bold')
    ax3.set_title('AP by Object Size', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    # 4. AR by Object Size
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(models))
    
    bars1 = ax4.bar(x - width, mean_df['AR_small'], width, label='Small', color='#FFD93D')
    bars2 = ax4.bar(x, mean_df['AR_medium'], width, label='Medium', color='#6BCB77')
    bars3 = ax4.bar(x + width, mean_df['AR_large'], width, label='Large', color='#4D96FF')
    
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Average Recall', fontweight='bold')
    ax4.set_title('AR by Object Size', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    # 5. Heatmap: AP by Model and Category
    ax5 = fig.add_subplot(gs[2, :])
    
    # Prepare data for heatmap
    heatmap_data = mean_df[['model', 'AP50', 'AP', 'AP75', 'AP_small', 'AP_medium', 'AP_large']].set_index('model')
    heatmap_data.columns = ['AP@50', 'AP@[50:95]', 'AP@75', 'Small', 'Medium', 'Large']
    
    # Create heatmap
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax5)
    ax5.set_title('AP Score Heatmap: Thresholds × Object Sizes', 
                  fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel('Model', fontweight='bold')
    ax5.set_ylabel('Metric', fontweight='bold')
    
    plt.suptitle('Detailed AP/AR Analysis Across Thresholds and Object Sizes', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(GRAPHS_OUTPUT / 'ap_ar_detailed_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'ap_ar_detailed_breakdown.png'}")
    plt.close()
    
    # Also create a summary table
    create_ap_ar_summary_table(mean_df)


def create_ap_ar_summary_table(mean_df: pd.DataFrame):
    """Create a comprehensive summary table for AP/AR metrics."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for the table
    table_cols = ['model', 'AP50', 'AP', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                  'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
    table_data = mean_df[table_cols].copy()
    
    # Format numerical columns
    for col in table_cols[1:]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    
    # Rename columns for display
    display_names = {
        'model': 'Model',
        'AP50': 'AP@50', 'AP': 'AP@[50:95]', 'AP75': 'AP@75',
        'AP_small': 'AP Small', 'AP_medium': 'AP Med', 'AP_large': 'AP Large',
        'AR_1': 'AR@1', 'AR_10': 'AR@10', 'AR_100': 'AR@100',
        'AR_small': 'AR Small', 'AR_medium': 'AR Med', 'AR_large': 'AR Large'
    }
    table_data = table_data.rename(columns=display_names)
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white')
        
        # Color-code sections
        if i <= 3:  # AP threshold columns
            cell.set_facecolor('#E74C3C')
        elif i <= 6:  # AP size columns
            cell.set_facecolor('#3498DB')
        elif i <= 9:  # AR detection limit columns
            cell.set_facecolor('#16A085')
        else:  # AR size columns
            cell.set_facecolor('#8E44AD')
    
    plt.title('Complete AP/AR Metrics Summary (Mean Across Sites)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(GRAPHS_OUTPUT / 'ap_ar_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'ap_ar_summary_table.png'}")
    plt.close()

def plot_tile_metrics_comparison(tile_df: pd.DataFrame):
    """Create comparison plots for tile-level metrics."""
    if tile_df.empty:
        print("No tile metrics to plot")
        return
    
    # Compute mean by model
    key_metrics = ['AP', 'AP50', 'AP75', 'AR', 'AR50', 'AR75']
    mean_df = compute_mean_by_model(tile_df, key_metrics)
    
    # Create comparison bar plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tile-Level Metrics Comparison (Mean Across Sites)', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in mean_df.columns:
            bars = ax.bar(mean_df['model'], mean_df[metric], color=sns.color_palette("husl", len(mean_df)))
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            # Rotate x labels if needed
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_OUTPUT / 'tile_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'tile_metrics_comparison.png'}")
    plt.close()
    
    # Create detailed metrics table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the table data
    table_data = mean_df.copy()
    for col in key_metrics:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Tile-Level Metrics Summary (Mean Across Sites)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(GRAPHS_OUTPUT / 'tile_metrics_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'tile_metrics_table.png'}")
    plt.close()


def plot_raster_metrics_comparison(raster_df: pd.DataFrame):
    """Create comparison plots for raster-level metrics."""
    if raster_df.empty:
        print("No raster metrics to plot")
        return
    
    # Compute mean by model
    key_metrics = ['precision', 'recall', 'f1']
    mean_df = compute_mean_by_model(raster_df, key_metrics)
    
    # Create comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(mean_df['model']))
    width = 0.25
    
    bars1 = ax.bar(x - width, mean_df['precision'], width, label='Precision', color='#FF6B6B')
    bars2 = ax.bar(x, mean_df['recall'], width, label='Recall', color='#4ECDC4')
    bars3 = ax.bar(x + width, mean_df['f1'], width, label='F1 Score', color='#45B7D1')
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Raster-Level Metrics Comparison (Mean Across Sites)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mean_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_OUTPUT / 'raster_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'raster_metrics_comparison.png'}")
    plt.close()
    
    # Create detailed metrics table with counts
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Include counts in the table
    count_metrics = ['tp', 'fp', 'fn', 'num_truths', 'num_preds']
    all_metrics = key_metrics + count_metrics
    table_df = compute_mean_by_model(raster_df, all_metrics)
    
    # Format the table data
    table_data = table_df.copy()
    for col in key_metrics:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}')
    for col in count_metrics:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{int(x)}')
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Raster-Level Metrics Summary (Mean Across Sites)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(GRAPHS_OUTPUT / 'raster_metrics_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'raster_metrics_table.png'}")
    plt.close()
def plot_ap_ar_by_size(tile_df: pd.DataFrame):
    """
    Comparison of AP and AR across object sizes.
    Models shown as different colored bars, sizes on x-axis.
    """
    if tile_df.empty:
        print("No tile metrics to plot")
        return
    
    # Compute mean by model
    metrics = ['AP_small', 'AP_medium', 'AP_large', 'AR_small', 'AR_medium', 'AR_large']
    mean_df = compute_mean_by_model(tile_df, metrics)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = mean_df['model'].values
    sizes = ['Small', 'Medium', 'Large']
    x = np.arange(len(sizes))
    n_models = len(models)
    width = 0.8 / n_models  # Adjust width based on number of models
    
    # Generate colors for models
    colors = sns.color_palette("husl", n_models)
    
    # AP by Size
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'AP_small'].values[0],
            mean_df.loc[mean_df['model'] == model, 'AP_medium'].values[0],
            mean_df.loc[mean_df['model'] == model, 'AP_large'].values[0]
        ]
        
        bars = ax1.bar(x + offset, values, width, label=model, color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Object Size', fontweight='bold')
    ax1.set_ylabel('Average Precision', fontweight='bold')
    ax1.set_title('AP by Object Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # AR by Size
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'AR_small'].values[0],
            mean_df.loc[mean_df['model'] == model, 'AR_medium'].values[0],
            mean_df.loc[mean_df['model'] == model, 'AR_large'].values[0]
        ]
        
        bars = ax2.bar(x + offset, values, width, label=model, color=colors[i])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Object Size', fontweight='bold')
    ax2.set_ylabel('Average Recall', fontweight='bold')
    ax2.set_title('AR by Object Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('AP and AR by Object Size', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(GRAPHS_OUTPUT / 'ap_ar_by_size.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'ap_ar_by_size.png'}")
    plt.close()

def plot_combined_f1_comparison(tile_df: pd.DataFrame, raster_df: pd.DataFrame):
    """Create a combined F1 comparison across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tile-level F1 (computed from AP and AR)
    if not tile_df.empty:
        tile_mean = tile_df.groupby('model')[['AP', 'AR']].mean()
        tile_mean['F1'] = 2 * (tile_mean['AP'] * tile_mean['AR']) / (tile_mean['AP'] + tile_mean['AR'])
        tile_mean['F1'] = tile_mean['F1'].fillna(0)
        
        bars = ax1.bar(tile_mean.index, tile_mean['F1'], color=sns.color_palette("Set2", len(tile_mean)))
        ax1.set_title('Tile-Level F1 Score\n(Computed from AP×AR)', fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_xlabel('Model')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    # Raster-level F1
    if not raster_df.empty:
        raster_mean = raster_df.groupby('model')['f1'].mean()
        
        bars = ax2.bar(raster_mean.index, raster_mean.values, color=sns.color_palette("Set1", len(raster_mean)))
        ax2.set_title('Raster-Level F1 Score', fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('Model')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.suptitle('F1 Score Comparison Across Models (Mean Across Sites)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(GRAPHS_OUTPUT / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'f1_comparison.png'}")
    plt.close()


def save_summary_csv(tile_df: pd.DataFrame, raster_df: pd.DataFrame):
    """Save summary statistics to CSV files."""
    if not tile_df.empty:
        tile_summary = tile_df.groupby('model').agg({
            'AP': 'mean',
            'AP50': 'mean',
            'AP75': 'mean',
            'AR': 'mean',
            'AR50': 'mean',
            'AR75': 'mean',
            'num_images': 'sum',
            'num_truths': 'sum',
            'num_preds': 'sum'
        }).round(4)
        
        csv_path = GRAPHS_OUTPUT / 'tile_summary.csv'
        tile_summary.to_csv(csv_path)
        print(f"✓ Saved: {csv_path}")
    
    if not raster_df.empty:
        raster_summary = raster_df.groupby('model').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'tp': 'sum',
            'fp': 'sum',
            'fn': 'sum',
            'num_truths': 'sum',
            'num_preds': 'sum'
        }).round(4)
        
        csv_path = GRAPHS_OUTPUT / 'raster_summary.csv'
        raster_summary.to_csv(csv_path)
        print(f"✓ Saved: {csv_path}")

def create_instance_quality_table(mean_df: pd.DataFrame):
    """Create a detailed table for instance quality metrics."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for the table
    table_cols = [
        'model',
        'instance_Dice_mean_50', 'instance_Dice_mean_75',
        'instance_mIoU_mean_50', 'instance_mIoU_mean_75',
        'PQ_50', 'PQ_75',
        'SQ_50', 'SQ_75',
        'RQ_50', 'RQ_75',
        'mean_boundary_IoU_50', 'mean_boundary_IoU_75',
        'mean_boundary_F1_50', 'mean_boundary_F1_75'
    ]
    
    # Filter to only existing columns
    existing_cols = [col for col in table_cols if col in mean_df.columns]
    table_data = mean_df[existing_cols].copy()
    
    # Format numerical columns
    for col in existing_cols[1:]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    
    # Rename columns for display
    display_names = {
        'model': 'Model',
        'instance_Dice_mean_50': 'Dice@50',
        'instance_Dice_mean_75': 'Dice@75',
        'instance_mIoU_mean_50': 'mIoU@50',
        'instance_mIoU_mean_75': 'mIoU@75',
        'PQ_50': 'PQ@50',
        'PQ_75': 'PQ@75',
        'SQ_50': 'SQ@50',
        'SQ_75': 'SQ@75',
        'RQ_50': 'RQ@50',
        'RQ_75': 'RQ@75',
        'mean_boundary_IoU_50': 'B-IoU@50',
        'mean_boundary_IoU_75': 'B-IoU@75',
        'mean_boundary_F1_50': 'B-F1@50',
        'mean_boundary_F1_75': 'B-F1@75'
    }
    table_data = table_data.rename(columns=display_names)
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.5)
    
    # Style header row with color-coding
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        
        # Color-code sections
        if i == 0:  # Model column
            cell.set_facecolor('#2C3E50')
        elif i <= 2:  # Dice columns
            cell.set_facecolor('#E74C3C')
        elif i <= 4:  # mIoU columns
            cell.set_facecolor('#3498DB')
        elif i <= 6:  # PQ columns
            cell.set_facecolor('#16A085')
        elif i <= 10:  # SQ/RQ columns
            cell.set_facecolor('#8E44AD')
        else:  # Boundary columns
            cell.set_facecolor('#F39C12')
    
    plt.title('Instance Quality Metrics Summary (Mean Across Sites)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(GRAPHS_OUTPUT / 'instance_quality_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'instance_quality_table.png'}")
    plt.close()

def plot_instance_quality_metrics(tile_df: pd.DataFrame):
    """
    Visualize instance quality metrics: Dice, mIoU, PQ, and Boundary F1.
    X-axis: IoU thresholds (50, 75), different colored bars for each model.
    """
    if tile_df.empty:
        print("No tile metrics to plot")
        return
    
    # Define metrics to extract - CHANGED: Using Boundary F1 instead of IoU
    metrics_50 = ['instance_Dice_mean_50', 'instance_mIoU_mean_50', 'PQ_50', 'mean_boundary_F1_50']
    metrics_75 = ['instance_Dice_mean_75', 'instance_mIoU_mean_75', 'PQ_75', 'mean_boundary_F1_75']
    
    all_metrics = metrics_50 + metrics_75
    mean_df = compute_mean_by_model(tile_df, all_metrics)
    
    if mean_df.empty:
        print("No instance quality metrics found")
        return
    
    # Create figure with subplots (2x2 grid for 4 metrics)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    models = mean_df['model'].values
    n_models = len(models)
    thresholds = ['IoU@50', 'IoU@75']
    x = np.arange(len(thresholds))
    width = 0.8 / n_models
    
    # Generate colors for models
    colors = sns.color_palette("husl", n_models)
    
    # 1. Dice Score Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'instance_Dice_mean_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_Dice_mean_75'].values[0]
        ]
        
        bars = ax1.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax1.set_xlabel('IoU Threshold', fontweight='bold')
    ax1.set_ylabel('Dice Score', fontweight='bold')
    ax1.set_title('Instance Dice Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. mIoU Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'instance_mIoU_mean_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_mIoU_mean_75'].values[0]
        ]
        
        bars = ax2.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax2.set_xlabel('IoU Threshold', fontweight='bold')
    ax2.set_ylabel('mIoU', fontweight='bold')
    ax2.set_title('Instance Mean IoU', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(thresholds)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. PQ (Panoptic Quality) Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'PQ_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'PQ_75'].values[0]
        ]
        
        bars = ax3.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax3.set_xlabel('IoU Threshold', fontweight='bold')
    ax3.set_ylabel('PQ Score', fontweight='bold')
    ax3.set_title('Panoptic Quality (PQ = SQ × RQ)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(thresholds)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Boundary F1 Comparison (CHANGED from Boundary IoU)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'mean_boundary_F1_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'mean_boundary_F1_75'].values[0]
        ]
        
        bars = ax4.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax4.set_xlabel('IoU Threshold', fontweight='bold')
    ax4.set_ylabel('Boundary F1', fontweight='bold')
    ax4.set_title('Boundary F1 Score', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(thresholds)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Instance Quality Metrics: Models Comparison at IoU@50 vs IoU@75', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(GRAPHS_OUTPUT / 'instance_quality_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'instance_quality_metrics.png'}")
    plt.close()
    
    # Create a comprehensive summary table
    create_instance_quality_table(mean_df)


def plot_instance_quality_by_size(tile_df: pd.DataFrame):
    """
    Visualize instance quality metrics broken down by object size.
    Shows Dice, mIoU, PQ, and Boundary F1 for small/medium/large objects.
    X-axis: Object sizes, different colored bars for each model.
    """
    if tile_df.empty:
        print("No tile metrics to plot")
        return
    
    # Define metrics to extract (using IoU@50 for size breakdown)
    # CHANGED: Using Boundary F1 instead of IoU (if available)
    metrics_dice = ['instance_Dice_small_50', 'instance_Dice_medium_50', 'instance_Dice_large_50']
    metrics_miou = ['instance_mIoU_small_50', 'instance_mIoU_medium_50', 'instance_mIoU_large_50']
    metrics_pq = ['PQ_small_50', 'PQ_medium_50', 'PQ_large_50']
    # Note: If boundary_F1 by size doesn't exist, this will be skipped in the plot
    
    all_metrics = metrics_dice + metrics_miou + metrics_pq
    mean_df = compute_mean_by_model(tile_df, all_metrics)
    
    if mean_df.empty:
        print("No size-based instance quality metrics found")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    models = mean_df['model'].values
    n_models = len(models)
    sizes = ['Small', 'Medium', 'Large']
    x = np.arange(len(sizes))
    width = 0.8 / n_models
    
    # Generate colors for models
    colors = sns.color_palette("husl", n_models)
    
    # 1. Dice Score by Size
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'instance_Dice_small_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_Dice_medium_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_Dice_large_50'].values[0]
        ]
        
        bars = ax1.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax1.set_xlabel('Object Size', fontweight='bold')
    ax1.set_ylabel('Dice Score', fontweight='bold')
    ax1.set_title('Instance Dice Score by Size (IoU@50)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. mIoU by Size
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'instance_mIoU_small_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_mIoU_medium_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'instance_mIoU_large_50'].values[0]
        ]
        
        bars = ax2.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax2.set_xlabel('Object Size', fontweight='bold')
    ax2.set_ylabel('mIoU', fontweight='bold')
    ax2.set_title('Instance Mean IoU by Size (IoU@50)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. PQ by Size
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, model in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        values = [
            mean_df.loc[mean_df['model'] == model, 'PQ_small_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'PQ_medium_50'].values[0],
            mean_df.loc[mean_df['model'] == model, 'PQ_large_50'].values[0]
        ]
        
        bars = ax3.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    ax3.set_xlabel('Object Size', fontweight='bold')
    ax3.set_ylabel('PQ Score', fontweight='bold')
    ax3.set_title('Panoptic Quality by Size (IoU@50)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Empty subplot (REMOVED Boundary IoU by size - not typically computed per size)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0.5, 0.5, 'Boundary F1 by size\nnot computed in current metrics', 
             ha='center', va='center', fontsize=12, style='italic', color='gray')
    
    plt.suptitle('Instance Quality Metrics by Object Size (Small/Medium/Large)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(GRAPHS_OUTPUT / 'instance_quality_by_size.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'instance_quality_by_size.png'}")
    plt.close()
    
    # Also create size breakdown table
    create_instance_quality_size_table(mean_df)


def create_instance_quality_size_table(mean_df: pd.DataFrame):
    """Create a detailed table for instance quality metrics broken down by size."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for the table (REMOVED boundary_IoU columns)
    table_cols = [
        'model',
        'instance_Dice_small_50', 'instance_Dice_medium_50', 'instance_Dice_large_50',
        'instance_mIoU_small_50', 'instance_mIoU_medium_50', 'instance_mIoU_large_50',
        'PQ_small_50', 'PQ_medium_50', 'PQ_large_50'
    ]
    
    # Filter to only existing columns
    existing_cols = [col for col in table_cols if col in mean_df.columns]
    table_data = mean_df[existing_cols].copy()
    
    # Format numerical columns
    for col in existing_cols[1:]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    
    # Rename columns for display
    display_names = {
        'model': 'Model',
        'instance_Dice_small_50': 'Dice S',
        'instance_Dice_medium_50': 'Dice M',
        'instance_Dice_large_50': 'Dice L',
        'instance_mIoU_small_50': 'mIoU S',
        'instance_mIoU_medium_50': 'mIoU M',
        'instance_mIoU_large_50': 'mIoU L',
        'PQ_small_50': 'PQ S',
        'PQ_medium_50': 'PQ M',
        'PQ_large_50': 'PQ L'
    }
    table_data = table_data.rename(columns=display_names)
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    
    # Style header row with color-coding
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        
        # Color-code sections
        if i == 0:  # Model column
            cell.set_facecolor('#2C3E50')
        elif i <= 3:  # Dice columns
            cell.set_facecolor('#E74C3C')
        elif i <= 6:  # mIoU columns
            cell.set_facecolor('#3498DB')
        else:  # PQ columns
            cell.set_facecolor('#16A085')
    
    plt.title('Instance Quality Metrics by Size (IoU@50) - Mean Across Sites', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(GRAPHS_OUTPUT / 'instance_quality_by_size_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {GRAPHS_OUTPUT / 'instance_quality_by_size_table.png'}")
    plt.close()

def main():
    print("="*80)
    print("MODEL COMPARISON VISUALIZATION")
    print(f"Processing only folders starting with: {', '.join(VALID_MODEL_PREFIXES)}")
    print("="*80)
    
    # Collect all metrics
    print("\nCollecting metrics from results folder...")
    tile_df, raster_df = collect_all_metrics()
    
    print(f"Found {len(tile_df)} tile-level results")
    print(f"Found {len(raster_df)} raster-level results")
    
    if tile_df.empty and raster_df.empty:
        print("\n❌ No metrics found in results folder!")
        print(f"Make sure you have folders starting with: {', '.join(VALID_MODEL_PREFIXES)}")
        return
    
    # Generate visualizations
    print("\nGenerating comparison graphs...")
    
    if not tile_df.empty:
        plot_tile_metrics_comparison(tile_df)
        plot_ap_ar_by_size(tile_df)  # Add this line
        plot_ap_ar_by_size_and_threshold(tile_df)
        plot_instance_quality_metrics(tile_df)  # ✅ NEW!
        plot_instance_quality_by_size(tile_df) 
    
    if not raster_df.empty:
        plot_raster_metrics_comparison(raster_df)
    
    if not tile_df.empty or not raster_df.empty:
        plot_combined_f1_comparison(tile_df, raster_df)
    
    # Save summary CSVs
    print("\nSaving summary statistics...")
    save_summary_csv(tile_df, raster_df)
    
    print("\n" + "="*80)
    print(f"✓ ALL GRAPHS SAVED TO: {GRAPHS_OUTPUT}")
    print("="*80)


if __name__ == "__main__":
    main()