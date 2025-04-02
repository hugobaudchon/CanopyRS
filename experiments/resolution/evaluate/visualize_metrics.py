import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def get_variant_color(base_color, total, index, range_l=0.15, range_s=0.5):
    """
    Generate a variant of the base_color by adjusting its lightness and saturation.
    - The darkest variant (index==0) is the base color.
    - The lightest variant (index==total-1) has its lightness increased by up to range_l
      and its saturation reduced by up to range_s.
    """
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if total > 1:
        factor = index / (total - 1)
    else:
        factor = 0
    new_l = l + factor * range_l
    new_s = s * (1 - factor * range_s)
    new_l = min(max(new_l, 0), 1)
    new_s = min(max(new_s, 0), 1)
    return colorsys.hls_to_rgb(h, new_l, new_s)

if __name__ == "__main__":
    # Set the extent and the raster metric to display ("AP", "AR", or "F1")
    extent = '80m'
    raster_metric = "AR"  # Change to "AR" or "F1" as needed
    root = '/network/scratch/h/hugo.baudchon'
    
    # Load CSVs
    faster_rcnn_results_tile = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/faster_rcnn_R_50_FPN_3x/test_tile_level_metrics.csv')
    faster_rcnn_results_raster = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/faster_rcnn_R_50_FPN_3x/test_raster_level_metrics.csv')
    dino_resnet_results_tile = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/dino_r50_4scale_24ep/test_tile_level_metrics.csv')
    dino_resnet_results_raster = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/dino_r50_4scale_24ep/test_raster_level_metrics.csv')
    dino_swin_results_tile = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/dino_swin_large_384_5scale_36ep/test_tile_level_metrics.csv')
    dino_swin_results_raster = pd.read_csv(f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED/dino_swin_large_384_5scale_36ep/test_raster_level_metrics.csv')
    
    # Define the base colors for each architecture.
    base_colors = {
        'faster_rcnn': 'blue',
        'dino_resnet': 'green',
        'dino_swin': 'red'
    }
    
    # Define linestyles and markers for each resolution.
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    # Prepare tile extent string: e.g. "80m" becomes "80X80m"
    tile_extent = extent.replace("m", "", 1) + "X" + extent
    
    # ----------------------------
    # Plot Tile Level Graph
    # ----------------------------
    plt.figure(figsize=(10, 6))
    
    # Re-order the dictionary to show dinoswin first, then dino_resnet, and finally faster_rcnn.
    tile_dfs = {
        'dino_swin': dino_swin_results_tile,
        'dino_resnet': dino_resnet_results_tile,
        'faster_rcnn': faster_rcnn_results_tile
    }
    
    for arch, df in tile_dfs.items():
        # For tile level, we already have product_name == 'all'
        df_tile = df[df['product_name'] == 'all']
        grouped = df_tile.groupby(['augmentation_image_size', 'ground_resolution']).agg(
            mean_AP=('AP', 'mean'),
            std_AP=('AP', 'std')
        ).reset_index()
    
        unique_ground_res = sorted(grouped['ground_resolution'].unique())
        num_groups = len(unique_ground_res)
    
        for i, gr in enumerate(unique_ground_res):
            df_grp = grouped[grouped['ground_resolution'] == gr].sort_values('augmentation_image_size')
            color_variant = get_variant_color(base_colors[arch], num_groups, i)
            marker = markers[i % len(markers)]
            label = f"{arch} @ {gr}"
            plt.errorbar(df_grp['augmentation_image_size'], df_grp['mean_AP'],
                         yerr=df_grp['std_AP'],
                         color=color_variant,
                         linestyle=linestyles[i % len(linestyles)],
                         marker=marker,
                         label=label)
    
    plt.xlabel("image size")
    plt.ylabel("mAP")
    plt.title(f"Tile Level mAP vs Image Size ({tile_extent})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Architecture @ Ground Resolution", framealpha=1)
    plt.tight_layout()
    plt.savefig(f"tile_level_map_{extent}.png")
    plt.close()
    
    # ----------------------------
    # Plot Raster Level Graph with Error Bars
    # ----------------------------
    plt.figure(figsize=(10, 6))
    
    # Re-order the dictionary to show dinoswin first, then dino_resnet, and finally faster_rcnn.
    raster_dfs = {
        'dino_swin': dino_swin_results_raster,
        'dino_resnet': dino_resnet_results_raster,
        'faster_rcnn': faster_rcnn_results_raster
    }
    
    # For each model, first compute a weighted average over product_name rows (for each seed)
    # then average over seeds.
    for arch, df in raster_dfs.items():
        # Step 1: Group by augmentation_image_size, ground_resolution, and seed,
        #         and compute the weighted average over product_name rows.
        temp = df.groupby(['augmentation_image_size', 'ground_resolution', 'seed']).apply(
            lambda x: (x[raster_metric] * x['num_truths']).sum() / x['num_truths'].sum()
        ).reset_index(name=f'{raster_metric}_all')
        
        # Step 2: Group by augmentation_image_size and ground_resolution (averaging over seeds)
        grouped = temp.groupby(['augmentation_image_size', 'ground_resolution']).agg(
            mean=(f'{raster_metric}_all', 'mean'),
            std=(f'{raster_metric}_all', 'std')
        ).reset_index()
    
        unique_ground_res = sorted(grouped['ground_resolution'].unique())
        num_groups = len(unique_ground_res)
    
        for i, gr in enumerate(unique_ground_res):
            df_grp = grouped[grouped['ground_resolution'] == gr].sort_values('augmentation_image_size')
            color_variant = get_variant_color(base_colors[arch], num_groups, i)
            marker = markers[i % len(markers)]
            label = f"{arch} @ {gr}"
            plt.errorbar(df_grp['augmentation_image_size'], df_grp['mean'],
                         yerr=df_grp['std'],
                         color=color_variant,
                         linestyle=linestyles[i % len(linestyles)],
                         marker=marker,
                         label=label)
    
    plt.xlabel("image size")
    plt.ylabel(f"m{raster_metric}")
    plt.title(f"Raster Level Weighted m{raster_metric} vs Image Size ({tile_extent})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Architecture @ Ground Resolution", framealpha=1)
    plt.tight_layout()
    plt.savefig(f"raster_level_map_{extent}_{raster_metric}.png")
    plt.close()
