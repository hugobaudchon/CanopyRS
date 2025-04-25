from pathlib import Path
import pandas as pd

# … your existing imports and get_variant_color, plotting code …

if __name__ == "__main__":
    extent = '80m'
    root   = Path('/network/scratch/h/hugo.baudchon')
    base   = root / 'eval' / f'detector_experience_resolution_optimalHPs_{extent}_FIXED_new5'

    model_dirs = {
        'faster_rcnn': 'faster_rcnn_R_50_FPN_3x',
        'dino_resnet': 'dino_r50_4scale_24ep',
        'dino_swin':   'dino_swin_large_384_5scale_36ep',
    }

    # tile- and raster-level dicts
    tile_dfs   = {}
    raster_dfs = {}

    for arch, subdir in model_dirs.items():
        folder = base / subdir
        tile_csv   = folder / 'test_tile_level_metrics.csv'
        raster_csv = folder / 'test_raster_level_metrics.csv'

        # sanity check
        if not tile_csv.exists():
            raise FileNotFoundError(f"Can't find {tile_csv}")
        if not raster_csv.exists():
            raise FileNotFoundError(f"Can't find {raster_csv}")

        tile_dfs[arch]   = pd.read_csv(tile_csv)
        raster_dfs[arch] = pd.read_csv(raster_csv)

        if arch == 'dino_resnet':
            print(8888, raster_dfs[arch][raster_dfs[arch]['model_name'] == 'dino_detrex_20250327_065028_712633_6444522'])

    # 1) Build tile‐level summary
    tile_summary = []
    for arch, df in tile_dfs.items():
        df_all = df[df['product_name']=='all']
        g = df_all.groupby(['augmentation_image_size','ground_resolution']).agg(
            tile_AP_mean = ('AP','mean'),
            tile_AP_std  = ('AP','std'),
            tile_AR_mean = ('AR','mean'),
            tile_AR_std  = ('AR','std'),
            tile_AP50_mean = ('AP50','mean'),
            tile_AP50_std  = ('AP50','std'),
            tile_AR50_mean = ('AR50','mean'),
            tile_AR50_std  = ('AR50','std'),
        ).reset_index()
        g['architecture'] = arch
        g['extent']       = extent
        tile_summary.append(g)
    tile_summary = pd.concat(tile_summary, ignore_index=True)

    # 2) Build raster‐level summary
    raster_summary = []
    for arch, df in raster_dfs.items():
        print(df.columns)
        print(arch, df[['product_name', 'augmentation_image_size', 'ground_resolution', 'seed', 'AP', 'AR', 'num_truths']])
        # for each metric, do weighted seed→mean/std
        stats = None
        for metric in ['AP','AR','AP50','AR50', 'AP_small', 'AP_medium', 'AP_large']:
            tmp = (
                df
                .groupby(['augmentation_image_size','ground_resolution','seed'])
                .apply(lambda x: (x[metric]*x['num_truths']).sum()/x['num_truths'].sum())
                .reset_index(name=f'{metric}_per_image')
            )
            agg = (
                tmp
                .groupby(['augmentation_image_size','ground_resolution'])
                .agg(
                    **{f'raster_{metric}_mean': (f'{metric}_per_image','mean'),
                       f'raster_{metric}_std':  (f'{metric}_per_image','std')}
                )
                .reset_index()
            )
            stats = agg if stats is None else stats.merge(agg, on=['augmentation_image_size','ground_resolution'])
        stats['architecture'] = arch
        stats['extent']       = extent
        raster_summary.append(stats)
    raster_summary = pd.concat(raster_summary, ignore_index=True)

    # 3) Merge tile + raster summaries
    summary = (
        tile_summary
        .merge(raster_summary,
               on=['architecture','extent','augmentation_image_size','ground_resolution'],
               how='outer')
    )

    # 4) write out
    summary.to_csv(f'all_metrics_summary_{extent}.csv', index=False)
