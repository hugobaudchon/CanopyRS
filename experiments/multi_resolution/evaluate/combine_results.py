import pandas as pd

img_size = 1333
# root_path = f'/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_cropprob1p0_gridsearch_{img_size}'
root_path1 = f'/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_imgsize1777_BETTER_{img_size}'
root_path2 = f'/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_multi_res_models_imgsize1777_{img_size}'

extent_to_dfs = {
    '36_88m': {
        'tile': f'{root_path1}/34_88m/tile_level_summary.csv',
        'raster': f'{root_path1}/34_88m/raster_level_summary.csv'
    },
    '30_100m': {
        'tile': f'{root_path1}/30_100m/tile_level_summary.csv',
        'raster': f'{root_path1}/30_100m/raster_level_summary.csv'
    },
    '30_120m': {
        'tile': f'{root_path1}/30_120m/tile_level_summary.csv',
        'raster': f'{root_path1}/30_120m/raster_level_summary.csv'
    },

    '4.5cm': {
        'tile': f'{root_path2}/singleres_1777_1777/tile_level_summary.csv',
        'raster': f'{root_path2}/singleres_1777_1777/raster_level_summary.csv'
    },
    '6cm': {
        'tile': f'{root_path2}/singleres_1333_1777/tile_level_summary.csv',
        'raster': f'{root_path2}/singleres_1333_1777/raster_level_summary.csv'
    },
    '10cm': {
        'tile': f'{root_path2}/singleres_800_1777/tile_level_summary.csv',
        'raster': f'{root_path2}/singleres_800_1777/raster_level_summary.csv'
    }
}

output_df_path = f'./combined_summary_multires_single_res_models_{img_size}.csv'


combined_dfs = []
for extent, paths in extent_to_dfs.items():
    df_tile = pd.read_csv(paths['tile'])
    df_raster = pd.read_csv(paths['raster'])

    df_tile['augment_extent'] = extent

    df_tile.drop(columns=['num_images', 'num_truths', 'num_preds', 'AR_1', 'AR_10', 'AR_100'], inplace=True)
    df_raster.drop(columns=['num_images', 'num_truths', 'num_preds'], inplace=True)

    print(df_tile.columns)
    print(df_raster.columns)

    metrics = [
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR', 'AR50', 'AR75', 'AR_small', 'AR_medium', 'AR_large',
        'F1', 'F1_50', 'F1_75', 'F1_small', 'F1_medium', 'F1_large',
        'precision', 'recall', 'f1'
    ]

    for metric in metrics:
        for stat in ['mean', 'std']:
            metric_stat = '{}_{}'.format(metric, stat)
            if metric_stat in df_tile.columns:
                df_tile.rename(columns={metric_stat: f"tile_{metric_stat}"}, inplace=True)
            if metric_stat in df_raster.columns:
                df_raster.rename(columns={metric_stat: f"raster_{metric_stat}"}, inplace=True)

    combined_df = df_tile.merge(df_raster, on=['location', 'product_name'], how='inner')

    print(combined_df.columns)
    print(len(combined_df))
    
    combined_dfs.append(combined_df)
    
final_combined_df = pd.DataFrame(pd.concat(combined_dfs, ignore_index=True))
final_combined_df.to_csv(output_df_path, index=False)

cols = ['augment_extent',
        'tile_AP_mean', 'tile_AP_std',
        'tile_AR_mean', 'tile_AR_std',
        'raster_f1_mean', 'raster_f1_std']

selva_df = final_combined_df.loc[
    final_combined_df['location'] == 'SelvaBox',
    cols
]

# define which cols to show as percentages
pct_cols = [c for c in cols if c != 'augment_extent']

# build a dict of formatter functions
formatters = {
    c: (lambda x: f"{x*100:.2f}%")
    for c in pct_cols
}

# print, leaving augment_extent as-is, all others as 2-dec percentage
print(
    selva_df.to_string(
        index=False,
        formatters=formatters
    )
)