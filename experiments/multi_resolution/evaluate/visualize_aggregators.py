import os
import pandas as pd

def get_best_nms(df):
    # keep only the aggregate-over-rasters row
    df_agg = df[df['raster_name'] == "average_over_rasters"]
    if df_agg.empty:
        return None
    best = df_agg.loc[df_agg['f1'].idxmax()]
    return best['nms_iou_threshold'], best['nms_score_threshold']

if __name__ == "__main__":
    root = '/network/scratch/h/hugo.baudchon/eval/detector_experience_multi_resolution_NEW_METRIC_800'
    subfolders = ['30_100m', '30_120m', '34_88m']

    # LaTeX table header
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Subfolder & $\tau_{\mathrm{nms}}$ & $s_{\min}$ \\")
    print(r"\midrule")

    for sub in subfolders:
        valid_dir = os.path.join(root, sub, 'valid')
        # find the single subfolder inside 'valid'
        inner_dirs = [d for d in os.listdir(valid_dir)
                      if os.path.isdir(os.path.join(valid_dir, d))]
        if not inner_dirs:
            print(f"{sub} & N/A & N/A \\\\")
            continue

        model_name = inner_dirs[0]
        csv_path = os.path.join(valid_dir, model_name, 'valid/optimal_nms_iou_threshold_search.csv')

        df = pd.read_csv(csv_path)
        hps = get_best_nms(df)
        if hps is None:
            print(f"{sub} & N/A & N/A \\\\")
        else:
            tau_nms, s_min = hps
            print(f"{sub} & {tau_nms:.2f} & {s_min:.2f} \\\\")

    # LaTeX table footer
    print(r"\bottomrule")
    print(r"\end{tabular}")
