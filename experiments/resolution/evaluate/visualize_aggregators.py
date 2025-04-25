import pandas as pd

def print_best_results(df, model_name):
    # Filter rows where raster_name is "average_over_rasters"
    filtered_df = df[df['raster_name'] == "average_over_rasters"]
    
    if filtered_df.empty:
        print(f"No rows with raster_name 'average_over_rasters' found for {model_name}.\n")
        return
    
    # Find the row with the maximum 'F1' score within the filtered dataframe
    best_row = filtered_df.loc[filtered_df['F1'].idxmax()]
    
    # Print header for clarity
    print(f"Best results for {model_name} (raster_name: average_over_rasters):")
    print("-" * 50)
    
    # Print each column and its value in a nicely formatted way
    for col, val in best_row.items():
        print(f"{col:25s}: {val}")
    print("\n")

if __name__ == "__main__":
    # Set the extent and the raster metric to display ("AP", "AR", or "F1")
    extent = '80m'
    raster_metric = "F1"  # Here we're using F1 since we're comparing by 'F1'
    root = '/network/scratch/h/hugo.baudchon'
    
    # Load CSVs
    faster_rcnn_results_aggregator = pd.read_csv(
        f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED_new/faster_rcnn_R_50_FPN_3x/aggregator_search_results_valid_fold.csv'
    )
    dino_resnet_results_aggregator = pd.read_csv(
        f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED_new/dino_r50_4scale_24ep/aggregator_search_results_valid_fold.csv'
    )
    dino_swin_results_aggregator = pd.read_csv(
        f'{root}/eval/detector_experience_resolution_optimalHPs_{extent}_FIXED_new/dino_swin_large_384_5scale_36ep/aggregator_search_results_valid_fold.csv'
    )
    
    # Optionally, print the columns to verify the structure
    print("Faster R-CNN columns:", faster_rcnn_results_aggregator.columns.tolist())
    print("Dino ResNet columns:", dino_resnet_results_aggregator.columns.tolist())
    print("Dino Swin columns:", dino_swin_results_aggregator.columns.tolist())
    
    # Print the best rows (based on 'F1') for each model where raster_name is "average_over_rasters"
    print_best_results(faster_rcnn_results_aggregator, "Faster R-CNN")
    print_best_results(dino_resnet_results_aggregator, "Dino ResNet")
    print_best_results(dino_swin_results_aggregator, "Dino Swin")
