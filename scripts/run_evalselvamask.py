#!/usr/bin/env python3
# filepath: /home/soduguay/selvamask/CanopyRS/run_evalselvamask.py
"""
Standalone script to evaluate SelvaMask segmentation predictions.
Usage: python run_evalselvamask.py --site BCI --model dino_sam2
"""
import argparse
from pathlib import Path
import json
from engine.benchmark.detector.evaluator import CocoEvaluator


# =========================
# CONFIGURATION
# =========================

SITES = {
    'BCI': {
        'product_name': '20241122_bcifairchildn_m3m_rgb',
        'location': 'panama_bcifairchildn'
    },
    'TBS': {
        'product_name': '20240613_tbsnewsite2_m3e_rgb',
        'location': 'ecuador_tbsnewsite2'
    },
    'ZF2': {
        'product_name': '20240131_zf2block4_ms_m3m_rgb',
        'location': 'brazil_zf2block4'
    }
}

MODELS = {
    'dino_sam2': '/data/soduguay/selvamask/selvamask_benchmark_dino_sam2',
    'deepforest_sam2': '/data/soduguay/selvamask/selvamask_benchmark_deepforest_sam2',
    'detectree2': '/data/soduguay/selvamask/selvamask_benchmark_detectree2',
    'detectree2_220723_withParacouUAV': '/data/soduguay/selvamask/selvamask_benchmark_detectree2_220723_withParacouUAV',
    'detectree2_230103_randresize_full': '/data/soduguay/selvamask/selvamask_benchmark_detectree2_230103_randresize_full',
    'detectree2_230717_base': '/data/soduguay/selvamask/selvamask_benchmark_detectree2_230717_base',
    'detectree2_230729_05dates': '/data/soduguay/selvamask/selvamask_benchmark_detectree2_230729_05dates'
}

TRUTH_BASE = Path('../selvamask')
RESULTS_BASE = Path('results')


def get_paths(site: str, model: str, fold: str = 'test'):
    """Generate all required paths for evaluation."""
    site_info = SITES[site.upper()]
    product = site_info['product_name']
    model_path = Path(MODELS[model])
    
    # Determine product-specific paths based on naming conventions
    truth_labels_name = product.replace('_rgb', '_labels_masks')
    tile_preds_folder = "1_segmenter"
    raster_preds_folder = "2_aggregator"
    if 'detectree2' in model:
        tile_preds_folder = "0_segmenter"
        raster_preds_folder = "1_aggregator"
        
    paths = {
        # Predictions
        'tile_preds_coco': model_path / tile_preds_folder / f'{product}_tile_coco_gr0p045_infer.json',
        'raster_preds_gpkg': model_path / raster_preds_folder / f'{product}_tile_gr0p045_infer.gpkg',
        
        # Ground truth
        'tile_truth_coco': TRUTH_BASE / product / f'{product}_coco_gr0p045_{fold}.json',
        'raster_truth_gpkg': TRUTH_BASE / product / f'{truth_labels_name}.gpkg',
        'raster_aoi_gpkg': TRUTH_BASE / product / f'{product}_aoi_gr0p045_{fold}.gpkg',
        
        # Outputs
        'tile_output_json': RESULTS_BASE / f'{model}_{site.lower()}_{fold}' / 'tile_metrics.json',
        'raster_output_json': RESULTS_BASE / f'{model}_{site.lower()}_{fold}' / 'raster_metrics.json',
    }
    
    return paths, site_info


def convert_metrics_for_json(metrics):
    """Convert numpy types to native Python types for JSON serialization."""
    json_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            json_metrics[key] = value.item()
        elif isinstance(value, (int, float, str, bool, type(None))):
            json_metrics[key] = value
        else:
            json_metrics[key] = str(value)
    return json_metrics


def evaluate_tile_level(iou_type, preds_coco, truth_coco, max_dets, output_json=None):
    """Evaluate predictions at tile level using COCO JSON files."""
    print("\n=== Tile-Level Evaluation ===")
    print(f"Predictions: {preds_coco}")
    print(f"Ground Truth: {truth_coco}")
    
    if not preds_coco.exists():
        print(f"ERROR: Predictions file not found: {preds_coco}")
        return None
    if not truth_coco.exists():
        print(f"ERROR: Ground truth file not found: {truth_coco}")
        return None
    
    evaluator = CocoEvaluator()
    
    metrics = evaluator.tile_level(
        iou_type=iou_type,
        preds_coco_path=str(preds_coco),
        truth_coco_path=str(truth_coco),
        max_dets=max_dets
    )
    
    print("\n--- Results ---")
    print(f"Images: {metrics['num_images']} | Truth: {metrics['num_truths']} | Preds: {metrics['num_preds']}")
    print(f"AP@50:95: {metrics['AP']:.4f} | AP@50: {metrics['AP50']:.4f} | AP@75: {metrics['AP75']:.4f}")
    print(f"AR@{max_dets[-1]}: {metrics['AR']:.4f} | AR@50: {metrics['AR50']:.4f} | AR@75: {metrics['AR75']:.4f}")
    
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(convert_metrics_for_json(metrics), f, indent=2)
        print(f"✓ Saved to: {output_json}")
    
    return metrics


def evaluate_raster_level(iou_type, preds_gpkg, truth_gpkg, aoi_gpkg, 
                          ground_resolution, single_iou_threshold, output_json=None):
    """Evaluate predictions at raster level using GeoPackage files."""
    print("\n=== Raster-Level Evaluation ===")
    print(f"Predictions: {preds_gpkg}")
    print(f"Ground Truth: {truth_gpkg}")
    print(f"AOI: {aoi_gpkg}")
    
    if not preds_gpkg.exists():
        print(f"ERROR: Predictions file not found: {preds_gpkg}")
        return None
    if not truth_gpkg.exists():
        print(f"ERROR: Ground truth file not found: {truth_gpkg}")
        return None
    if aoi_gpkg and not aoi_gpkg.exists():
        print(f"ERROR: AOI file not found: {aoi_gpkg}")
        return None
    
    evaluator = CocoEvaluator()
    
    if single_iou_threshold is not None:
        print(f"Single IoU threshold: {single_iou_threshold}")
        
        metrics = evaluator.raster_level_single_iou_threshold(
            iou_type=iou_type,
            preds_gpkg_path=str(preds_gpkg),
            truth_gpkg_path=str(truth_gpkg),
            aoi_gpkg_path=str(aoi_gpkg) if aoi_gpkg else None,
            ground_resolution=ground_resolution,
            iou_threshold=single_iou_threshold
        )
        
        print("\n--- Results ---")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
        
    else:
        print("Full COCO evaluation across IoU thresholds 0.50:0.95")
        
        metrics = evaluator.raster_level(
            iou_type=iou_type,
            preds_gpkg_path=str(preds_gpkg),
            truth_gpkg_path=str(truth_gpkg),
            aoi_gpkg_path=str(aoi_gpkg) if aoi_gpkg else None,
            ground_resolution=ground_resolution
        )
        
        print("\n--- Results ---")
        print(f"Truth: {metrics['num_truths']} | Preds: {metrics['num_preds']}")
        print(f"AP@50:95: {metrics['AP']:.4f} | AP@50: {metrics['AP50']:.4f} | AP@75: {metrics['AP75']:.4f}")
        print(f"AR: {metrics['AR']:.4f} | AR@50: {metrics['AR50']:.4f} | AR@75: {metrics['AR75']:.4f}")
        print(f"F1@50:95: {metrics['F1']:.4f} | F1@50: {metrics['F1_50']:.4f} | F1@75: {metrics['F1_75']:.4f}")
    
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(convert_metrics_for_json(metrics), f, indent=2)
        print(f"✓ Saved to: {output_json}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SelvaMask predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available sites: {', '.join(SITES.keys())}
Available models: {', '.join(MODELS.keys())}

Examples:
  python run_evalselvamask.py --site BCI --model dino_sam2
  python run_evalselvamask.py --site TBS --model deepforest_sam2 --fold valid
  python run_evalselvamask.py --site ZF2 --model detectree2 --no-tile --iou 0.5
        """
    )
    
    parser.add_argument('--site', type=str, required=True, choices=SITES.keys(),
                       help='Site to evaluate (BCI, TBS, or ZF2)')
    parser.add_argument('--model', type=str, required=True, choices=MODELS.keys(),
                       help='Model to evaluate')
    parser.add_argument('--fold', type=str, default='test', choices=['test', 'valid'],
                       help='Data fold to evaluate (default: test)')
    parser.add_argument('--iou-type', type=str, default='segm', choices=['segm', 'bbox'],
                       help='IoU type (default: segm)')
    parser.add_argument('--iou', type=float, default=0.50,
                       help='Single IoU threshold for raster evaluation (default: 0.75)')
    parser.add_argument('--ground-res', type=float, default=0.045,
                       help='Ground resolution in meters (default: 0.045)')
    parser.add_argument('--max-dets', type=int, nargs='+', default=[1, 10, 100, 400],
                       help='Max detections for tile evaluation (default: 1 10 100 400)')
    parser.add_argument('--no-tile', action='store_true',
                       help='Skip tile-level evaluation')
    parser.add_argument('--no-raster', action='store_true',
                       help='Skip raster-level evaluation')
    
    args = parser.parse_args()
    
    # Get all paths
    paths, site_info = get_paths(args.site, args.model, args.fold)
    
    print("\n" + "="*80)
    print(f"EVALUATING: {args.model.upper()} on {args.site} ({site_info['location']}) - {args.fold} fold")
    print("="*80)
    
    # Tile-level evaluation
    if not args.no_tile:
        print("\n" + "="*60)
        print("TILE-LEVEL EVALUATION")
        print("="*60)
        evaluate_tile_level(
            iou_type=args.iou_type,
            preds_coco=paths['tile_preds_coco'],
            truth_coco=paths['tile_truth_coco'],
            max_dets=args.max_dets,
            output_json=paths['tile_output_json']
        )
    
    # Raster-level evaluation
    if not args.no_raster:
        print("\n" + "="*60)
        print("RASTER-LEVEL EVALUATION")
        print("="*60)
        evaluate_raster_level(
            iou_type=args.iou_type,
            preds_gpkg=paths['raster_preds_gpkg'],
            truth_gpkg=paths['raster_truth_gpkg'],
            aoi_gpkg=paths['raster_aoi_gpkg'],
            ground_resolution=args.ground_res,
            single_iou_threshold=args.iou,
            output_json=paths['raster_output_json']
        )
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()