# engine/benchmark/classifier/evaluator.py
import json
from typing import Optional
from pathlib import Path
from faster_coco_eval.core.coco import COCO

class ClassifierCocoEvaluator:
    def __init__(self):
        pass
    
    def inspect_coco_files(self, preds_coco_path: str, truth_coco_path: str):
        """
        Inspect COCO files to verify they're correctly formatted for evaluation.
        This is for testing/debugging before implementing full evaluation.
        """
        print("="*60)
        print("COCO FILES INSPECTION")
        print("="*60)
        
        # Load COCO files
        try:
            truth_coco = COCO(str(truth_coco_path))
            preds_coco = COCO(str(preds_coco_path))
            print("✅ Successfully loaded both COCO files")
        except Exception as e:
            print(f"❌ Error loading COCO files: {e}")
            return False
        
        # Inspect ground truth
        print(f"\n📋 GROUND TRUTH COCO ({Path(truth_coco_path).name}):")
        self._inspect_single_coco(truth_coco, "Ground Truth")
        
        # Inspect predictions
        print(f"\n🔍 PREDICTIONS COCO ({Path(preds_coco_path).name}):")
        self._inspect_single_coco(preds_coco, "Predictions")
        
        # Cross-validation
        print(f"\n🔄 CROSS-VALIDATION:")
        self._cross_validate_cocos(truth_coco, preds_coco)
        
        return True
    
    def _inspect_single_coco(self, coco_obj: COCO, label: str):
        """Inspect a single COCO object"""
        dataset = coco_obj.dataset
        
        # Basic counts
        num_images = len(dataset.get('images', []))
        num_annotations = len(dataset.get('annotations', []))
        num_categories = len(dataset.get('categories', []))
        
        print(f"  📊 Stats: {num_images} images, {num_annotations} annotations, {num_categories} categories")
        
        # Categories info
        print(f"  🏷️  Categories:")
        for cat in dataset.get('categories', []):
            print(f"     ID {cat['id']}: {cat['name']}")
        
        # Sample images
        print(f"  🖼️  Sample images:")
        for img in dataset.get('images', [])[:3]:
            print(f"     ID {img['id']}: {img['file_name']} ({img['width']}x{img['height']})")
        
        # Sample annotations
        print(f"  📝 Sample annotations:")
        for ann in dataset.get('annotations', [])[:3]:
            print(f"     ID {ann['id']}: category {ann['category_id']}, "
                  f"area {ann.get('area', 'N/A')}, "
                  f"{'has score' if 'score' in ann else 'no score'}")
            if 'segmentation' in ann:
                seg_type = 'RLE' if isinstance(ann['segmentation'], dict) else 'polygon'
                print(f"              segmentation: {seg_type}")
    
    def _cross_validate_cocos(self, truth_coco: COCO, preds_coco: COCO):
        """Cross-validate that prediction and truth COCOs are compatible"""
        truth_imgs = {img['file_name']: img['id'] for img in truth_coco.dataset.get('images', [])}
        preds_imgs = {img['file_name']: img['id'] for img in preds_coco.dataset.get('images', [])}
        
        # Check image alignment
        common_images = set(truth_imgs.keys()) & set(preds_imgs.keys())
        truth_only = set(truth_imgs.keys()) - set(preds_imgs.keys())
        preds_only = set(preds_imgs.keys()) - set(truth_imgs.keys())
        
        print(f"  🖼️  Image alignment:")
        print(f"     Common images: {len(common_images)}")
        if truth_only:
            print(f"     Truth-only images: {len(truth_only)} (first 3: {list(truth_only)[:3]})")
        if preds_only:
            print(f"     Preds-only images: {len(preds_only)} (first 3: {list(preds_only)[:3]})")
        
        # Check category alignment
        truth_cats = {cat['name']: cat['id'] for cat in truth_coco.dataset.get('categories', [])}
        preds_cats = {cat['name']: cat['id'] for cat in preds_coco.dataset.get('categories', [])}
        
        print(f"  🏷️  Category alignment:")
        print(f"     Truth categories: {list(truth_cats.keys())}")
        print(f"     Preds categories: {list(preds_cats.keys())}")
        
        mismatched_cats = []
        for name in truth_cats:
            if name in preds_cats and truth_cats[name] != preds_cats[name]:
                mismatched_cats.append((name, truth_cats[name], preds_cats[name]))
        
        if mismatched_cats:
            print(f"     ⚠️  Mismatched category IDs:")
            for name, truth_id, pred_id in mismatched_cats:
                print(f"        {name}: truth={truth_id}, preds={pred_id}")
        else:
            print(f"     ✅ Category IDs match")
    
    def tile_level(self, preds_coco_path: str, truth_coco_path: str, max_dets=(1, 10, 100)):
        """
        Placeholder tile-level evaluation that inspects the data.
        TODO: Implement actual mAP calculation.
        """
        print("📊 TILE-LEVEL EVALUATION (INSPECTION MODE)")
        
        success = self.inspect_coco_files(preds_coco_path, truth_coco_path)
        
        if success:
            return {
                'status': 'inspection_complete',
                'message': 'Data loading and structure validation passed',
                'num_images': 'see_logs',
                'num_truths': 'see_logs', 
                'num_preds': 'see_logs'
            }
        else:
            return {
                'status': 'inspection_failed',
                'message': 'Data loading or structure validation failed'
            }