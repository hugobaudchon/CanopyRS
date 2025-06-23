# engine/benchmark/classifier/evaluator.py
import warnings
import numpy as np
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster

class ClassifierCocoEvaluator:
    small_max_sq_meters = 16
    medium_max_sq_meters = 100

    def tile_level(self,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list[int] = (1, 10, 100)) -> dict:
        """
        Tile-level evaluation for instance segmentation with classes.
        Uses 'segm' IoU type for polygon/mask evaluation.
        """
        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        # Remove scores from ground truth annotations (if any)
        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']

        # Align predictions to truth based on file name
        self._align_coco_datasets_by_name(truth_coco, preds_coco)

        # Set up and run COCO evaluation for segmentation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType='segm'  # Use segmentation IoU instead of bbox
        )
        coco_evaluator.params.maxDets = max_dets
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        # Get metrics as a dictionary and add debug info
        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(truth_coco.dataset.get('images', []))
        num_truths = len(truth_coco.dataset.get('annotations', []))
        num_preds = len(preds_coco.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds

        return metrics

    def _align_coco_datasets_by_name(self, truth_coco: COCO, preds_coco: COCO) -> None:
        """
        Align the predictions COCO dataset to follow the order of the truth COCO dataset,
        matching based on the image 'file_name'. For any truth image missing in preds,
        insert a dummy image (with no annotations) so that the image IDs match.
        This function updates the preds_coco in-place.
        """
        preds_by_name = {img['file_name']: img for img in preds_coco.dataset.get('images', [])}
        id_mapping = {}
        new_preds_images = []
        
        for truth_img in truth_coco.dataset.get('images', []):
            file_name = truth_img['file_name']
            truth_id = truth_img['id']
            if file_name in preds_by_name:
                preds_img = preds_by_name[file_name]
                id_mapping[preds_img['id']] = truth_id
                new_img = preds_img.copy()
                new_img['id'] = truth_id
                new_preds_images.append(new_img)
            else:
                new_preds_images.append(truth_img.copy())
        
        preds_coco.dataset['images'] = new_preds_images

        new_preds_annotations = []
        for ann in preds_coco.dataset.get('annotations', []):
            orig_img_id = ann['image_id']
            if orig_img_id in id_mapping:
                ann['image_id'] = id_mapping[orig_img_id]
                new_preds_annotations.append(ann)
        
        preds_coco.dataset['annotations'] = new_preds_annotations
        preds_coco.createIndex()


class Summarize2COCOEval(COCOeval_faster):
    def summarize_custom(self):
        max_dets_index = len(self.params.maxDets) - 1

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            mean_s = np.mean(s[s > -1]) if len(s[s > -1]) > 0 else -1
            stat_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            print(stat_string)
            return mean_s, stat_string

        def _summarizeDets():
            stats = np.zeros((15,))
            stats_strings = ['' for _ in range(15)]

            # AP metrics
            stats[0], stats_strings[0] = _summarize(1, maxDets=self.params.maxDets[max_dets_index])
            stats[1], stats_strings[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[2], stats_strings[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            stats[3], stats_strings[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[4], stats_strings[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[5], stats_strings[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])

            # AR metrics
            stats[6], stats_strings[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7], stats_strings[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8], stats_strings[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9], stats_strings[9] = (_summarize(0, maxDets=self.params.maxDets[3])
                                          if len(self.params.maxDets) > 3
                                          else _summarize(0, maxDets=self.params.maxDets[2]))

            # AR at specific IoU thresholds
            stats[10], stats_strings[10] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[11], stats_strings[11] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])

            # AR for different object sizes
            stats[12], stats_strings[12] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[13], stats_strings[13] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[14], stats_strings[14] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            return stats, stats_strings

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        iouType = self.params.iouType
        if iouType in ['segm', 'bbox']:
            summarize = _summarizeDets
        else:
            raise Exception('Unknown iouType: {}'.format(iouType))
        
        self.stats, stats_strings = summarize()
        return stats_strings

    def summarize_to_dict(self):
        self.summarize_custom()
        stats = self.stats
        metric_names = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR", "AR50", "AR75", "AR_small", "AR_medium", "AR_large"
        ]
        metrics_dict = {name: float(value) for name, value in zip(metric_names, stats)}
        return metrics_dict