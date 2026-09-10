import warnings
from abc import ABC, abstractmethod

import torch
import torchmetrics
from shapely import box

from canopyrs.engine.loader import InferTimer
from canopyrs.engine.models.utils import load_state_dict_with_key_repair

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing from timm.models.layers is deprecated"
)


class DetectorWrapperBase(ABC):
    def __init__(self, config, ):
        self.config = config

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = None

    @abstractmethod
    def forward(self, images, targets=None):
        pass

    def infer(self, loader):
        """Consume a ``tile_loader``, iterated as ``(object_ids, images)`` batches, and
        return ``(object_ids, boxes, scores, classes)`` as aligned per-tile lists. Reuses
        ``forward``; builds no DataLoader of its own."""
        self.model.eval()
        object_ids, results = [], []
        timer = InferTimer("Inferring detector...")
        with torch.no_grad():
            for batch_ids, images in timer.batches(loader):
                images = [img.to(self.device) for img in images]
                timer.mark("prep")
                results.extend(self.forward(images))
                object_ids.extend(batch_ids)
                timer.mark("gpu")

        boxes, boxes_scores, classes = detector_result_to_lists(results)   # to cpu + shapely boxes
        timer.mark("post")
        timer.report()
        return object_ids, boxes, boxes_scores, classes


class TorchVisionDetectorWrapperBase(DetectorWrapperBase, ABC):
    def __init__(self, config, ):
        super().__init__(config)

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            # backend='faster_coco_eval',   # Requires additional dependencies
            iou_type="bbox",
            # max_detection_thresholds=[1, 10, self.box_predictions_per_image]
        ).to(self.device)

    def load_checkpoint(self, checkpoint_state_dict_path):
        load_state_dict_with_key_repair(self.model, checkpoint_state_dict_path)

    def _save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


def detector_result_to_lists(detector_result):
    detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
    for x in detector_result:
        x['boxes'] = [box(*b) for b in x['boxes']]
        x['scores'] = x['scores'].tolist()
        x['classes'] = x['labels'].tolist()
    boxes = [x['boxes'] for x in detector_result]
    scores = [x['scores'] for x in detector_result]
    classes = [x['classes'] for x in detector_result]

    return boxes, scores, classes
