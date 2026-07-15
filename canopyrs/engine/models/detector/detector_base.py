import warnings
from abc import ABC, abstractmethod

import torch
import torchmetrics
from geodataset.dataset import UnlabeledRasterDataset
from shapely import box
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def _infer(self, data_loader):
        self.model.eval()

        predictions = []

        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc="Inferring detector...",
                                             leave=True)
            for images in data_loader_with_progress:
                images = list(img.to(self.device) for img in images)
                outputs = self.forward(images)
                predictions.extend(outputs)

        return predictions

    def infer(self, infer_ds: UnlabeledRasterDataset, collate_fn: callable):
        infer_dl = DataLoader(infer_ds, batch_size=self.config.batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        results = self._infer(infer_dl)
        boxes, boxes_scores, classes = detector_result_to_lists(results)
        tiles_paths = infer_ds.tile_paths
        return tiles_paths, boxes, boxes_scores, classes

    def infer_v2(self, loader):
        """v2 inference: consume a loader yielding ``(object_ids, images)`` batches (e.g. the v2
        ``tile_loader``) and return ``(object_ids, boxes, scores, classes)`` as aligned per-tile
        lists. Reuses ``forward``; builds no DataLoader of its own."""
        self.model.eval()
        object_ids, results = [], []
        with torch.no_grad():
            for batch_ids, images in tqdm(loader, desc="Inferring detector...", leave=True):
                images = [img.to(self.device) for img in images]
                results.extend(self.forward(images))
                object_ids.extend(batch_ids)

        boxes, boxes_scores, classes = detector_result_to_lists(results)
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
