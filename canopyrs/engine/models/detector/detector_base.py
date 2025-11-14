import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
import torchmetrics
from geodataset.dataset import UnlabeledRasterDataset
from huggingface_hub import hf_hub_download
from shapely import box
from torch.utils.data import DataLoader
from tqdm import tqdm

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


class TorchVisionDetectorWrapperBase(DetectorWrapperBase, ABC):
    def __init__(self, config, ):
        super().__init__(config)

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            # backend='faster_coco_eval',   # Requires additional dependencies
            iou_type="bbox",
            # max_detection_thresholds=[1, 10, self.box_predictions_per_image]
        ).to(self.device)

    def load_checkpoint(self, checkpoint_state_dict_path):
        checkpoint_state_dict_path = Path(checkpoint_state_dict_path)
        if 'huggingface.co' in checkpoint_state_dict_path.parts:
            if "huggingface.co" not in checkpoint_state_dict_path.as_posix():
                raise ValueError("The provided Path does not contain a valid Hugging Face URL.")
            # Remove the "https://huggingface.co/" part
            path = Path(str(checkpoint_state_dict_path).replace("\\", "/").split("huggingface.co/")[-1])
            if "resolve" not in path.parts:
                raise ValueError("The provided Path is not in the expected Hugging Face format.")
            # Extract repo_id and filename
            repo_id = "/".join(path.parts[:2])
            filename = path.name
            checkpoint_state_dict_path = hf_hub_download(repo_id=repo_id, filename=filename)

        if checkpoint_state_dict_path:
            try:
                self.model.load_state_dict(torch.load(checkpoint_state_dict_path, weights_only=False))
            except RuntimeError as e:
                state_dict = try_rename_state_dict_keys_with_model(checkpoint_state_dict_path)
                self.model.load_state_dict(state_dict)

    def _save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


def try_rename_state_dict_keys_with_model(checkpoint_state_dict_path):
    # Structure the OrderedDict keys to match requirements
    checkpoint = torch.load(checkpoint_state_dict_path, weights_only=True)
    if "model" in checkpoint.keys():
        # Case where other attributes are stored in the checkpoint
        checkpoint = checkpoint["model"]
    # Create a new OrderedDict with the keys prefixed with "model."
    new_state_dict = OrderedDict()
    if all(s.startswith("model.") for s in checkpoint.keys()):
        # try removing the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = key[6:]
            new_state_dict[new_key] = value
    elif all(s.startswith("module.") for s in checkpoint.keys()):
        # try removing the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
    else:
        # try adding the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = 'model.' + key  # Prefix "model." to each key
            new_state_dict[new_key] = value
    return new_state_dict


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
