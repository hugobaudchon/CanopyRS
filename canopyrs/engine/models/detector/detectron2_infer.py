import logging
import warnings
from typing import List

import numpy as np
import torch
# Patch torch.load for PyTorch 2.6+ to allow loading detrex checkpoints
_original_torch_load = torch.load

def _patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for detrex compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load
from detectron2.config import instantiate
from detectron2.data.transforms import AugmentationList, AugInput, ResizeShortestEdge
from detectron2.modeling import build_model
from detrex.checkpoint import DetectionCheckpointer as DetrexDetectionCheckpointer
from detectron2.checkpoint import DetectionCheckpointer as Detectron2DetectionCheckpointer
from torch import Tensor

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.models.detector.detector_base import DetectorWrapperBase
from canopyrs.engine.models.detector.train_detectron2.augmentation import AugmentationAdder
from canopyrs.engine.models.detector.train_detectron2.train_detectron2 import get_base_detectron2_model_cfg
from canopyrs.engine.models.detector.train_detectron2.train_detrex import get_base_detrex_model_cfg
from canopyrs.engine.models.segmenter.detectree2 import setup_detectree2_cfg
from canopyrs.engine.models.registry import DETECTOR_REGISTRY

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
detrex_logger = logging.getLogger("detrex.checkpoint.c2_model_loading")
detrex_logger.disabled = True


@DETECTOR_REGISTRY.register('dino_detrex', 'faster_rcnn_detectron2', 'retinanet_detectron2', 'detectree2')
class Detectron2DetectorWrapper(DetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        if self.config.model.endswith('detectron2'):
            cfg = get_base_detectron2_model_cfg(self.config)
            self.model = build_model(cfg)
            self.model.eval()
            checkpointer = Detectron2DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = AugmentationList(AugmentationAdder().get_augmentation_detectron2_test(cfg))
            self.input_format = cfg.INPUT.FORMAT
        elif self.config.model == 'detectree2':
            cfg = setup_detectree2_cfg(
                base_model=self.config.architecture,
                update_model=self.config.checkpoint_path,
                ims_per_batch=self.config.batch_size
            )
            self.model = build_model(cfg)
            self.model.eval()
            checkpointer = Detectron2DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
            self.input_format = cfg.INPUT.FORMAT
        elif self.config.model.endswith('detrex'):
            cfg = get_base_detrex_model_cfg(self.config)
            self.model = instantiate(cfg.model)
            self.model.eval()
            checkpointer = DetrexDetectionCheckpointer(self.model)
            checkpointer.load(cfg.train.init_checkpoint)
            self.model.to(self.device)
            self.aug = AugmentationList(cfg.dataloader.test.mapper.augmentation)
            self.input_format = getattr(cfg.model, 'input_format',
                                        getattr(cfg.dataloader.test.mapper, 'image_format', 'RGB'))
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")

    def forward(self, images: List[Tensor], targets=None):
        """
        Perform inference on a batch of images.

        Args:
            images (List[Tensor]): A batch of images as Tensors in (C, H, W) format.
            targets: Ignored for inference.

        Returns:
            predictions (list): A list of prediction dictionaries, one per image, where the boxes
                                are scaled back to the original image size.
        """
        with torch.no_grad():
            inputs = []
            scale_factors = []  # List to hold (scale_x, scale_y) for each image.

            # Loop over the batch of images.
            for image in images:
                # Convert from tensor (C, H, W) to numpy (H, W, C)
                image = image.cpu().numpy().transpose(1, 2, 0)
                # scale back to [0, 255]
                image = (image * 255.0).astype(np.uint8)

                # Save original dimensions.
                orig_h, orig_w = image.shape[:2]

                # Convert channels if needed.
                if self.input_format == "BGR":
                    image = image[:, :, ::-1]

                # Apply augmentations.
                aug_input = AugInput(image)
                self.aug(aug_input)  # This applies the augmentations in-place.
                image_transformed = aug_input.image
                aug_h, aug_w = image_transformed.shape[:2]

                # Compute scaling factors to map from augmented image to original.
                scale_x = orig_w / aug_w
                scale_y = orig_h / aug_h
                scale_factors.append((scale_x, scale_y))

                # Convert the transformed image to a tensor in CHW order.
                image_tensor = torch.as_tensor(image_transformed.astype("float32").transpose(2, 0, 1))

                # Create the input dictionary expected by the model.
                inputs.append({
                    "image": image_tensor,
                    "height": aug_h,
                    "width": aug_w
                })

            # Run inference.
            predictions = self.model(inputs)
            # Convert to torchvision Faster R-CNN style format.
            predictions = self.convert_detectron2_predictions(predictions)

            # Rescale predicted boxes to original image sizes.
            for i, (scale_x, scale_y) in enumerate(scale_factors):
                boxes = predictions[i]["boxes"]
                if boxes.numel() > 0:
                    # boxes are in [x1, y1, x2, y2] format.
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

            return predictions

    @staticmethod
    def convert_detectron2_predictions(d2_predictions):
        """
        Convert Detectron2 predictions to torchvision Faster R-CNN format.

        Args:
            d2_predictions (list[dict]): List of predictions as returned by Detectron2.
                Each element is expected to have an "instances" key containing an Instances object.

        Returns:
            list[dict]: Each dictionary has keys "boxes", "scores", and "labels", matching
                        the torchvision Faster R-CNN output format.
        """
        converted = []
        for pred in d2_predictions:
            # Check that "instances" exists.
            if "instances" not in pred:
                converted.append(pred)
                continue

            # Move predictions to CPU (if on GPU).
            instances = pred["instances"].to("cpu")

            # If no instances were predicted, return empty tensors.
            if len(instances) == 0:
                converted.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                })
                continue

            # Extract boxes, scores, and labels.
            boxes = instances.pred_boxes.tensor  # shape: (N, 4)
            scores = instances.scores  # shape: (N,)
            labels = instances.pred_classes  # shape: (N,)

            converted.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        return converted

