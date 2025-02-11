from typing import List

import torch
from detectron2.config import instantiate
from detectron2.data.transforms import AugmentationList, AugInput
from detrex.checkpoint import DetectionCheckpointer as DetrexDetectionCheckpointer
from detectron2.checkpoint import DetectionCheckpointer as Detectron2DetectionCheckpointer
from torch import Tensor

from engine.config_parsers import DetectorConfig
from engine.models.detector.detector_base import DetectorWrapperBase
from engine.models.detector.train_detectron2.augmentation import AugmentationAdder
from engine.models.detector.train_detectron2.train_detectron2 import get_base_detectron2_model_cfg
from engine.models.detector.train_detectron2.train_detrex import get_base_detrex_model_cfg


class Detectron2PredictorWrapper(DetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        if self.config.model.endswith('detectron2'):
            cfg = get_base_detectron2_model_cfg(self.config)
            self.model = instantiate(cfg.MODEL)
            self.model.eval()
            checkpointer = Detectron2DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            # Instantiate the augmentations from the config
            self.aug = AugmentationList(AugmentationAdder().get_augmentation_detectron2_test(cfg))
            self.input_format = cfg.input.format
        elif self.config.model.endswith('detrex'):
            cfg = get_base_detrex_model_cfg(self.config)
            self.model = instantiate(cfg.model)
            self.model.eval()
            checkpointer = DetrexDetectionCheckpointer(self.model)
            checkpointer.load(cfg.train.init_checkpoint)
            self.model.to(self.device)
            self.aug = AugmentationList(cfg.dataloader.test.mapper.augmentation)
            self.input_format = cfg.model.input_format
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
                # Your dataset normalizes images to [0, 1]; scale back to [0, 255]
                image = image * 255.0

                # Save original dimensions.
                orig_h, orig_w = image.shape[:2]

                # Convert channels if needed.
                if self.input_format == "RGB":
                    # Assuming the model expects BGR, reverse channels.
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

