import multiprocessing
from typing import List

import numpy as np
import torch
from detectron2.config import instantiate
from detectron2.data.transforms import AugmentationList, AugInput, ResizeShortestEdge
from detectron2.modeling import build_model
from detrex.checkpoint import DetectionCheckpointer as DetrexDetectionCheckpointer
from detectron2.checkpoint import DetectionCheckpointer as Detectron2DetectionCheckpointer
from geodataset.dataset import UnlabeledRasterDataset

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.detector.train_detectron2.augmentation import AugmentationAdder
from canopyrs.engine.models.detector.train_detectron2.train_detectron2 import get_base_detectron2_model_cfg
from canopyrs.engine.models.detector.train_detectron2.train_detrex import get_base_detrex_model_cfg
from canopyrs.engine.models.segmenter.detectree2 import setup_detectree2_cfg
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_trivial


@SEGMENTER_REGISTRY.register('detectree2', 'mask_rcnn_detectron2', 'mask2former_detrex')
class Detectron2SegmenterWrapper(SegmenterWrapperBase):
    REQUIRES_BOX_PROMPT = False

    def __init__(self, config: SegmenterConfig):
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
            cfg = setup_detectree2_cfg(base_model=self.config.architecture, update_model=self.config.checkpoint_path)
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

    def forward(self,
                images: List[np.array],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue,
                **kwargs):
        """
         Perform inference on a batch of images.

         Args:
             images (List[np.array]): A batch of images as numpy arrays
             tile_idx (List[int]): A list of tile indices corresponding to the images.
             queue (multiprocessing.JoinableQueue): A queue to put the results in.

         Returns:
             predictions (list): A list of prediction dictionaries, one per image, where the boxes
                                 are scaled back to the original image size.
        """

        with torch.no_grad():
            inputs = []
            scale_factors = []
            orig_dims = []

            # Preprocess images
            for image in images:
                # Convert from tensor (C, H, W) to numpy (H, W, C) and rescale to [0, 255]
                image = image.transpose(1, 2, 0) * 255.0
                orig_h, orig_w = image.shape[:2]
                orig_dims.append((orig_h, orig_w))

                # Convert channels if needed
                if self.input_format == "RGB":
                    image = image[:, :, ::-1]

                # Apply augmentations
                aug_input = AugInput(image)
                self.aug(aug_input)
                image_transformed = aug_input.image
                aug_h, aug_w = image_transformed.shape[:2]

                # Compute scaling factors (augmented -> original)
                scale_x = orig_w / aug_w
                scale_y = orig_h / aug_h
                scale_factors.append((scale_x, scale_y))

                # Convert to tensor in CHW order
                image_tensor = torch.as_tensor(image_transformed.astype("float32").transpose(2, 0, 1))
                inputs.append({
                    "image": image_tensor,
                    "height": aug_h,
                    "width": aug_w
                })

            # Run inference
            predictions = self.model(inputs)

            # Process predictions and queue masks for each image
            for pred, (scale_x, scale_y), (orig_h, orig_w), tile_idx in zip(predictions, scale_factors, orig_dims, tiles_idx):
                # Assume predictions always include "instances" with masks
                instances = pred["instances"].to("cpu")
                masks = instances.pred_masks.numpy()
                if masks.ndim == 4:
                    masks = masks[:, 0, :, :]

                scores = instances.scores.numpy() if instances.has("scores") else None
                image_size = (orig_h, orig_w)
                n_masks_processed = 0

                image_boxes_object_ids = [None] * masks.shape[0]

                _ = self.queue_masks(image_boxes_object_ids, masks, image_size, scores, tile_idx, n_masks_processed, queue)

    def infer_on_dataset(self, dataset: UnlabeledRasterDataset):
        return self._infer_on_dataset(dataset, collate_fn_trivial)

