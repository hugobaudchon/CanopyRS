from typing import List

import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import SamModel, SamProcessor

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box


@SEGMENTER_REGISTRY.register('sam')
class SamPredictorWrapper(SegmenterWrapperBase):
    MODEL_TYPE_MAPPING = {
        'b': "facebook/sam-vit-base",
        'l': "facebook/sam-vit-large",
        'h': "facebook/sam-vit-huge"
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        self.model_name = self.MODEL_TYPE_MAPPING[self.config.architecture]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.model = SamModel.from_pretrained(self.model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

    def forward(self,
                images: List[np.array],
                boxes: List[np.array],
                boxes_object_ids: List[np.array],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue):

        # Only 1 image per batch supported for now
        for image, image_boxes, image_boxes_object_ids, tile_idx in zip(images, boxes, boxes_object_ids, tiles_idx):
            image = image[:3, :, :]
            image = image.transpose((1, 2, 0))
            image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image).convert("RGB")
            image_boxes = image_boxes.tolist()

            # Process bounding boxes in batches
            n_masks_processed = 0
            for i in range(0, len(image_boxes), self.config.box_batch_size):
                box_batch = image_boxes[i:i + self.config.box_batch_size]  # Select a batch of boxes
                boxes_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]
                inputs = self.processor(pil_image, input_boxes=[box_batch], return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, multimask_output=False)

                # Post-process masks
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )

                # reshaping to (batch_size, h, w)
                masks = masks[0]
                if masks.ndim == 4:
                    masks = masks[:, 0, :, :]

                masks = masks.cpu().numpy().astype(np.uint8)

                scores = outputs.iou_scores.cpu()
                scores = scores[0]
                image_size = (image.shape[0], image.shape[1])

                n_masks_processed = self.queue_masks(
                    boxes_object_ids_batch, masks, image_size, scores, tile_idx, n_masks_processed, queue
                )

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)
