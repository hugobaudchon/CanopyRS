from typing import List

import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from sam2.sam2_image_predictor import SAM2ImagePredictor

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box
from pathlib import Path

@SEGMENTER_REGISTRY.register('sam2')
class Sam2PredictorWrapper(SegmenterWrapperBase):
    MODEL_MAPPING = {
        't': "facebook/sam2-hiera-tiny",
        's': "facebook/sam2-hiera-small",
        'b': "facebook/sam2-hiera-base-plus",
        'l': "facebook/sam2-hiera-large",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):

        super().__init__(config)

        self.model_name = self.MODEL_MAPPING[self.config.architecture]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                print(f"Loading fine-tuned checkpoint:")
                print(f"  Path: {checkpoint_path}")
                
                # Load state dict
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    # Full checkpoint with optimizer, etc.
                    model_state_dict = state_dict['model_state_dict']
                    print(f"  Checkpoint type: Full training checkpoint")
                else:
                    # Just model weights
                    model_state_dict = state_dict
                    print(f"  Checkpoint type: Model weights only")
                
                # Load weights into predictor model
                self.predictor.model.load_state_dict(model_state_dict)
                print(f"Fine-tuned weights loaded successfully!")
            else:
                print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
                print(f"Using base pretrained model instead.\n")
                
    def forward(self,
                images: List[np.array],
                boxes: List[np.array],
                boxes_object_ids: List[int],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue):

        # Only 1 image per batch supported for now
        for image, image_boxes, image_boxes_object_ids, tile_idx in zip(images, boxes, boxes_object_ids, tiles_idx):
            image = image[:3, :, :]
            image = image.transpose((1, 2, 0))
            image = (image * 255).astype(np.uint8)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pil_image = Image.fromarray(image).convert("RGB")
                self.predictor.set_image(pil_image)

                # Process bounding boxes in batches
                n_masks_processed = 0
                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = image_boxes[i:i + self.config.box_batch_size]  # Select a batch of boxes
                    box_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]
                    masks, scores, _ = self.predictor.predict(box=box_batch, multimask_output=False, normalize_coords=True)

                    # reshaping to (batch_size, h, w)
                    if masks.ndim == 4:
                        masks = masks[:, 0, :, :]

                    image_size = (image.shape[0], image.shape[1])
                    n_masks_processed = self.queue_masks(
                        box_object_ids_batch, masks, image_size, scores, tile_idx, n_masks_processed, queue
                    )

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)
