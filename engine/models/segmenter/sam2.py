import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from sam2.sam2_image_predictor import SAM2ImagePredictor

from engine.config_parsers import SegmenterConfig
from engine.models.segmenter.segmenter_base import SegmenterWrapperBase


def sam_collate_fn(single_image_batch):
    return single_image_batch[0]

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

        self.model_name = self.MODEL_MAPPING[self.config.backbone]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

    def infer_image(self,
                    image: np.array,
                    boxes: np.array,
                    tile_idx: int,
                    queue: multiprocessing.JoinableQueue):

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pil_image = Image.fromarray(image).convert("RGB")
            self.predictor.set_image(pil_image)

            # Process bounding boxes in batches
            n_masks_processed = 0
            for i in range(0, len(boxes), self.config.box_batch_size):
                box_batch = boxes[i:i + self.config.box_batch_size]  # Select a batch of boxes
                masks, scores, _ = self.predictor.predict(box=box_batch, multimask_output=False, normalize_coords=True)

                n_masks_processed = self.queue_masks(
                    masks, scores, tile_idx, n_masks_processed, queue
                )

    def infer_on_multi_box_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_multi_box_dataset(dataset, sam_collate_fn)
