import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import SamModel, SamProcessor

from engine.config_parsers import SegmenterConfig
from engine.models.segmenter.segmenter_base import SegmenterWrapperBase


def sam_collate_fn(single_image_batch):
    return single_image_batch[0]

class SamPredictorWrapper(SegmenterWrapperBase):
    MODEL_TYPE_MAPPING = {
        'b': "facebook/sam-vit-base",
        'l': "facebook/sam-vit-large",
        'h': "facebook/sam-vit-huge"
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        self.model_name = self.MODEL_TYPE_MAPPING[self.config.backbone]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.model = SamModel.from_pretrained(self.model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded")

    def infer_image(self,
                    image: np.array,
                    boxes: np.array,
                    tile_idx: int,
                    queue: multiprocessing.JoinableQueue):

        image = image[:3, :, :]
        image = image.transpose((1, 2, 0))
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
        boxes = boxes.tolist()

        # Process bounding boxes in batches
        n_masks_processed = 0
        for i in range(0, len(boxes), self.config.box_batch_size):
            box_batch = boxes[i:i + self.config.box_batch_size]  # Select a batch of boxes
            inputs = self.processor(pil_image, input_boxes=[box_batch], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)

            # Post-process masks
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            scores = outputs.iou_scores.cpu()

            n_masks_processed = self.queue_masks(
                masks[0], scores[0], tile_idx, n_masks_processed, queue
            )

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, sam_collate_fn)
