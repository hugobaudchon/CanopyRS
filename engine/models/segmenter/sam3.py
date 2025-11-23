from typing import List
from pathlib import Path
import multiprocessing

import numpy as np
import torch
from PIL import Image

from transformers import Sam3Processor, Sam3Model

from geodataset.dataset import DetectionLabeledRasterCocoDataset

from engine.config_parsers import SegmenterConfig
from engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from engine.models.registry import SEGMENTER_REGISTRY
from engine.models.utils import collate_fn_infer_image_box


@SEGMENTER_REGISTRY.register('sam3')
class Sam3PredictorWrapper(SegmenterWrapperBase):
    """
    SAM3 wrapper with the same external interface as your Sam2PredictorWrapper.
    Uses bounding boxes as *visual exemplars* (positive prompts) and, for each
    input box, returns a single mask by choosing the prediction with the
    highest IoU to the prompt box.
    """

    # SAM3 currently has a single HF ID, but we keep mapping for compatibility.
    MODEL_MAPPING = {
        't': "facebook/sam3",
        's': "facebook/sam3",
        'b': "facebook/sam3",
        'l': "facebook/sam3",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        # ---- Model name / HF checkpoint ----
        # If you want to override, you can set config.hf_model_name = "...".
        self.model_name = getattr(
            self.config,
            "hf_model_name",
            self.MODEL_MAPPING.get(self.config.architecture, "facebook/sam3"),
        )

        print(f"Loading SAM3 model: {self.model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HF SAM3 model + processor
        self.model = Sam3Model.from_pretrained(self.model_name).to(self.device)
        self.processor = Sam3Processor.from_pretrained(self.model_name)
        self.model.eval()
        print(f"SAM3 model {self.model_name} loaded")

        # Optional fine-tuned checkpoint
        checkpoint_path = getattr(self.config, "checkpoint_path", None)
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                print("Loading fine-tuned checkpoint:")
                print(f"  Path: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=self.device)

                if "model_state_dict" in state_dict:
                    model_state_dict = state_dict["model_state_dict"]
                    print("  Checkpoint type: Full training checkpoint")
                else:
                    model_state_dict = state_dict
                    print("  Checkpoint type: Model weights only")

                self.model.load_state_dict(model_state_dict, strict=False)
                print("Fine-tuned SAM3 weights loaded successfully!")
            else:
                print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
                print("Using base pretrained SAM3 model instead.\n")

        # Thresholds (you can expose these in your config)
        self.score_threshold = getattr(self.config, "sam3_score_threshold", 0.0)
        self.mask_threshold = getattr(self.config, "sam3_mask_threshold", 0.5)

    # ------------------------------------------------------------------
    # Utility: box IoU and best-matching prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between a single box and an array of boxes.
        box:  (4,)   [x1, y1, x2, y2]
        boxes:(N, 4) [x1, y1, x2, y2]
        Returns: (N,) IoUs
        """
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)

        # Intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
        inter = inter_w * inter_h

        # Areas
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area_box + area_boxes - inter + 1e-6
        return inter / union

    def _run_sam3_single_box(
        self,
        image: np.ndarray,
        box_xyxy: np.ndarray,
    ):
        """
        Run SAM3 on a single image with a single positive box prompt.

        Args:
            image: H×W×3 uint8
            box_xyxy: (4,) float, [x1, y1, x2, y2] in pixel coords

        Returns:
            best_mask: (H, W) float32 in {0,1}, or None if nothing found
            best_score: float
        """
        pil_image = Image.fromarray(image).convert("RGB")

        # HF expects batch dimensions: input_boxes = [[box]]
        input_boxes = [[box_xyxy.tolist()]]
        input_boxes_labels = [[1]]  # positive exemplar

        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process instance segmentation to original size
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),  # [[H, W]]
        )[0]

        masks = results["masks"]   # list/array of (H, W) bool
        boxes = np.array(results["boxes"], dtype=np.float32)   # (N_pred, 4)
        scores = np.array(results["scores"], dtype=np.float32) # (N_pred,)

        if len(masks) == 0:
            return None, None

        # Choose mask whose predicted box best matches the prompt box
        ious = self._box_iou_one_to_many(box_xyxy.astype(np.float32), boxes)
        best_idx = int(ious.argmax())
        best_mask = np.array(masks[best_idx], dtype=np.float32)
        best_score = float(scores[best_idx])
        return best_mask, best_score

    # ------------------------------------------------------------------
    # Main inference API (same signature as Sam2PredictorWrapper.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        images: List[np.ndarray],
        boxes: List[np.ndarray],
        boxes_object_ids: List[int],
        tiles_idx: List[int],
        queue: multiprocessing.JoinableQueue,
    ):
        """
        Args:
            images: list of C×H×W tensors/arrays in [0,1] or [0,255]
            boxes: list of (N_i, 4) arrays, xyxy in pixel coords
            boxes_object_ids: list of object IDs aligned with boxes (flattened)
            tiles_idx: list of tile indices (same length as images)
            queue: multiprocessing.JoinableQueue used by SegmenterWrapperBase.queue_masks
        """

        # We assume 1 image per tile, as in your SAM2 code
        for image, image_boxes, image_boxes_object_ids, tile_idx in zip(
            images, boxes, boxes_object_ids, tiles_idx
        ):
            # Make sure we have only 3 channels and convert to H×W×3 uint8
            image = image[:3, :, :]
            image = image.transpose((1, 2, 0))  # C,H,W -> H,W,C

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            H, W, _ = image.shape
            image_size = (H, W)

            n_masks_processed = 0

            # For SAM3, we call once per box (simpler mapping: 1 prompt -> 1 mask)
            # You could later batch/optimize this if needed.
            for box_xyxy, obj_id in zip(image_boxes, image_boxes_object_ids):
                box_xyxy = np.array(box_xyxy, dtype=np.float32)

                # Sanity: clip box to image bounds
                box_xyxy[0] = np.clip(box_xyxy[0], 0, W)
                box_xyxy[1] = np.clip(box_xyxy[1], 0, H)
                box_xyxy[2] = np.clip(box_xyxy[2], 0, W)
                box_xyxy[3] = np.clip(box_xyxy[3], 0, H)

                # Skip degenerate boxes
                if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
                    continue

                best_mask, best_score = self._run_sam3_single_box(
                    image=image,
                    box_xyxy=box_xyxy,
                )

                if best_mask is None:
                    continue

                # Shape to (1, H, W) so queue_masks sees a batch
                masks_batch = np.expand_dims(best_mask, axis=0)
                scores_batch = np.array([best_score], dtype=np.float32)
                ids_batch = [obj_id]

                n_masks_processed = self.queue_masks(
                    ids_batch,
                    masks_batch,
                    image_size,
                    scores_batch,
                    tile_idx,
                    n_masks_processed,
                    queue,
                )

    # ------------------------------------------------------------------
    # Dataset-level inference wrapper
    # ------------------------------------------------------------------

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)
