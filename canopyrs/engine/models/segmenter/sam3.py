from typing import List
import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from pathlib import Path

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box


@SEGMENTER_REGISTRY.register('sam3')
class Sam3PredictorWrapper(SegmenterWrapperBase):
    """
    SAM3 Tracker (PVS) wrapper.

    - Uses Sam3TrackerModel / Sam3TrackerProcessor (Promptable Visual Segmentation).
    - Takes GT / detector boxes as prompts and returns ONE mask per box (no PCS / concept detection).
    """

    # For now, everything maps to the single HF checkpoint.
    MODEL_MAPPING = {
        't': "facebook/sam3",
        's': "facebook/sam3",
        'b': "facebook/sam3",
        'l': "facebook/sam3",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = self.MODEL_MAPPING[self.config.architecture]

        print(f"Loading SAM3 Tracker model {self.model_name}")
        self.processor = Sam3TrackerProcessor.from_pretrained(self.model_name)
        self.model = Sam3TrackerModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"SAM3 Tracker model {self.model_name} loaded")

        # thresholds (optional, from config)
        # score_threshold is not really used in tracker mode but kept for API symmetry
        self.score_threshold = getattr(self.config, "sam3_score_threshold", 0.0)
        # For tracker, HF examples use default mask_threshold=0.0
        self.mask_threshold = getattr(self.config, "sam3_mask_threshold", 0.0)
                # Target tile size for fair comparison with Detectree2 (default 1777 to match typical Detectree2 input)
        self.target_tile_size = getattr(self.config, "target_tile_size", 1777)
        print(f"SAM3 will resize tiles to {self.target_tile_size}x{self.target_tile_size} for processing")

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"\n{'='*60}")
            print(f"Loading fine-tuned checkpoint:")
            print(f"  Path: {checkpoint_path}")

            state_dict = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in state_dict:
                model_state_dict = state_dict['model_state_dict']
                print("  Checkpoint type: Full training checkpoint")
                if 'epoch' in state_dict:
                    print(f"  Epoch: {state_dict['epoch']}")
            else:
                model_state_dict = state_dict
                print("  Checkpoint type: Model weights only")

            self.model.load_state_dict(model_state_dict, strict=False)
            print("✓ Fine-tuned weights loaded successfully!")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  WARNING: Checkpoint not found: {checkpoint_path}")
            print("   Using base pretrained model instead.\n")

    # ------------------------ main API ------------------------ #

    def forward(
        self,
        images: List[np.array],
        boxes: List[np.array],
        boxes_object_ids: List[int],
        tiles_idx: List[int],
        queue: multiprocessing.JoinableQueue,
    ):
        """
        images: list of CxHxW np arrays (C at least 3)
        boxes:  list of (Ni, 4) xyxy boxes per image
        boxes_object_ids: list of object-id lists aligned with boxes
        tiles_idx: list of tile indices
        """

        for image, image_boxes, image_boxes_object_ids, tile_idx in zip(
            images, boxes, boxes_object_ids, tiles_idx
        ):
            # C,H,W -> H,W,C, RGB only
            image = image[:3, :, :]
            image = image.transpose((1, 2, 0))

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            orig_H, orig_W, _ = image.shape
            
            # Resize to target tile size for fair comparison with Detectree2
            if orig_H != self.target_tile_size or orig_W != self.target_tile_size:
                import cv2
                image = cv2.resize(image, (self.target_tile_size, self.target_tile_size), interpolation=cv2.INTER_LINEAR)
                
                # Scale boxes to match resized image (convert to float to avoid casting errors)
                scale_x = self.target_tile_size / orig_W
                scale_y = self.target_tile_size / orig_H
                image_boxes = image_boxes.astype(np.float32)
                image_boxes[:, [0, 2]] *= scale_x
                image_boxes[:, [1, 3]] *= scale_y
            
            H, W, _ = image.shape
            # For output masks, we want them at ORIGINAL resolution
            output_size = (orig_H, orig_W)

            # sanity
            if len(image_boxes) == 0:
                continue

            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                pil_image = Image.fromarray(image).convert("RGB")

                n_masks_processed = 0
                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = np.asarray(
                        image_boxes[i : i + self.config.box_batch_size],
                        dtype=np.float32,
                    )
                    box_object_ids_batch = image_boxes_object_ids[
                        i : i + self.config.box_batch_size
                    ]

                    # Clip boxes to image bounds
                    box_batch[:, 0] = np.clip(box_batch[:, 0], 0, W)
                    box_batch[:, 1] = np.clip(box_batch[:, 1], 0, H)
                    box_batch[:, 2] = np.clip(box_batch[:, 2], 0, W)
                    box_batch[:, 3] = np.clip(box_batch[:, 3], 0, H)

                    # Drop degenerate boxes
                    valid = (box_batch[:, 2] > box_batch[:, 0]) & (
                        box_batch[:, 3] > box_batch[:, 1]
                    )
                    if not valid.any():
                        continue
                    box_batch = box_batch[valid]
                    box_object_ids_batch = [
                        oid for oid, v in zip(box_object_ids_batch, valid) if v
                    ]

                    masks, scores = self._predict_batch(pil_image, box_batch)

                    if masks is None or len(masks) == 0:
                        continue
                    
                    # If we resized, scale masks back to original resolution
                    if orig_H != self.target_tile_size or orig_W != self.target_tile_size:
                        import cv2
                        masks_resized = []
                        for mask in masks:
                            mask_resized = cv2.resize(mask, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
                            masks_resized.append(mask_resized)
                        masks = np.array(masks_resized)

                    n_masks_processed = self.queue_masks(
                        box_object_ids_batch,
                        masks,
                        output_size,
                        scores,
                        tile_idx,
                        n_masks_processed,
                        queue,
                    )

    def _predict_batch(self, image: Image.Image, boxes: np.ndarray):
        """
        Run SAM3 Tracker (PVS) on one image with a batch of box prompts.

        Args:
            image: PIL.Image
            boxes: (Nq, 4) float32 [x1,y1,x2,y2]

        Returns:
            masks: (Nq, H, W) uint8 (one mask per input box)
            scores: (Nq,) float32 (IoU scores or dummy ones)
        """
        if len(boxes) == 0:
            return None, None

        # Sam3Tracker expects list-of-list of boxes: [batch, num_objects, 4]
        input_boxes = [boxes.tolist()]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        # We want a single best mask per object (no 3-masks-per-object multimask)
        outputs = self.model(**inputs, multimask_output=False)

        # Post-process to original resolution
        # returns list over batch; we have 1 image → [0]
        masks_t = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            mask_threshold=self.mask_threshold,
            binarize=True,
        )[0]

        # shapes can be:
        # - (Nobj, H, W)
        # - or (Nobj, 1, H, W)
        if masks_t.ndim == 4 and masks_t.shape[1] == 1:
            masks_t = masks_t[:, 0]  # (Nobj, H, W)

        masks = masks_t.to(torch.uint8).numpy()  # (Nobj, H, W), 0/1

        # IoU scores per object (if provided)
        # Sam-like output: iou_scores: (batch_size, num_masks) or (batch_size, num_objects)
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            scores_t = outputs.iou_scores[0]
            scores = scores_t.detach().float().cpu().numpy()
            # If there's a mismatch for any reason, fall back to ones
            if scores.shape[0] != masks.shape[0]:
                scores = np.ones(masks.shape[0], dtype=np.float32)
        else:
            scores = np.ones(masks.shape[0], dtype=np.float32)

        return masks, scores

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)