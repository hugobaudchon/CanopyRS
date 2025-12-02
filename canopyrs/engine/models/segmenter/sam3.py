from typing import List
import numpy as np
import torch
from PIL import Image
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing
from transformers import Sam3Processor, Sam3Model
from pathlib import Path

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box


@SEGMENTER_REGISTRY.register('sam3')
class Sam3PredictorWrapper(SegmenterWrapperBase):
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

        print(f"Loading SAM3 model {self.model_name}")
        self.processor = Sam3Processor.from_pretrained(self.model_name)
        self.model = Sam3Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"SAM3 model {self.model_name} loaded")

        # thresholds (optional, from config)
        self.score_threshold = getattr(self.config, "sam3_score_threshold", 0.0)
        self.mask_threshold = getattr(self.config, "sam3_mask_threshold", 0.5)

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

    # --------------------- IoU helper --------------------- #

    @staticmethod
    def _box_iou_matrix(query_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """
        IoU between each query box and each predicted box.
        query_boxes: (Nq, 4)  [x1,y1,x2,y2]
        pred_boxes:  (Np, 4)
        returns: (Nq, Np)
        """
        if len(query_boxes) == 0 or len(pred_boxes) == 0:
            return np.zeros((len(query_boxes), len(pred_boxes)), dtype=np.float32)

        q = query_boxes[:, None, :]   # (Nq,1,4)
        p = pred_boxes[None, :, :]    # (1,Np,4)

        x1 = np.maximum(q[..., 0], p[..., 0])
        y1 = np.maximum(q[..., 1], p[..., 1])
        x2 = np.minimum(q[..., 2], p[..., 2])
        y2 = np.minimum(q[..., 3], p[..., 3])

        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter = inter_w * inter_h

        area_q = (q[..., 2] - q[..., 0]) * (q[..., 3] - q[..., 1])
        area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
        union = area_q + area_p - inter + 1e-6

        return inter / union

    # ------------------------ main API ------------------------ #

    def forward(self,
                images: List[np.array],
                boxes: List[np.array],
                boxes_object_ids: List[int],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue):

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

            H, W, _ = image.shape
            image_size = (H, W)

            # sanity
            if len(image_boxes) == 0:
                continue

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                pil_image = Image.fromarray(image).convert("RGB")

                n_masks_processed = 0
                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = np.asarray(
                        image_boxes[i:i + self.config.box_batch_size],
                        dtype=np.float32
                    )
                    box_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]

                    # Clip boxes to image bounds
                    box_batch[:, 0] = np.clip(box_batch[:, 0], 0, W)
                    box_batch[:, 1] = np.clip(box_batch[:, 1], 0, H)
                    box_batch[:, 2] = np.clip(box_batch[:, 2], 0, W)
                    box_batch[:, 3] = np.clip(box_batch[:, 3], 0, H)

                    # Drop degenerate boxes
                    valid = (box_batch[:, 2] > box_batch[:, 0]) & (box_batch[:, 3] > box_batch[:, 1])
                    if not valid.any():
                        continue
                    box_batch = box_batch[valid]
                    box_object_ids_batch = [oid for oid, v in zip(box_object_ids_batch, valid) if v]

                    masks, scores = self._predict_batch(pil_image, box_batch)

                    if masks is None or len(masks) == 0:
                        continue

                    n_masks_processed = self.queue_masks(
                        box_object_ids_batch,
                        masks,
                        image_size,
                        scores,
                        tile_idx,
                        n_masks_processed,
                        queue,
                    )

    def _predict_batch(self, image: Image.Image, boxes: np.ndarray):
        """
        Run SAM3 PCS on one image with a batch of positive box prompts,
        then assign 1 mask per box by IoU matching.

        Args:
            image: PIL.Image
            boxes: (Nq, 4) float32 [x1,y1,x2,y2]

        Returns:
            masks: (Nq, H, W) uint8
            scores: (Nq,) float32
        """
        if len(boxes) == 0:
            return None, None

        # SAM3 expects list-of-list of boxes + labels
        input_boxes = [boxes.tolist()]
        input_boxes_labels = [[1] * len(boxes)]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # PCS post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]  # single image

        pred_masks_t = results["masks"]
        pred_boxes_t = results["boxes"]
        pred_scores_t = results["scores"]

        pred_boxes = pred_boxes_t.to(torch.float32).cpu().numpy()
        pred_scores = pred_scores_t.to(torch.float32).cpu().numpy()

        if isinstance(pred_masks_t, torch.Tensor):
            pred_masks_np = pred_masks_t.to(torch.uint8).cpu().numpy()
        else:
            pred_masks_np = np.asarray(pred_masks_t, dtype=np.uint8)

        if len(pred_masks_np) == 0:
            return None, None

        # ---- Map predictions to queries via IoU ----
        iou_matrix = self._box_iou_matrix(boxes, pred_boxes)
        best_idx = iou_matrix.argmax(axis=1)

        out_masks = []
        out_scores = []
        for qi, pi in enumerate(best_idx):
            out_masks.append(pred_masks_np[pi])
            out_scores.append(float(pred_scores[pi]))

        masks = np.stack(out_masks, axis=0)
        scores = np.array(out_scores, dtype=np.float32)
        return masks, scores

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)
