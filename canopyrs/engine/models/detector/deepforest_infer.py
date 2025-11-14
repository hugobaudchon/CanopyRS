import cv2
import numpy as np
import torch

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.models.detector.detector_base import DetectorWrapperBase
from canopyrs.engine.models.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("deepforest")
class DeepForestWrapper(DetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        from deepforest import main

        deepf = main.deepforest()
        deepf.load_model(model_name=config.checkpoint_path, revision="main")

        self.model = deepf.model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def forward(self, images, targets=None):
        """
        images: list[Tensor], each [C, H, W], values in [0, 1]
        returns: list[dict(boxes, scores, labels)] on CPU
        """
        preds = []

        for img in images:
            img_cpu = img.detach().cpu()
            np_img = img_cpu.permute(1, 2, 0).numpy()  # [H, W, C], float32 in [0, 1]
            orig_h, orig_w = np_img.shape[:2]

            scale = 1.0
            if max(orig_h, orig_w) > 1000:
                if orig_h >= orig_w:
                    scale = 1000.0 / orig_h
                else:
                    scale = 1000.0 / orig_w
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                np_img_rs = cv2.resize(
                    np_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                np_img_rs = np_img

            img_t = (
                torch.from_numpy(np_img_rs)
                .permute(2, 0, 1)   # HWC -> CHW
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                output = self.model([img_t])[0]

            # Bring predictions back to CPU
            boxes = output["boxes"].detach().cpu()       # [N, 4]
            scores = output["scores"].detach().cpu()     # [N]
            labels = output.get("labels", torch.zeros(len(boxes), dtype=torch.int64))
            labels = labels.detach().cpu()

            # Undo resize
            if scale != 1.0:
                boxes /= scale

            # If there are no boxes, keep empty tensors with correct shapes
            if boxes.numel() == 0:
                preds.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "scores": torch.empty((0,), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64),
                    }
                )
            else:
                preds.append(
                    {
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    }
                )

        return preds
