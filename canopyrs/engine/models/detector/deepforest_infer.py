import cv2
import numpy as np
import torch

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.models.detector.detector_base import DetectorWrapperBase
from canopyrs.engine.models.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register('deepforest')
class DeepForestWrapper(DetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        from deepforest import main
        self.model = main.deepforest()
        self.model.load_model(model_name=config.checkpoint_path, revision='main')

    def forward(self, images, targets=None):
        preds = []
        for img in images:
            np_img = img.permute(1, 2, 0).cpu().numpy() * 255.0
            orig_h, orig_w = np_img.shape[:2]

            scale = 1.0
            if max(orig_h, orig_w) > 1000:
                if orig_h >= orig_w:
                    scale = 1000.0 / orig_h
                else:
                    scale = 1000.0 / orig_w
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                np_img_rs = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                np_img_rs = np_img

            df = self.model.predict_image(np_img_rs)

            if df is None or df.empty:
                preds.append({
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                })
                continue

            boxes = torch.as_tensor(df[["xmin", "ymin", "xmax", "ymax"]].values,
                                    dtype=torch.float32)
            scores = torch.as_tensor(df["score"].values, dtype=torch.float32)

            # undo resize on boxes
            if scale != 1.0:  # only applied when we resized
                boxes /= scale

            if "label" in df.columns and np.issubdtype(df["label"].dtype, np.number):
                labels = torch.as_tensor(df["label"].values, dtype=torch.int64)
            else:
                labels = torch.zeros(len(df), dtype=torch.int64)

            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        return preds

