# Inference-safe re-exports ONLY — never import the train_* modules here (see package __init__).
from canopyrs.engine.frameworks.detectron2.augmentation import AugmentationAdder
from canopyrs.engine.frameworks.detectron2.cfg import (get_base_detectron2_model_cfg,
                                                       get_base_detrex_model_cfg)
