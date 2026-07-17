"""
RSPrompter inference wrapper for CanopyRS segmentation pipeline.

This module provides inference capabilities for RSPrompter models (both anchor and query variants)
trained using MMDetection framework. It loads models from MMDetection config files and checkpoints.
"""

import multiprocessing
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.extras import require_extra
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY


def _get_rsprompter_root() -> Path:
    """The mmdet_rsprompter fork checkout (provides the `mmdet` package + its config files).

    Derived from the installed mmdet package itself (editable install of the fork), so the
    checkout can live anywhere; falls back to the legacy location next to the CanopyRS root.
    """
    import mmdet
    fork_root = Path(mmdet.__file__).resolve().parents[1]
    if (fork_root / "configs" / "rsprompter").exists():
        return fork_root
    legacy = Path(__file__).resolve().parents[4] / "RSPrompter"
    if (legacy / "configs" / "rsprompter").exists():
        return legacy
    raise RuntimeError(
        "mmdet is installed but the RSPrompter fork checkout (with configs/rsprompter/) was "
        f"not found at {fork_root} or {legacy}. Install the fork editable — see "
        "RSPROMPTER_NOTES.md — or pass an absolute `architecture` config path."
    )


def _setup_rsprompter_imports():
    """
    Register everything RSPrompter model building needs.

    The mmdet_rsprompter fork installs AS the real `mmdet` package, so this is just standard
    mmdet registration plus the fork's custom modules — the legacy sys.path/sys.modules
    aliasing (clone-based scheme) and the mmengine parent-registry copying hack are gone:
    ``init_default_scope='mmdet'`` makes mmengine's scope switching resolve the mmdet registry
    (LN2d, DetDataPreprocessor, ...) properly.
    """
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)      # stock mmdet modules + 'mmdet' scope
    import mmdet.rsprompter                            # noqa: F401  custom models (RSPrompterAnchor, LN2d, ...)
    import mmdet.rsprompter.selvamask_augmentations    # noqa: F401  custom transforms configs reference


@SEGMENTER_REGISTRY.register('rsprompter_anchor', 'rsprompter_query')
class RSPrompterSegmenterWrapper(SegmenterWrapperBase):
    """
    Segmenter wrapper for RSPrompter models (both anchor and query variants).
    
    RSPrompter is a prompt-based segmentation model built on top of SAM (Segment Anything Model)
    and uses MMDetection as the training/inference framework.
    
    Args:
        config (SegmenterConfig): Configuration object containing:
            - model: 'rsprompter_anchor' or 'rsprompter_query'
            - architecture: Path to the MMDetection config file (relative to RSPrompter/configs/rsprompter)
              or absolute path
            - checkpoint_path: Path to the trained model checkpoint
    
    Example:
        >>> config = SegmenterConfig(
        ...     model='rsprompter_anchor',
        ...     architecture='rsprompter_anchor-whu.py',
        ...     checkpoint_path='/path/to/checkpoint.pth'
        ... )
        >>> wrapper = RSPrompterSegmenterWrapper(config)
    """
    
    REQUIRES_BOX_PROMPT = False

    @classmethod
    def preflight(cls, config):
        """This wrapper registers without mmdet (all framework imports are deferred to
        __init__), so verify the mmdet stack here — at pipeline construction — instead of
        dying with a raw ModuleNotFoundError deep inside model build."""
        require_extra('mmdet', packages=('mmengine', 'mmdet', 'mmcv'))
        _get_rsprompter_root()   # RSPrompter clone must exist next to the CanopyRS root

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)
        self.preflight(config)

        # Setup RSPrompter imports (this also registers the custom modules)
        rsprompter_root = _get_rsprompter_root()
        _setup_rsprompter_imports()
        
        # Import MMDetection/MMEngine components
        from mmengine.config import Config
        # Use mmdet.registry.MODELS (from RSPrompter's mmdet package) which is the 
        # child registry where RSPrompter models are registered
        from mmdet.registry import MODELS as MMDET_MODELS
        
        # Determine config path
        if os.path.isabs(self.config.architecture):
            config_path = self.config.architecture
        else:
            # Assume relative path from RSPrompter/configs/rsprompter
            config_path = str(rsprompter_root / "configs" / "rsprompter" / self.config.architecture)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"RSPrompter config file not found at {config_path}. "
                f"Please check the architecture path in your configuration."
            )
        
        print(f"Loading RSPrompter config from: {config_path}")
        print(f"Loading RSPrompter checkpoint from: {self.config.checkpoint_path}")
        
        # Load config
        self.cfg = Config.fromfile(config_path)
        
        # Build model from config
        self.model = MMDET_MODELS.build(self.cfg.model)
        self.model.to(self.device)
        
        # Load checkpoint
        if self.config.checkpoint_path:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                self.model, 
                self.config.checkpoint_path, 
                map_location=str(self.device)
            )
        
        self.model.eval()
        
        # Setup preprocessing from config
        self._setup_preprocessing()
        
        print(f"RSPrompter model loaded successfully: {self.config.model}")

    def _setup_preprocessing(self):
        """Setup image preprocessing parameters from the config."""
        # Default values (ImageNet normalization scaled for [0, 255])
        default_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        default_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        
        # Extract preprocessing parameters from data_preprocessor config
        if hasattr(self.cfg, 'data_preprocessor') and self.cfg.data_preprocessor is not None:
            dp = self.cfg.data_preprocessor
            # Handle both dict-like and object-like access
            if isinstance(dp, dict):
                self.mean = np.array(dp.get('mean', default_mean))
                self.std = np.array(dp.get('std', default_std))
                self.bgr_to_rgb = dp.get('bgr_to_rgb', True)
                self.pad_size_divisor = dp.get('pad_size_divisor', 32)
            else:
                # Config object
                self.mean = np.array(getattr(dp, 'mean', default_mean))
                self.std = np.array(getattr(dp, 'std', default_std))
                self.bgr_to_rgb = getattr(dp, 'bgr_to_rgb', True)
                self.pad_size_divisor = getattr(dp, 'pad_size_divisor', 32)
        else:
            self.mean = np.array(default_mean)
            self.std = np.array(default_std)
            self.bgr_to_rgb = True
            self.pad_size_divisor = 32
        
        # Build data preprocessor from config for proper handling
        from mmdet.registry import MODELS as MMDET_MODELS
        if hasattr(self.cfg, 'data_preprocessor') and self.cfg.data_preprocessor is not None:
            try:
                self.data_preprocessor = MMDET_MODELS.build(self.cfg.data_preprocessor)
                self.data_preprocessor.to(self.device)
                self.data_preprocessor.eval()
            except Exception as e:
                print(f"Warning: Could not build data_preprocessor, using manual preprocessing: {e}")
                self.data_preprocessor = None
        else:
            self.data_preprocessor = None

    def _create_data_sample(self, image: np.ndarray) -> "DetDataSample":
        """
        Create a DetDataSample for the given image.
        
        Args:
            image: Image as numpy array in HWC format with values in [0, 255]
            
        Returns:
            DetDataSample with proper metadata
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        h, w = image.shape[:2]
        
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': (h, w),
            'ori_shape': (h, w),
            'pad_shape': (h, w),
            'scale_factor': (1.0, 1.0),
        })
        
        # Initialize empty pred_instances
        data_sample.pred_instances = InstanceData()
        
        return data_sample

    def _pad_to_divisor(self, image: np.ndarray) -> np.ndarray:
        """Pad image dimensions to be divisible by pad_size_divisor."""
        h, w = image.shape[:2]
        new_h = ((h + self.pad_size_divisor - 1) // self.pad_size_divisor) * self.pad_size_divisor
        new_w = ((w + self.pad_size_divisor - 1) // self.pad_size_divisor) * self.pad_size_divisor
        
        if new_h == h and new_w == w:
            return image
        
        # Pad with zeros
        if image.ndim == 3:
            padded = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
            padded[:h, :w, :] = image
        else:
            padded = np.zeros((new_h, new_w), dtype=image.dtype)
            padded[:h, :w] = image
            
        return padded
    
    def _resize_image(self, image: np.ndarray, target_size: int = 1024) -> tuple:
        """
        Resize image so the longest side is target_size while maintaining aspect ratio.
        SAM/RSPrompter expects 1024x1024 input, so we resize to fit within that.
        
        Args:
            image: Image as numpy array in HWC format
            target_size: Target size for the longest side (default 1024)
            
        Returns:
            Tuple of (resized_image, scale_factor_h, scale_factor_w)
        """
        import cv2
        
        h, w = image.shape[:2]
        
        # Compute scale to fit longest side to target_size
        scale = target_size / max(h, w)
        
        # If already small enough, no need to resize
        if scale >= 1.0:
            return image, 1.0, 1.0
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize using bilinear interpolation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Return scale factors for rescaling predictions back to original size
        scale_factor_h = h / new_h
        scale_factor_w = w / new_w
        
        return resized, scale_factor_h, scale_factor_w

    def forward(self,
                images: List[np.array],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue,
                **kwargs):
        """
        Perform inference on a batch of images.

        Args:
            images (List[np.array]): A batch of images as numpy arrays in CHW format with values in [0, 1]
            tiles_idx (List[int]): A list of tile indices corresponding to the images.
            queue (multiprocessing.JoinableQueue): A queue to put the results in.
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        with torch.no_grad():
            for image, tile_idx in zip(images, tiles_idx):
                # Get original dimensions
                orig_c, orig_h, orig_w = image.shape
                
                # Convert from CHW [0, 1] to HWC [0, 255]
                image_hwc = image.transpose(1, 2, 0) * 255.0
                image_hwc = image_hwc.astype(np.float32)
                
                # Ensure we have 3 channels
                if image_hwc.shape[2] > 3:
                    image_hwc = image_hwc[:, :, :3]
                
                # Convert RGB to BGR if model expects BGR (most mmdet models do)
                # bgr_to_rgb=True in config means input is BGR and will be converted to RGB
                # So for our RGB input, we need to convert to BGR first
                if self.bgr_to_rgb:
                    image_hwc = image_hwc[:, :, ::-1].copy()
                
                # Resize image to fit model's expected input size (1024x1024 for SAM)
                # This is equivalent to what test_pipeline does with DeterministicResizeWithinRange
                image_resized, scale_h, scale_w = self._resize_image(image_hwc, target_size=1024)
                resized_h, resized_w = image_resized.shape[:2]
                
                # Pad image to be divisible by pad_size_divisor
                image_padded = self._pad_to_divisor(image_resized)
                padded_h, padded_w = image_padded.shape[:2]
                
                # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
                image_tensor = torch.from_numpy(image_padded.transpose(2, 0, 1)).float()
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Create data sample with metadata
                # Note: scale_factor is used by the model to rescale predictions back to ori_shape
                data_sample = self._create_data_sample(image_padded)
                data_sample.set_metainfo({
                    'img_shape': (resized_h, resized_w),
                    'ori_shape': (orig_h, orig_w),
                    'pad_shape': (padded_h, padded_w),
                    'scale_factor': (scale_w, scale_h),  # (w_scale, h_scale) for rescaling
                })
                
                # Prepare inputs for model
                # Apply data preprocessor if available
                if self.data_preprocessor is not None:
                    data = {
                        'inputs': image_tensor,
                        'data_samples': [data_sample]
                    }
                    data = self.data_preprocessor(data, training=False)
                    inputs = data['inputs']
                    data_samples = data['data_samples']
                else:
                    # Manual normalization
                    mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
                    inputs = (image_tensor - mean) / std
                    data_samples = [data_sample]
                
                # Run inference
                results = self.model.predict(inputs, data_samples, rescale=True)
                
                # Process results
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                else:
                    result = results
                
                # Extract predictions from DetDataSample
                if hasattr(result, 'pred_instances') and result.pred_instances is not None:
                    instances = result.pred_instances
                    
                    # Get masks
                    if hasattr(instances, 'masks') and instances.masks is not None:
                        masks = instances.masks
                        
                        # Handle different mask types
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().numpy()
                        elif hasattr(masks, 'masks'):
                            # BitmapMasks object from mmdet
                            masks = masks.masks
                        elif hasattr(masks, 'to_ndarray'):
                            masks = masks.to_ndarray()
                        
                        # Ensure masks are boolean/uint8
                        if masks.dtype == np.float32 or masks.dtype == np.float64:
                            masks = (masks > 0.5).astype(np.uint8)
                        elif masks.dtype == bool:
                            masks = masks.astype(np.uint8)
                        
                        # Ensure masks are in the right format (N, H, W)
                        if masks.ndim == 4:
                            masks = masks[:, 0, :, :]
                        
                        # Resize masks to original size if they don't match
                        # This handles cases where rescale=True didn't work as expected
                        # or masks are at resized resolution
                        mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                        if mask_h != orig_h or mask_w != orig_w:
                            import cv2
                            resized_masks = np.zeros((masks.shape[0], orig_h, orig_w), dtype=np.uint8)
                            for i in range(masks.shape[0]):
                                # Use nearest neighbor interpolation for masks to preserve binary values
                                resized_masks[i] = cv2.resize(
                                    masks[i].astype(np.uint8), 
                                    (orig_w, orig_h), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                            masks = resized_masks
                        
                        # Get scores
                        if hasattr(instances, 'scores') and instances.scores is not None:
                            scores = instances.scores
                            if isinstance(scores, torch.Tensor):
                                scores = scores.cpu().numpy()
                        else:
                            scores = np.ones(masks.shape[0], dtype=np.float32)
                        
                        image_size = (orig_h, orig_w)
                        n_masks_processed = 0
                        
                        # No box prompts for RSPrompter
                        image_boxes_object_ids = [None] * masks.shape[0]
                        
                        _ = self.queue_masks(
                            image_boxes_object_ids,
                            masks,
                            image_size,
                            scores,
                            tile_idx,
                            n_masks_processed,
                            queue
                        )
