from random import random

import numpy as np

from detectron2.data.transforms import Augmentation, RandomRotation, NoOpTransform, CropTransform, ResizeTransform, \
    ResizeShortestEdge, RandomContrast, RandomBrightness, RandomFlip, RandomSaturation
from engine.config_parsers import DetectorConfig


class RandomRotationWithProb(Augmentation):
    def __init__(self, angle, prob=0.5):
        """
        angle: the angle range for rotation (e.g., [-30, 30])
        p: probability of applying the rotation
        """
        super().__init__()
        self.angle = angle
        self.prob = prob

    def get_transform(self, image):
        # With probability p, apply the rotation; otherwise, return an identity transform.
        if random() < self.prob:
            return RandomRotation(angle=self.angle).get_transform(image)
        else:
            return NoOpTransform()


class SquareRandomCrop(Augmentation):
    """
    Randomly crop a square region from an image.
    The side length of the square is uniformly sampled from [min_side, max_side],
    constrained by the image dimensions.
    """
    def __init__(self, min_side: int, max_side: int):
        """
        Args:
            min_side (int): minimum side length in pixels.
            max_side (int): maximum side length in pixels.
        """
        super().__init__()
        assert min_side <= max_side, "min_side should be <= max_side"
        self.min_side = min_side
        self.max_side = max_side

    def get_crop_size(self, image_size):
        h, w = image_size
        # The maximum allowed side length is limited by the image dimensions
        max_possible = min(h, w, self.max_side)
        # Ensure that the minimum is not greater than max_possible
        min_possible = min(self.min_side, max_possible)
        # Sample a square side length from [min_possible, max_possible]
        side = np.random.randint(min_possible, max_possible + 1)
        return side, side

    def get_transform(self, image):
        h, w = image.shape[:2]
        crop_h, crop_w = self.get_crop_size((h, w))
        # Ensure the crop size does not exceed the image dimensions.
        assert h >= crop_h and w >= crop_w, "Crop size is larger than image dimensions."
        # Randomly select the top-left coordinate for cropping.
        y0 = np.random.randint(0, h - crop_h + 1)
        x0 = np.random.randint(0, w - crop_w + 1)
        return CropTransform(x0, y0, crop_w, crop_h)


class SquareRandomCropWithBoxDiscard(Augmentation):
    """
    Randomly crop a square region from an image. The side length of the square is sampled
    from a given range (crop_range). The crop is chosen only once (no iterative search). After
    cropping, bounding boxes are adjusted; any box whose intersection with the crop covers less
    than `min_intersection_ratio` of its original area is discarded.

    Args:
        crop_range (tuple): (min_side, max_side) specifying the range (in pixels) of the square side.
        min_intersection_ratio (float): Minimum fraction of a box’s area that must lie within the crop.
    """
    def __init__(self, crop_range, min_intersection_ratio: float):
        super().__init__()
        self.crop_range = crop_range
        self.min_intersection_ratio = min_intersection_ratio

    def get_transform(self, image: np.ndarray, boxes: np.ndarray = None):
        """
        Args:
            image (np.ndarray): the image as a HxWxC array.
            boxes (np.ndarray): an array of bounding boxes in [x1, y1, x2, y2] format.
                                If None, only the image will be cropped.
        Returns:
            A CropTransform object. If boxes are provided, the transform’s attribute
            `new_boxes` will contain the adjusted boxes (boxes that did not meet the threshold are dropped).
        """
        H, W = image.shape[:2]
        min_side, max_side = self.crop_range

        # Clamp the maximum allowed crop side to the image dimensions.
        max_possible = min(H, W)
        if max_side > max_possible:
            max_side = max_possible
        if min_side > max_possible:
            # If even the smallest allowed crop is larger than the image,
            # simply do nothing.
            return NoOpTransform()

        # Sample a side length uniformly from [min_side, max_side]
        side = np.random.randint(min_side, max_side + 1)

        # Randomly choose the top-left coordinate for the crop
        x0 = np.random.randint(0, W - side + 1)
        y0 = np.random.randint(0, H - side + 1)

        crop_transform = CropTransform(x0, y0, side, side)

        # If there are no bounding boxes, simply return the crop transform.
        if boxes is None or len(boxes) == 0:
            crop_transform.new_boxes = np.empty((0, 4))
            return crop_transform

        new_boxes = []
        for box in boxes:
            # Each box is assumed to be in [x1, y1, x2, y2] format.
            # Compute the area of the original box.
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area <= 0:
                continue

            # Compute the coordinates of the intersection between the box and crop.
            inter_x1 = max(box[0], x0)
            inter_y1 = max(box[1], y0)
            inter_x2 = min(box[2], x0 + side)
            inter_y2 = min(box[3], y0 + side)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            # Keep the box only if the intersection is large enough.
            if inter_area / box_area >= self.min_intersection_ratio:
                # Adjust the box coordinates relative to the crop.
                new_box = [inter_x1 - x0, inter_y1 - y0, inter_x2 - x0, inter_y2 - y0]
                new_boxes.append(new_box)

        if new_boxes:
            crop_transform.new_boxes = np.array(new_boxes)
        else:
            crop_transform.new_boxes = np.empty((0, 4))

        return crop_transform


class ConditionalResize(Augmentation):
    """
    Upscales an image if its smallest side is below a given minimum.
    If the image’s smallest side is already >= min_size, it does nothing.
    """

    def __init__(self, min_size):
        """
        Args:
            min_size (int): The minimum allowed size for the image's shortest side.
        """
        self.min_size = min_size

    def get_transform(self, image):
        orig_h, orig_w = image.shape[:2]
        # Determine the scale factor only if the smallest side is less than min_size.
        scale = max(1.0, self.min_size / min(orig_h, orig_w))

        if scale == 1.0:
            # Return an identity (no-op) transform if no scaling is needed.
            return NoOpTransform()
        else:
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            # The ResizeTransform constructor takes the original and new dimensions.
            return ResizeTransform(0, 0, new_w, new_h)


class AugmentationAdder:
    @staticmethod
    def modify_detectron2_augmentation_config(config: DetectorConfig, cfg):
        """
        Insert your augmentation parameters from `config` into the detectron2 `cfg`.

        Typically, you can store them inside a sub-config node such as:
            cfg.AUGMENTATION.<PARAM_NAME> = ...
        or
            cfg.INPUT.<PARAM_NAME> = ...
        """
        # Example structure: store them all under cfg.AUGMENTATION
        # (Create the node if it doesn't exist)
        if not hasattr(cfg, "AUGMENTATION"):
            from detectron2.config import CfgNode as CN
            cfg.AUGMENTATION = CN()

        cfg.AUGMENTATION.IMAGE_SIZE = config.augmentation_image_size
        cfg.AUGMENTATION.FLIP_HORIZONTAL = config.augmentation_flip_horizontal
        cfg.AUGMENTATION.FLIP_VERTICAL = config.augmentation_flip_vertical
        cfg.AUGMENTATION.ROTATION = config.augmentation_rotation
        cfg.AUGMENTATION.ROTATION_PROB = config.augmentation_rotation_prob
        cfg.AUGMENTATION.BRIGHTNESS = config.augmentation_brightness
        cfg.AUGMENTATION.CONTRAST = config.augmentation_contrast
        cfg.AUGMENTATION.SATURATION = config.augmentation_saturation
        cfg.AUGMENTATION.CROP_SIZE_RANGE = config.augmentation_train_crop_size_range
        cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO = config.augmentation_crop_min_intersection_ratio

    @staticmethod
    def get_augmentation_detectron2_train(cfg):
        """
        Build a list of detectron2 augmentations (or custom augmentations)
        based on the parameters in cfg.AUGMENTATION.
        """
        augs = []

        # Conditional resize
        augs.append(
            ConditionalResize(min_size=cfg.AUGMENTATION.IMAGE_SIZE)
        )

        # Horizontal flip
        if cfg.AUGMENTATION.FLIP_HORIZONTAL:
            augs.append(RandomFlip(prob=0.5, horizontal=True, vertical=False))

        # Vertical flip
        if cfg.AUGMENTATION.FLIP_VERTICAL:
            augs.append(RandomFlip(prob=0.5, horizontal=False, vertical=True))

        # Random rotation (custom augmentation)
        if cfg.AUGMENTATION.ROTATION:
            augs.append(
                RandomRotationWithProb(
                    angle=cfg.AUGMENTATION.ROTATION,
                    prob=cfg.AUGMENTATION.ROTATION_PROB
                )
            )

        # Random brightness
        if cfg.AUGMENTATION.BRIGHTNESS:
            brightness_min = 1.0 - cfg.AUGMENTATION.BRIGHTNESS
            brightness_max = 1.0 + cfg.AUGMENTATION.BRIGHTNESS
            augs.append(RandomBrightness(intensity_min=brightness_min, intensity_max=brightness_max))

        # Random contrast
        if cfg.AUGMENTATION.CONTRAST:
            contrast_min = 1.0 - cfg.AUGMENTATION.CONTRAST
            contrast_max = 1.0 + cfg.AUGMENTATION.CONTRAST
            augs.append(RandomContrast(intensity_min=contrast_min, intensity_max=contrast_max))

        if cfg.AUGMENTATION.SATURATION:
            saturation_min = 1.0 - cfg.AUGMENTATION.SATURATION
            saturation_max = 1.0 + cfg.AUGMENTATION.SATURATION
            augs.append(RandomSaturation(intensity_min=saturation_min, intensity_max=saturation_max))

        # Random crop (custom) - SquareRandomCropWithBoxDiscard
        augs.append(
            SquareRandomCropWithBoxDiscard(
                crop_range=cfg.AUGMENTATION.CROP_SIZE_RANGE,
                min_intersection_ratio=cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO
            )
        )

        # Resize to final, fixed size
        augs.append(
            ResizeShortestEdge(
                short_edge_length=[cfg.AUGMENTATION.IMAGE_SIZE],
                max_size=cfg.AUGMENTATION.IMAGE_SIZE,
                sample_style="choice"
            )
        )

        return augs

    @staticmethod
    def get_augmentation_detectron2_test(cfg):
        """
        Build the test-time augmentation. Typically just a fixed resize.
        """
        return [
            ResizeShortestEdge(
                short_edge_length=[cfg.AUGMENTATION.IMAGE_SIZE],
                max_size=cfg.AUGMENTATION.IMAGE_SIZE,
                sample_style="choice"
            )
        ]

    @staticmethod
    def get_augmentation_detrex_train(config: DetectorConfig):
        new_augmentations = []

        # Resize small images
        new_augmentations.append(
            f"{{'_target_': 'engine.models.detector.detectron2.augmentation.ConditionalResize',"
            f" 'min_size': {config.augmentation_image_size}}}"
        )

        # Horizontal flip
        if config.augmentation_flip_horizontal:
            new_augmentations.append(
                f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.RandomFlip',"
                f" 'horizontal': True,"
                f" 'vertical': False,"
                f" 'prob': 0.5}}"
            )

        # Vertical flip
        if config.augmentation_flip_vertical:
            new_augmentations.append(
                f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.RandomFlip',"
                f" 'horizontal': False,"
                f" 'vertical': True,"
                f" 'prob': 0.5}}"
            )

        # Random rotation
        if config.augmentation_rotation:
            new_augmentations.append(
                f"{{'_target_': 'engine.models.detector.detectron2.augmentation.RandomRotationWithProb',"
                f" 'angle': {config.augmentation_rotation},"
                f" 'prob': {config.augmentation_rotation_prob}}}"
            )

        # Random brightness
        if config.augmentation_brightness:
            new_augmentations.append(
                f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.RandomBrightness',"
                f" 'intensity_min': {1.0 - config.augmentation_brightness},"
                f" 'intensity_max': {1.0 + config.augmentation_brightness}}}"
            )

        # Random contrast
        if config.augmentation_contrast:
            new_augmentations.append(
                f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.RandomContrast',"
                f" 'intensity_min': {1.0 - config.augmentation_contrast},"
                f" 'intensity_max': {1.0 + config.augmentation_contrast}}}"
            )

        # Random saturation
        if config.augmentation_saturation:
            new_augmentations.append(
                f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.RandomSaturation',"
                f" 'intensity_min': {1.0 - config.augmentation_saturation},"
                f" 'intensity_max': {1.0 + config.augmentation_saturation}}}"
            )

        # Random crop
        new_augmentations.append(
            f"{{'_target_': 'engine.models.detector.detectron2.augmentation.SquareRandomCropWithBoxDiscard',"
            f" 'crop_range': {config.augmentation_train_crop_size_range},"
            f" 'min_intersection_ratio': {config.augmentation_crop_min_intersection_ratio}}}"
        )

        # Resize to a final, fixed size
        new_augmentations.append(
            f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.ResizeShortestEdge',"
            f" 'short_edge_length': [{config.augmentation_image_size}],"
            f" 'max_size': {config.augmentation_image_size},"
            f" 'sample_style': 'choice'}}"
        )

        return new_augmentations

    @staticmethod
    def get_augmentation_detrex_test(config: DetectorConfig):
        # Resize to a final, fixed size
        new_augmentations = [
            f"{{'_target_': 'detectron2.data.transforms.augmentation_impl.ResizeShortestEdge',"
            f" 'short_edge_length': [{config.augmentation_image_size}],"
            f" 'max_size': {config.augmentation_image_size},"
            f" 'sample_style': 'choice'}}"
        ]
        return new_augmentations