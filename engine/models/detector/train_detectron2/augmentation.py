from random import random

import cv2
import numpy as np

from detectron2.data.transforms import Augmentation, Transform, RandomRotation, NoOpTransform, CropTransform, ResizeTransform, \
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


class RandomHueTransform(Transform):
    def __init__(self, hue_delta: int):
        """
        Args:
            hue_delta (int): Amount to shift the hue channel. This value will be added
                             to the hue channel (in OpenCV’s 0-179 scale) and wrapped around.
        """
        self.hue_delta = hue_delta

    def apply_image(self, img):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        # Modify the hue channel and wrap-around using modulo 180
        hsv[..., 0] = (hsv[..., 0] + self.hue_delta) % 180
        # Convert back to uint8 and then to BGR
        hsv = hsv.astype(np.uint8)
        img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_aug

    def apply_coords(self, coords):
        # Hue changes do not affect coordinates.
        return coords

class RandomHueAugmentation(Augmentation):
    def __init__(self, hue_delta_range: tuple = (-10, 10)):
        """
        Args:
            hue_delta_range (tuple): A tuple specifying the minimum and maximum delta to apply
                                     to the hue channel.
        """
        self.hue_delta_range = hue_delta_range

    def get_transform(self, image):
        # Randomly select a hue delta within the provided range.
        hue_delta = random.randint(self.hue_delta_range[0], self.hue_delta_range[1])
        return RandomHueTransform(hue_delta)


class AugmentationAdder:
    @staticmethod
    def modify_detectron2_augmentation_config(config: DetectorConfig, cfg):
        """
        Insert your augmentation parameters from `config` into the train_detectron2 `cfg`.

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
        cfg.AUGMENTATION.HUE = config.augmentation_hue
        cfg.AUGMENTATION.CROP_SIZE_RANGE = config.augmentation_train_crop_size_range
        cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO = config.augmentation_crop_min_intersection_ratio

    @staticmethod
    def get_augmentation_detectron2_train(cfg):
        """
        Build a list of train_detectron2 augmentations (or custom augmentations)
        based on the parameters in cfg.AUGMENTATION.
        """
        augs = AugmentationAdder._get_augmentation_list_train(
            image_size=cfg.AUGMENTATION.IMAGE_SIZE,
            flip_vertical=cfg.AUGMENTATION.FLIP_VERTICAL,
            flip_horizontal=cfg.AUGMENTATION.FLIP_HORIZONTAL,
            rotation=cfg.AUGMENTATION.ROTATION,
            rotation_prob=cfg.AUGMENTATION.ROTATION_PROB,
            brightness=cfg.AUGMENTATION.BRIGHTNESS,
            contrast=cfg.AUGMENTATION.CONTRAST,
            saturation=cfg.AUGMENTATION.SATURATION,
            hue=cfg.AUGMENTATION.HUE,
            crop_size_range=cfg.AUGMENTATION.CROP_SIZE_RANGE,
            crop_min_intersection_ratio=cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO
        )

        return augs

    @staticmethod
    def get_augmentation_detectron2_test(cfg):
        """
        Build the test-time augmentation. Typically just a fixed resize.
        """
        augs = AugmentationAdder._get_augmentation_list_test(
            image_size=cfg.AUGMENTATION.IMAGE_SIZE
        )

        return augs

    @staticmethod
    def get_augmentation_detrex_train(config: DetectorConfig):
        augs = AugmentationAdder._get_augmentation_list_train(
            image_size=config.augmentation_image_size,
            flip_vertical=config.augmentation_flip_vertical,
            flip_horizontal=config.augmentation_flip_horizontal,
            rotation=config.augmentation_rotation,
            rotation_prob=config.augmentation_rotation_prob,
            brightness=config.augmentation_brightness,
            contrast=config.augmentation_contrast,
            saturation=config.augmentation_saturation,
            hue=config.augmentation_hue,
            crop_size_range=config.augmentation_train_crop_size_range,
            crop_min_intersection_ratio=config.augmentation_crop_min_intersection_ratio
        )

        return augs

    @staticmethod
    def get_augmentation_detrex_test(config: DetectorConfig):
        # Resize to a final, fixed size
        augs = AugmentationAdder._get_augmentation_list_test(
            image_size=config.augmentation_image_size
        )

        return augs

    @staticmethod
    def _get_augmentation_list_train(
            image_size: int,
            flip_vertical: bool,
            flip_horizontal: bool,
            rotation: float,
            rotation_prob: float,
            brightness: float,
            contrast: float,
            saturation: float,
            hue: float,
            crop_size_range: tuple or list,
            crop_min_intersection_ratio: float,
    ):
        """
        Build a list of Detectron2 augmentations based on the given parameters.

        Args:
            image_size (int): minimum size used for ConditionalResize and final ResizeShortestEdge.
            flip_vertical (bool): if True, applies RandomFlip( vertical=True ).
            flip_horizontal (bool): if True, applies RandomFlip( horizontal=True ).
            rotation (float): if non-zero, range of angles (e.g., [-rotation, rotation]) for random rotation.
            rotation_prob (float): probability of applying the random rotation.
            brightness (float): amount for RandomBrightness (e.g., 0.2 means ±20%).
            contrast (float): amount for RandomContrast (e.g., 0.2 means ±20%).
            saturation (float): amount for RandomSaturation (e.g., 0.2 means ±20%).
            hue (int): amount for RandomHue (e.g., ±hue, in [0, 255]).
            crop_size_range (tuple or list): (min_side, max_side) range for SquareRandomCropWithBoxDiscard.
            crop_min_intersection_ratio (float): ratio threshold to keep boxes that partially fall outside the crop.

        Returns:
            list[Augmentation]: a list of Detectron2-compatible augmentation objects.
        """
        augs = []

        # 1) Conditional resize
        augs.append(
            ConditionalResize(min_size=image_size)
        )

        # 2) Horizontal flip
        if flip_horizontal:
            augs.append(RandomFlip(prob=0.5, horizontal=True, vertical=False))

        # 3) Vertical flip
        if flip_vertical:
            augs.append(RandomFlip(prob=0.5, horizontal=False, vertical=True))

        # 4) Random rotation (custom augmentation)
        #    If rotation is non-zero, we apply a random rotation from [-rotation, rotation].
        if rotation:
            # e.g., angle = [-rotation, rotation]
            angle_range = (-rotation, rotation)
            augs.append(
                RandomRotationWithProb(angle=angle_range, prob=rotation_prob)
            )

        # 5) Random brightness
        if brightness:
            brightness_min = 1.0 - brightness
            brightness_max = 1.0 + brightness
            augs.append(RandomBrightness(intensity_min=brightness_min, intensity_max=brightness_max))

        # 6) Random contrast
        if contrast:
            contrast_min = 1.0 - contrast
            contrast_max = 1.0 + contrast
            augs.append(RandomContrast(intensity_min=contrast_min, intensity_max=contrast_max))

        # 7) Random saturation
        if saturation:
            saturation_min = 1.0 - saturation
            saturation_max = 1.0 + saturation
            augs.append(RandomSaturation(intensity_min=saturation_min, intensity_max=saturation_max))

        # 8) Random hue
        if hue:
            augs.append(RandomHueAugmentation(hue_delta_range=(-hue, hue)))

        # 9) Random crop (custom) - SquareRandomCropWithBoxDiscard
        augs.append(
            SquareRandomCropWithBoxDiscard(
                crop_range=crop_size_range,
                min_intersection_ratio=crop_min_intersection_ratio
            )
        )

        # 10) Resize to final, fixed size
        augs.append(
            ResizeShortestEdge(
                short_edge_length=[image_size],
                max_size=image_size,
                sample_style="choice"
            )
        )

        return augs

    @staticmethod
    def _get_augmentation_list_test(image_size: int):
        """
        Build a list of Detectron2 augmentations for test-time based on the given parameters.

        Args:
            image_size (int): final image size for resizing.

        Returns:
            list[Augmentation]: a list of Detectron2-compatible augmentation objects.
        """
        return [
            ResizeShortestEdge(
                short_edge_length=[image_size],
                max_size=image_size,
                sample_style="choice"
            )
        ]
