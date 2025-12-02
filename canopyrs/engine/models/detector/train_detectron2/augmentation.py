from random import random, randint

import cv2
import numpy as np

from detectron2.data.transforms import Augmentation, Transform, RandomRotation, NoOpTransform, CropTransform, ResizeTransform, \
    ResizeShortestEdge, RandomContrast, RandomBrightness, RandomFlip, RandomSaturation, RandomApply
from canopyrs.engine.config_parsers import DetectorConfig


class RandomChoiceAugmentation(Augmentation):
    """
    Randomly selects one of two augmentations based on a given probability.
    With probability `prob`, applies aug1; otherwise, applies aug2.
    """
    def __init__(self, aug1: Augmentation, aug2: Augmentation, prob: float):
        super().__init__()
        self.aug1 = aug1
        self.aug2 = aug2
        self.prob = prob

    def get_transform(self, image):
        if random() < self.prob:
            return self.aug1.get_transform(image)
        else:
            return self.aug2.get_transform(image)


class RandomChoiceAugmentationWithBox(Augmentation):
    """
    Randomly selects one of two augmentations based on a given probability.
    With probability `prob`, applies aug1; otherwise, applies aug2.
    """
    def __init__(self, aug1: Augmentation, aug2: Augmentation, prob: float):
        super().__init__()
        self.aug1 = aug1
        self.aug2 = aug2
        self.prob = prob

    def get_transform(self, image, boxes=None):
        if random() < self.prob:
            return self.aug1.get_transform(image, boxes)
        else:
            return self.aug2.get_transform(image, boxes)


class RandomHueTransform(Transform):
    def __init__(self, hue_delta: int):
        """
        Args:
            hue_delta (int): Amount to shift the hue channel. This value will be added
                             to the hue channel (in OpenCV's 0-179 scale) and wrapped around.
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

    def apply_segmentation(self, segmentation):
        # Don't apply color augmentation to segmentation masks
        return segmentation

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
        hue_delta = randint(self.hue_delta_range[0], self.hue_delta_range[1])
        return RandomHueTransform(hue_delta)


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


class SquareRandomCrop(Augmentation):
    """
    Randomly crop a square region from an image. The side length of the square is sampled
    from a given range (crop_range). The crop is chosen only once (no iterative search).

    Args:
        crop_range (tuple): (min_side, max_side) specifying the range (in pixels) of the square side.
    """
    def __init__(self, crop_range):
        super().__init__()
        self.crop_range = crop_range

    def get_transform(self, image: np.ndarray, boxes: np.ndarray = None):
        """
        Args:
            image (np.ndarray): the image as a HxWxC array.
            boxes (np.ndarray): an array of bounding boxes in [x1, y1, x2, y2] format.
                                If None, only the image will be cropped.
        Returns:
            A CropTransform object. If boxes are provided, the transform's attribute
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

        return crop_transform


class ConditionalResize(Augmentation):
    """
    Upscales an image if its smallest side is below a given minimum.
    If the image's smallest side is already >= min_size, it does nothing.
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
            return ResizeTransform(orig_h, orig_w, new_h, new_w)


class DeterministicResizeWithinRange(Augmentation):
    """
    Deterministically resizes the image so that its smallest edge is at least min_size
    and its largest edge is at most max_size. If the image already meets these criteria,
    it is left unchanged.
    """
    def __init__(self, min_size, max_size):
        """
        Args:
            min_size (int): Desired minimum size for the shortest edge.
            max_size (int): Desired maximum size for the longest edge.
        """
        self.min_size = min_size
        self.max_size = max_size

    def get_transform(self, image):
        h, w = image.shape[:2]
        short_edge = min(h, w)
        long_edge = max(h, w)

        # Default to no scaling.
        scale = 1.0

        # If the image is too small.
        if short_edge < self.min_size:
            scale_up = self.min_size / short_edge
            # Check if upscaling causes the long edge to exceed max_size.
            if long_edge * scale_up <= self.max_size:
                scale = scale_up
            else:
                # Upscaling would exceed max_size; instead, downscale to max_size.
                scale = self.max_size / long_edge

        # Else if the image is too large.
        elif long_edge > self.max_size:
            scale = self.max_size / long_edge

        # If no scaling is necessary, return a no-op.
        if scale == 1.0:
            return NoOpTransform()

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return ResizeTransform(h, w, new_h, new_w)
    

class RandomBoxDropTransform(Transform):
    def __init__(self, drop_prob: float):
        super().__init__()
        assert 0.0 <= drop_prob < 1.0
        self.drop_prob = drop_prob

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        # box: np.ndarray of shape (4,) or (1,4)
        if random() < self.drop_prob:
            return np.zeros_like(box)
        return box


class RandomBoxDropAugmentation(Augmentation):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def get_transform(self, image: np.ndarray):
        return RandomBoxDropTransform(self.drop_prob)


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
        cfg.AUGMENTATION.EARLY_CONDITIONAL_IMAGE_SIZE = config.augmentation_early_conditional_image_size
        cfg.AUGMENTATION_EARLY_IMAGE_RESIZE_TEST_ONLY = config.augmentation_early_image_resize_test_only
        cfg.AUGMENTATION.FLIP_HORIZONTAL = config.augmentation_flip_horizontal
        cfg.AUGMENTATION.FLIP_VERTICAL = config.augmentation_flip_vertical
        cfg.AUGMENTATION.ROTATION = config.augmentation_rotation
        cfg.AUGMENTATION.ROTATION_PROB = config.augmentation_rotation_prob
        cfg.AUGMENTATION.BRIGHTNESS = config.augmentation_brightness
        cfg.AUGMENTATION.BRIGHTNESS_PROB = config.augmentation_brightness_prob
        cfg.AUGMENTATION.CONTRAST = config.augmentation_contrast
        cfg.AUGMENTATION.CONTRAST_PROB = config.augmentation_contrast_prob
        cfg.AUGMENTATION.SATURATION = config.augmentation_saturation
        cfg.AUGMENTATION.SATURATION_PROB = config.augmentation_saturation_prob
        cfg.AUGMENTATION.HUE = config.augmentation_hue
        cfg.AUGMENTATION.HUE_PROB = config.augmentation_hue_prob
        cfg.AUGMENTATION.CROP_SIZE_RANGE = config.augmentation_train_crop_size_range
        cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO = config.augmentation_crop_min_intersection_ratio
        cfg.AUGMENTATION.CROP_PROB = config.augmentation_crop_prob
        cfg.AUGMENTATION.CROP_FALLBACK_TO_AUGMENTATION_IMAGE_SIZE = config.augmentation_crop_fallback_to_augmentation_image_size
        cfg.AUGMENTATION_DROP_ANNOTATION_RANDOM_PROB = config.augmentation_drop_annotation_random_prob

    @staticmethod
    def get_augmentation_detectron2_train(cfg):
        """
        Build a list of train_detectron2 augmentations (or custom augmentations)
        based on the parameters in cfg.AUGMENTATION.
        """
        augs = AugmentationAdder._get_augmentation_list_train(
            final_image_size=cfg.AUGMENTATION.IMAGE_SIZE,
            early_conditional_image_size=cfg.AUGMENTATION.EARLY_CONDITIONAL_IMAGE_SIZE,
            flip_vertical=cfg.AUGMENTATION.FLIP_VERTICAL,
            flip_horizontal=cfg.AUGMENTATION.FLIP_HORIZONTAL,
            rotation=cfg.AUGMENTATION.ROTATION,
            rotation_prob=cfg.AUGMENTATION.ROTATION_PROB,
            brightness=cfg.AUGMENTATION.BRIGHTNESS,
            brightness_prob=cfg.AUGMENTATION.BRIGHTNESS_PROB,
            contrast=cfg.AUGMENTATION.CONTRAST,
            contrast_prob=cfg.AUGMENTATION.CONTRAST_PROB,
            saturation=cfg.AUGMENTATION.SATURATION,
            saturation_prob=cfg.AUGMENTATION.SATURATION_PROB,
            hue=cfg.AUGMENTATION.HUE,
            hue_prob=cfg.AUGMENTATION.HUE_PROB,
            crop_size_range=cfg.AUGMENTATION.CROP_SIZE_RANGE,
            crop_min_intersection_ratio=cfg.AUGMENTATION.CROP_MIN_INTERSECTION_RATIO,
            crop_prob=cfg.AUGMENTATION.CROP_PROB,
            crop_fallback_to_augmentation_image_size=cfg.AUGMENTATION.CROP_FALLBACK_TO_AUGMENTATION_IMAGE_SIZE,
            box_drop_prob=cfg.AUGMENTATION_DROP_ANNOTATION_RANDOM_PROB
        )

        return augs

    @staticmethod
    def get_augmentation_detectron2_test(cfg):
        """
        Build the test-time augmentation. Typically just a fixed resize.
        """
        augs = AugmentationAdder._get_augmentation_list_test(
            image_size=cfg.AUGMENTATION.IMAGE_SIZE,
            augmentation_early_image_resize_test_only=cfg.AUGMENTATION_EARLY_IMAGE_RESIZE_TEST_ONLY
        )

        return augs

    @staticmethod
    def get_augmentation_detrex_train(config: DetectorConfig):
        augs = AugmentationAdder._get_augmentation_list_train(
            final_image_size=config.augmentation_image_size,
            early_conditional_image_size=config.augmentation_early_conditional_image_size,
            flip_vertical=config.augmentation_flip_vertical,
            flip_horizontal=config.augmentation_flip_horizontal,
            rotation=config.augmentation_rotation,
            rotation_prob=config.augmentation_rotation_prob,
            brightness=config.augmentation_brightness,
            brightness_prob=config.augmentation_brightness_prob,
            contrast=config.augmentation_contrast,
            contrast_prob=config.augmentation_contrast_prob,
            saturation=config.augmentation_saturation,
            saturation_prob=config.augmentation_saturation_prob,
            hue=config.augmentation_hue,
            hue_prob=config.augmentation_hue_prob,
            crop_size_range=config.augmentation_train_crop_size_range,
            crop_min_intersection_ratio=config.augmentation_crop_min_intersection_ratio,
            crop_prob=config.augmentation_crop_prob,
            crop_fallback_to_augmentation_image_size=config.augmentation_crop_fallback_to_augmentation_image_size,
            box_drop_prob=config.augmentation_drop_annotation_random_prob
        )

        return augs

    @staticmethod
    def get_augmentation_detrex_test(config: DetectorConfig):
        # Resize to a final, fixed size
        augs = AugmentationAdder._get_augmentation_list_test(
            image_size=config.augmentation_image_size,
            augmentation_early_image_resize_test_only=config.augmentation_early_image_resize_test_only
        )

        return augs

    @staticmethod
    def _get_augmentation_list_train(
            final_image_size: int or tuple,
            early_conditional_image_size: int,
            flip_vertical: bool,
            flip_horizontal: bool,
            rotation: float,
            rotation_prob: float,
            brightness: float,
            brightness_prob: float,
            contrast: float,
            contrast_prob: float,
            saturation: float,
            saturation_prob: float,
            hue: float,
            hue_prob: float,
            crop_size_range: tuple or list,
            crop_min_intersection_ratio: float,
            crop_prob: float,
            crop_fallback_to_augmentation_image_size: bool,
            box_drop_prob: float
    ):
        """
        Build a list of Detectron2 augmentations based on the given parameters.

        Args:
            final_image_size (int): for final ResizeShortestEdge.
            early_conditional_image_size (int): for ConditionalResize at beginning in case image is too small. Can be None or False to deactivate.
            flip_vertical (bool): if True, applies RandomFlip( vertical=True ).
            flip_horizontal (bool): if True, applies RandomFlip( horizontal=True ).
            rotation (float): if non-zero, range of angles (e.g., [-rotation, rotation]) for random rotation.
            rotation_prob (float): probability of applying the random rotation.
            brightness (float): amount for RandomBrightness (e.g., 0.2 means ±20%).
            brightness_prob (float): probability of applying brightness augmentation.
            contrast (float): amount for RandomContrast (e.g., 0.2 means ±20%).
            contrast_prob (float): probability of applying contrast augmentation.
            saturation (float): amount for RandomSaturation (e.g., 0.2 means ±20%).
            saturation_prob (float): probability of applying saturation augmentation.
            hue (int): amount for RandomHue (e.g., ±hue, in [0, 255]).
            hue_prob (float): probability of applying hue augmentation.
            crop_size_range (tuple or list): (min_side, max_side) range for SquareRandomCropWithBoxDiscard.
            crop_min_intersection_ratio (float): ratio threshold to keep boxes that partially fall outside the crop.
            crop_prob (float): probability of applying the random crop.
            crop_fallback_to_augmentation_image_size (bool): if True, fallback to a centered crop of final_image_size.
            box_drop_prob (float): probability to drop box annotations during training. Should usually be 0 except for specific experiments.

        Returns:
            list[Augmentation]: a list of Detectron2-compatible augmentation objects.
        """
        augs = []

        # 1) Conditional resize
        if early_conditional_image_size:
            augs.append(
                ConditionalResize(min_size=early_conditional_image_size)
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
            angle_range = [-rotation, rotation]
            rotation_aug = RandomRotation(angle=angle_range, sample_style="range")
            augs.append(RandomApply(rotation_aug, prob=rotation_prob))

        # 5) Random brightness
        if brightness:
            brightness_min = 1.0 - brightness
            brightness_max = 1.0 + brightness
            brightness_aug = RandomBrightness(intensity_min=brightness_min, intensity_max=brightness_max)
            augs.append(RandomApply(brightness_aug, prob=brightness_prob))

        # 6) Random contrast
        if contrast:
            contrast_min = 1.0 - contrast
            contrast_max = 1.0 + contrast
            contrast_aug = RandomContrast(intensity_min=contrast_min, intensity_max=contrast_max)
            augs.append(RandomApply(contrast_aug, prob=contrast_prob))

        # 7) Random saturation
        if saturation:
            saturation_min = 1.0 - saturation
            saturation_max = 1.0 + saturation
            saturation_aug = RandomSaturation(intensity_min=saturation_min, intensity_max=saturation_max)
            augs.append(RandomApply(saturation_aug, prob=saturation_prob))

        # 8) Random hue
        if hue:
            hue_aug = RandomHueAugmentation(hue_delta_range=(-hue, hue))
            augs.append(RandomApply(hue_aug, prob=hue_prob))

        # 9) Random crop (custom) with probability - SquareRandomCropWithBoxDiscard
        crop_aug = SquareRandomCrop(
            crop_range=crop_size_range,
        )
        if crop_fallback_to_augmentation_image_size:
            assert isinstance(final_image_size, int),\
                (f"final_image_size should be a single int if crop_fallback_to_augmentation_image_size is enabled."
                 f" Got {final_image_size}.")
            # Use our custom conditional random apply that falls back to a crop of final_image_size.
            # This is useful if for exemple we have 2048x2048 images for training,
            # but we only want to train on 1024x1024 and don't want to always apply cropping, which can distort
            # the image with Bilinear transformations etc.
            fallback_crop_aug = SquareRandomCrop(
                crop_range=(final_image_size, final_image_size),
            )
            augs.append(RandomChoiceAugmentationWithBox(crop_aug, fallback_crop_aug, prob=crop_prob))
        else:
            # Otherwise, regularly crop with crop_prob.
            augs.append(RandomApply(crop_aug, prob=crop_prob))

        if box_drop_prob > 0:
            augs.append(RandomBoxDropAugmentation(drop_prob=box_drop_prob))

        # 11) Resize to final size or apply range-based resizing
        if isinstance(final_image_size, int):
            augs.append(
                ResizeShortestEdge(
                    short_edge_length=[final_image_size],
                    max_size=final_image_size,
                    sample_style="choice"
                )
            )
        else:
            assert isinstance(final_image_size, (tuple, list))
            assert len(final_image_size) == 2, "final_image_size should be a single int or a tuple of two ints."

            # Random resize with crop_prob probability, else deterministic resize within range.
            resize_shortest_edge = ResizeShortestEdge(
                short_edge_length=final_image_size,
                max_size=final_image_size[1],
                sample_style="range"
            )
            deterministic_resize = DeterministicResizeWithinRange(
                min_size=final_image_size[0],
                max_size=final_image_size[1]
            )
            augs.append(RandomChoiceAugmentation(resize_shortest_edge, deterministic_resize, prob=crop_prob))

        return augs

    @staticmethod
    def _get_augmentation_list_test(image_size: int or tuple[int],
                                    augmentation_early_image_resize_test_only: int or None):
        """
        Build a list of Detectron2 augmentations for test-time based on the given parameters.

        Args:
            image_size (int): final image size for resizing.

        Returns:
            list[Augmentation]: a list of Detectron2-compatible augmentation objects.
        """

        augs = []

        if augmentation_early_image_resize_test_only:       # only to simulate other image resolutions at test time (for evaluation purposes)
            augs.append(
                ResizeShortestEdge(
                    short_edge_length=[augmentation_early_image_resize_test_only],
                    max_size=augmentation_early_image_resize_test_only,
                    sample_style="choice"
                )
            )

        if isinstance(image_size, int):
            augs.append(
                ResizeShortestEdge(
                    short_edge_length=[image_size],
                    max_size=image_size,
                    sample_style="choice"
                )
            )
        else:
            assert len(image_size) == 2, "image_size should be a single int or a tuple of two ints."
            augs.append(
                DeterministicResizeWithinRange(
                    min_size=image_size[0],
                    max_size=image_size[1]
                )
            )

        return augs
