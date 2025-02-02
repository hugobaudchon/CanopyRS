import random

import cv2
import numpy as np
import torch
from albumentations import ImageOnlyTransform
from geodataset.dataset import UnlabeledRasterDataset
import multiprocessing
import albumentations as A
from matplotlib import pyplot as plt

from engine.config_parsers import SegmenterConfig
from engine.models.segmenter.segmenter_base import SegmenterWrapperBase

class ConvertRGB2BGR(ImageOnlyTransform):
    def apply(self, img, **params):
        # Albumentations calls apply(...) with (img, rows=..., cols=..., etc.)
        return img[..., ::-1]

def collate_fn(single_image_batch):
    return single_image_batch[0]

class Detectree2TracedWrapper(SegmenterWrapperBase):
    REQUIRES_BOX_PROMPT = False

    infer_transform = A.Compose([
        # A.SmallestMaxSize(max_size=800, interpolation=cv2.INTER_LINEAR),
        # A.LongestMaxSize(max_size=1333, interpolation=cv2.INTER_LINEAR),
        ConvertRGB2BGR(),      # RGB to BGR
    ])
    detectree2_means = [103.53 / 255.0, 116.28 / 255.0, 123.675 / 255.0]
    detectree2_stds = [1.0, 1.0, 1.0]

    def __init__(self, config: SegmenterConfig):
        super().__init__(config)

        # Load SAM model and processor
        print(f"Loading model {self.config.model}")
        self.model = torch.jit.load('/home/hugo/PycharmProjects/detectree2/230103_randresize_full_traced_cuda.pt')    # TODO change this path
        self.model.to(self.device)
        self.model.eval()
        print(f"Model {self.config.model} loaded")

    def infer_image(self,
                    image: np.array,
                    boxes: np.array,
                    tile_idx: int,
                    queue: multiprocessing.JoinableQueue):
        original_image = image[:3, :, :].copy()  # Keep a copy for visualization
        image = image[:3, :, :]
        # image = image.transpose((1, 2, 0))
        image = (image * 255).astype(np.uint8)
        for c in range(3):
            image[c] = (image[c] - self.detectree2_means[c]) / self.detectree2_stds[c]

        with torch.no_grad():
            image = torch.from_numpy(image).float()
            outputs = self.model(image)
            bboxes, class_indices, mask_logits, scores, image_size = outputs
            # print(bboxes.shape, class_indices.shape, mask_logits.shape, scores.shape, image_size)
            bboxes = bboxes.cpu().numpy()
            # print(bboxes)
            class_indices = class_indices.cpu().numpy()
            mask_logits = mask_logits.cpu().numpy().astype(np.uint8)
            scores = scores.cpu().numpy()
            image_size = (image_size[0].item(), image_size[1].item())
            print(image_size)
            # print(mask_logits.shape, mask_logits.dtype)
            # print(scores)
            # Debug: Display the image and one random mask using Matplotlib
            # Debug: Display the image and one random mask using Matplotlib
            # if mask_logits.shape[0] > 0:
            #     selected_mask = mask_logits[0]
            #
            #     # Debug: Inspect the selected mask
            #     print(f"Selected mask shape: {selected_mask.shape}")
            #     print(f"Selected mask unique values: {np.unique(selected_mask)}")
            #
            #     # Resize mask to match the original image size if necessary
            #     mask_resized = cv2.resize(selected_mask, (image_size[1], image_size[0]),
            #                               interpolation=cv2.INTER_NEAREST)
            #     mask_binary = (mask_resized > 0).astype(np.uint8)  # Binary mask
            #
            #     # Debug: Inspect the resized mask
            #     print(f"Resized mask shape: {mask_resized.shape}")
            #     print(f"Resized mask unique values: {np.unique(mask_resized)}")
            #
            #     # Convert original image to HWC and RGB for Matplotlib
            #     display_image = original_image.transpose(1, 2, 0)
            #     display_image = (display_image * 255).astype(np.uint8)
            #
            #     # Debug: Inspect the display image
            #     print(f"Display image shape: {display_image.shape}")
            #     print(f"Display image dtype: {display_image.dtype}")
            #
            #     # Create a colormap for the mask
            #     cmap = plt.cm.jet  # You can choose other colormaps like 'viridis', 'hot', etc.
            #
            #     # Create a figure with two subplots side by side
            #     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            #
            #     # First subplot: Original Image
            #     axes[0].imshow(display_image)
            #     axes[0].set_title("Original Image")
            #     axes[0].axis('off')
            #
            #     # Second subplot: Image with Mask
            #     axes[1].imshow(display_image)
            #     # Overlay the mask with transparency
            #     axes[1].imshow(mask_binary, alpha=0.5, cmap=cmap)
            #     axes[1].set_title(f"Image with Mask")
            #     axes[1].axis('off')
            #
            #     plt.show()
            #
            #     # Optional: Save the figure instead of displaying (useful for multiprocessing)
            #     # fig.savefig(f"debug_figure_{tile_idx}_mask_{random_idx}.png")
            #     # print(f"Saved debug figure with mask {random_idx} as debug_figure_{tile_idx}_mask_{random_idx}.png")
            #     # plt.close(fig)  # Close the figure to free memory
            #
            #     # Optional: Save the image instead of displaying (useful for multiprocessing)
            #     # plt.imsave(f"debug_image_{tile_idx}_mask_{random_idx}.png", display_image)
            #     # print(f"Saved debug image with mask {random_idx} as debug_image_{tile_idx}_mask_{random_idx}.png")
            #
            # else:
            #     print("No masks to display for this image.")

            self.queue_masks(
                mask_logits, image_size, scores, tile_idx, 0, queue,
            )

    def infer_on_dataset(self, dataset: UnlabeledRasterDataset):
        return self._infer_on_dataset(dataset, collate_fn)
