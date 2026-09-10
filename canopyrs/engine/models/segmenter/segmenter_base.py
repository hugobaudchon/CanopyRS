from abc import ABC, abstractmethod
from typing import Tuple, List
import multiprocessing
import warnings
import cv2
import numpy as np
import psutil
import torch

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"None of the inputs have requires_grad=True\. Gradients will be None"
)
from shapely import box
from shapely.affinity import scale

from geodataset.utils import mask_to_polygon

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.loader import InferTimer


def get_memory_usage():
    memory_info = psutil.virtual_memory()
    memory_percentage = memory_info.percent

    return memory_percentage


def process_masks(queue,
                  output_dict,
                  output_dict_lock,
                  simplify_tolerance,
                  remove_rings,
                  remove_small_geoms,
                  processed_counter):
    results = {}
    while True:
        item = queue.get()
        if item is None:
            break
        tile_idx, mask_ids, box_object_ids, masks, scores, image_size = item
        masks_polygons = [mask_to_polygon(mask,
                                          simplify_tolerance=simplify_tolerance,
                                          remove_rings=remove_rings,
                                          remove_small_geoms=remove_small_geoms) for mask in masks]

        # Fix invalid polygons
        for id, polygon in enumerate(masks_polygons):
            if not polygon.is_valid:
                # If the polygon is still invalid, set its score to 0 and create a dummy box polygon
                polygon = box(0, 0, 1, 1)
                scores[id] = 0.0
            if polygon.is_empty:
                # If the polygon is empty, set its score to 0 and create a dummy box polygon
                polygon = box(0, 0, 1, 1)
                scores[id] = 0.0
            masks_polygons[id] = polygon

        mask_h, mask_w = masks.shape[-2], masks.shape[-1]  # e.g. 28,28
        orig_h, orig_w = image_size[0], image_size[1]  # e.g. 1024,1024
        if (mask_h != orig_h) or (mask_w != orig_w):
            # Compute scaling factors for x (width) and y (height)
            scale_x = float(orig_w) / float(mask_w)
            scale_y = float(orig_h) / float(mask_h)
            resized_polygons = []
            for poly in masks_polygons:
                # Scale shapely polygon from (0,0)
                poly_scaled = scale(poly, xfact=scale_x, yfact=scale_y, origin=(0, 0))
                resized_polygons.append(poly_scaled)

            masks_polygons = resized_polygons

        # Store the tile/image results
        if tile_idx not in results:
            results[tile_idx] = []
        [results[tile_idx].append((mask_id, box_object_id, mask_poly, score)) for mask_id, box_object_id, mask_poly, score in
         zip(mask_ids, box_object_ids, masks_polygons, scores)]

        queue.task_done()  # Indicate that the task is complete
        with processed_counter.get_lock():
            processed_counter.value += 1

    with output_dict_lock:
        for tile_idx in results:
            if tile_idx not in output_dict:
                output_dict[tile_idx] = results[tile_idx]
            else:
                current_list = output_dict[tile_idx]
                current_list.extend(results[tile_idx])
                output_dict[tile_idx] = current_list


class SegmenterWrapperBase(ABC):
    REQUIRES_BOX_PROMPT = None

    def __init__(self, config: SegmenterConfig):
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._check_init()

    def _check_init(self):
        assert self.REQUIRES_BOX_PROMPT is not None,\
            "Classes built from SegmenterWrapperBase must have REQUIRES_BOX_PROMPT set to True or False"

    @abstractmethod
    def forward(self,
                images: List[np.array],
                boxes: List[np.array],
                boxes_object_ids: List[int or None],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue):
        pass

    def queue_masks(self,
                    box_object_ids: List[int or None],
                    masks: np.array,
                    image_size: Tuple[int, int],
                    scores: np.array,
                    tile_idx: int,
                    n_masks_processed: int,
                    queue: multiprocessing.JoinableQueue):
        
        # Scale down the masks to a fixed size to reduce memory footprint during postprocessing
        if self.config.pp_down_scale_masks_px and masks.shape[-1] > self.config.pp_down_scale_masks_px:
            resized_list = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8)

                mask_resized = cv2.resize(
                    mask,
                    (self.config.pp_down_scale_masks_px, self.config.pp_down_scale_masks_px),
                    interpolation=cv2.INTER_LINEAR
                )
                if masks.dtype == bool:
                    mask_resized = mask_resized > 0.5
                resized_list.append(mask_resized)

            # Stack them back along the batch dimension if you want a single tensor
            masks = np.stack(resized_list, axis=0)

        # Split masks and scores into chunks and put them into the queue for post-processing
        num_masks = masks.shape[0]
        chunk_size = max(1, num_masks // self.config.pp_n_workers)
        for j in range(0, num_masks, chunk_size):
            chunk_masks = masks[j:j + chunk_size]
            chunk_scores = scores[j:j + chunk_size]
            chunk_box_object_ids = box_object_ids[j:j + chunk_size]
            mask_ids = list(range(n_masks_processed, n_masks_processed + len(chunk_masks)))
            queue.put((tile_idx, mask_ids, chunk_box_object_ids, chunk_masks, chunk_scores, image_size))
            n_masks_processed += len(chunk_masks)

        return n_masks_processed

    def infer(self, loader, boxes_by_tile=None):
        """Consume a ``tile_loader``, iterated as ``(object_ids, images)`` batches, and return per-tile
        ``(tile_object_ids, mask_object_ids, mask_polygons, mask_scores)`` — polygons in tile-pixel
        coords. Reuses ``forward`` and the multiprocessing mask->polygon postprocessing; builds no
        DataLoader of its own.

        boxes_by_tile : {tile_object_id: (boxes np[N,4] xyxy pixel, box_object_ids list[N])} for
        box-prompted models; None for automatic mask generation.
        """
        queue = multiprocessing.JoinableQueue()
        manager = multiprocessing.Manager()
        output_dict = manager.dict()
        processed_counter = multiprocessing.Value('i', 0)
        output_dict_lock = multiprocessing.Lock()

        workers = []
        for _ in range(self.config.pp_n_workers):
            p = multiprocessing.Process(target=process_masks,
                                        args=(queue, output_dict, output_dict_lock,
                                              self.config.pp_simplify_tolerance, self.config.pp_remove_rings,
                                              self.config.pp_remove_small_geoms, processed_counter))
            p.start()
            workers.append(p)

        tile_object_ids = []                                   # tile_idx (running) -> tile object_id
        timer = InferTimer("Inferring the segmenter...")
        for object_ids, images in timer.batches(loader):
            images = [img.numpy() if hasattr(img, "numpy") else np.asarray(img) for img in images]
            base = len(tile_object_ids)
            tiles_idx = list(range(base, base + len(object_ids)))
            tile_object_ids.extend(object_ids)
            if boxes_by_tile is None:
                boxes = [None] * len(images)
                boxes_object_ids = [None] * len(images)
            else:
                boxes = [boxes_by_tile[oid][0] for oid in object_ids]
                boxes_object_ids = [boxes_by_tile[oid][1] for oid in object_ids]
            timer.mark("prep")
            # forward moves the images to the device itself, so unlike the other models the transfer
            # falls in ``gpu`` here rather than in ``prep``; and the masks it queues are polygonized
            # concurrently, so ``post`` below is the drain, not the polygonization.
            self.forward(images=images, boxes=boxes, boxes_object_ids=boxes_object_ids,
                         tiles_idx=tiles_idx, queue=queue)
            timer.mark("gpu")

        queue.join()
        for _ in range(self.config.pp_n_workers):
            queue.put(None)
        for p in workers:
            p.join()
        queue.close()
        timer.mark("post")                                     # draining the mask->polygon workers
        timer.report()

        mask_object_ids, mask_polygons, mask_scores = [], [], []
        for tile_idx in range(len(tile_object_ids)):           # aligned to tile_object_ids order
            entries = sorted(output_dict.get(tile_idx, []), key=lambda x: x[0])
            if entries:
                _, box_object_ids, polygons, scores = zip(*entries)
                mask_object_ids.append(list(box_object_ids))
                mask_polygons.append(list(polygons))
                mask_scores.append([s.item() if hasattr(s, "item") else s for s in scores])
            else:
                mask_object_ids.append([])
                mask_polygons.append([])
                mask_scores.append([])

        return tile_object_ids, mask_object_ids, mask_polygons, mask_scores

