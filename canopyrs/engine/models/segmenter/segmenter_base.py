from abc import ABC, abstractmethod
from typing import Tuple, List
import multiprocessing
import cv2
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from shapely import box
from shapely.affinity import scale
from tqdm import tqdm

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset, BaseDataset
from geodataset.utils import mask_to_polygon

from canopyrs.engine.config_parsers import SegmenterConfig


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

    @abstractmethod
    def infer_on_dataset(self, dataset: BaseDataset):
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

    def _infer_on_dataset(self, dataset: BaseDataset, collate_fn: object):
        infer_dl = DataLoader(dataset, batch_size=self.config.image_batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        tiles_paths = []
        tiles_boxes_object_ids = []
        tiles_masks_polygons = []
        tiles_masks_scores = []
        queue = multiprocessing.JoinableQueue()  # Create a JoinableQueue

        print(f"Setting up {self.config.pp_n_workers} post-processing workers...")
        # Create a manager to share data across processes
        manager = multiprocessing.Manager()
        output_dict = manager.dict()
        processed_counter = multiprocessing.Value('i', 0)
        output_dict_lock = multiprocessing.Lock()

        # Start post-processing processes
        post_process_processes = []
        for _ in range(self.config.pp_n_workers):
            p = multiprocessing.Process(target=process_masks,
                                        args=(queue,
                                              output_dict,
                                              output_dict_lock,
                                              self.config.pp_simplify_tolerance,
                                              self.config.pp_remove_rings,
                                              self.config.pp_remove_small_geoms,
                                              processed_counter))
            p.start()
            post_process_processes.append(p)

        print("Post-processing workers are set up.")

        dataset_with_progress = tqdm(infer_dl,
                                     desc="Inferring the segmenter...",
                                     leave=True)                            # TODO check why its so slow here, like 30 seconds

        for i, sample in enumerate(dataset_with_progress):
            tiles_idx = list(range(i * self.config.image_batch_size, (i + 1) * self.config.image_batch_size))[:len(sample)]
            if isinstance(dataset, DetectionLabeledRasterCocoDataset):
                images, boxes, boxes_object_ids = sample
                tiles_paths.extend([dataset.tiles[tile_idx]['path'] for tile_idx in tiles_idx])     # TODO tiles idx should be returned by the dataset __getitem__ method
            elif isinstance(dataset, UnlabeledRasterDataset):
                images = list(sample)
                boxes = [None] * len(images)
                boxes_object_ids = [None] * len(images)
                tiles_paths.extend([dataset.tile_paths[tile_idx] for tile_idx in tiles_idx])        # TODO tiles idx should be returned by the dataset __getitem__ method
            else:
                raise ValueError("Dataset type not supported.")

            self.forward(
                images=images,
                boxes=boxes,
                boxes_object_ids=boxes_object_ids,
                tiles_idx=tiles_idx,
                queue=queue
            )

        print("Waiting for all postprocessing workers to be finished...")

        # Wait for all tasks in the queue to be completed
        queue.join()

        # Signal the end of input to the queue
        for _ in range(self.config.pp_n_workers):
            queue.put(None)

        # Wait for post-processing processes to finish
        for p in post_process_processes:
            p.join()

        # Close the queue
        queue.close()

        # Sorting the results within each tile_idx by mask_id to maintain order
        for tile_idx in output_dict.keys():
            output_dict[tile_idx] = sorted(output_dict[tile_idx], key=lambda x: x[0])

        # Assemble the results into tiles_masks_polygons
        for tile_idx in sorted(output_dict.keys()):
            _, box_object_ids, masks_polygons, scores = zip(*output_dict[tile_idx])
            box_object_ids = list(box_object_ids)
            masks_polygons = list(masks_polygons)
            scores = [score.item() for score in scores]

            tiles_boxes_object_ids.append(box_object_ids)
            tiles_masks_polygons.append(masks_polygons)
            tiles_masks_scores.append(scores)

        print(f"Finished inferring the segmenter {self.config.model}-{self.config.architecture}.")

        if isinstance(dataset, UnlabeledRasterDataset):
            # There were no box prompts, so we return None for the boxes_object_ids instead of lists of None values
            tiles_boxes_object_ids = None

        return tiles_paths, tiles_boxes_object_ids, tiles_masks_polygons, tiles_masks_scores
