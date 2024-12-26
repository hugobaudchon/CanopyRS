from abc import ABC, abstractmethod

import numpy as np
import psutil
import torch
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing

from geodataset.utils import mask_to_polygon

from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.config_parsers import SegmenterConfig


def get_memory_usage():
    memory_info = psutil.virtual_memory()
    memory_percentage = memory_info.percent

    return memory_percentage


def process_masks(queue, output_dict, output_dict_lock, simplify_tolerance, processed_counter):
    results = {}
    while True:
        item = queue.get()
        if item is None:
            break
        tile_idx, mask_ids, masks, scores = item
        masks_polygons = [mask_to_polygon(mask.squeeze(), simplify_tolerance=simplify_tolerance) for mask in masks]

        if tile_idx not in results:
            results[tile_idx] = []
        [results[tile_idx].append((mask_id, mask_poly, score)) for mask_id, mask_poly, score in
         zip(mask_ids, masks_polygons, scores)]

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
    def infer_image(self,
                     image: np.array,
                     boxes: np.array,
                     tile_idx: int,
                     queue: multiprocessing.JoinableQueue):
        pass

    @abstractmethod
    def infer_on_multi_box_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        pass

    def queue_masks(self,
                    masks: np.array,
                    scores: np.array,
                    tile_idx: int,
                    n_masks_processed: int,
                    queue: multiprocessing.JoinableQueue):

        # Split masks and scores into chunks and put them into the queue for post-processing
        num_masks = masks.shape[0]
        chunk_size = max(1, num_masks // self.config.n_postprocess_workers)
        for j in range(0, num_masks, chunk_size):
            chunk_masks = masks[j:j + chunk_size]
            chunk_scores = scores[j:j + chunk_size]
            mask_ids = list(range(n_masks_processed, n_masks_processed + len(chunk_masks)))
            queue.put((tile_idx, mask_ids, chunk_masks, chunk_scores))
            n_masks_processed += len(chunk_masks)

        return n_masks_processed

    def _infer_on_multi_box_dataset(self, dataset: DetectionLabeledRasterCocoDataset, collate_fn: object):
        infer_dl = DataLoader(dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        tiles_paths = []
        tiles_masks_polygons = []
        tiles_masks_scores = []
        queue = multiprocessing.JoinableQueue()  # Create a JoinableQueue

        print(f"Setting up {self.config.n_postprocess_workers} post-processing workers...")
        # Create a manager to share data across processes
        manager = multiprocessing.Manager()
        output_dict = manager.dict()
        processed_counter = multiprocessing.Value('i', 0)
        output_dict_lock = multiprocessing.Lock()

        # Start post-processing processes
        post_process_processes = []
        for _ in range(self.config.n_postprocess_workers):
            p = multiprocessing.Process(target=process_masks, args=(queue, output_dict, output_dict_lock, self.config.simplify_tolerance, processed_counter))
            p.start()
            post_process_processes.append(p)

        print("Post-processing workers are set up.")

        dataset_with_progress = tqdm(infer_dl,
                                     desc="Inferring the segmenter...",
                                     leave=True)                            # TODO check why its so slow here, like 30 seconds

        for tile_idx, sample in enumerate(dataset_with_progress):
            image, boxes_data = sample
            image = image[:3, :, :]
            image_hwc = image.transpose((1, 2, 0))
            image_hwc = (image_hwc * 255).astype(np.uint8)
            self.infer_image(
                image=image_hwc,
                boxes=np.array(boxes_data['boxes']),
                tile_idx=tile_idx,
                queue=queue
            )
            tiles_paths.append(dataset.tiles[tile_idx]['path'])

        print("Waiting for all postprocessing workers to be finished...")

        # Wait for all tasks in the queue to be completed
        queue.join()

        # Signal the end of input to the queue
        for _ in range(self.config.n_postprocess_workers):
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
            _, masks_polygons, scores = zip(*output_dict[tile_idx])
            masks_polygons = list(masks_polygons)
            scores = [score.item() for score in scores]
            tiles_masks_polygons.append(masks_polygons)
            tiles_masks_scores.append(scores)

        print("Finished inferring SAM.")

        return tiles_paths, tiles_masks_polygons, tiles_masks_scores
