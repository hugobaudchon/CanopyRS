import gc
import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import psutil
import rasterio
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor


# from sam2.sam2.benchmark import out_frame_idx


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def gis_bbox_to_pixel(bbox, src):
    max_row, min_col = src.index(bbox[0], bbox[1])
    min_row, max_col = src.index(bbox[2], bbox[3])
    return min_col, min_row, max_col, max_row


def print_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def save_all_frames(frame_names, input_folder, output_folder, strategy, video_segments):
    for out_frame_idx in range(0, len(frame_names), 1):
        plt.close("all")
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_folder, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if strategy == "feed_boxes":
                show_mask(out_mask["mask"], plt.gca(), obj_id=out_obj_id)
                if out_mask["bbox"]:
                    show_box(out_mask["bbox"], plt.gca())
            else:
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(os.path.join(output_folder, f"frame_{out_frame_idx:04d}-{strategy}.png"), bbox_inches="tight",
                    dpi=300)


def feed_boxes(ann_frame_idx, frame_names, inference_state, predictor, video_segments):
    for frame_nr in range(len(frame_names)):  # Process one frame at a time
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                        reverse=False):
            if out_frame_idx != frame_nr:
                continue  # Skip if it's not the current frame

            video_segments[out_frame_idx] = {}

            bounding_boxes = {}

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                # Find bounding box of the mask
                y_indices, x_indices = np.where(mask.squeeze())
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bbox = (x_min, y_min, x_max, y_max)
                else:
                    bbox = None  # Handle empty masks

                video_segments[out_frame_idx][out_obj_id] = {
                    "mask": mask,
                    "bbox": bbox
                }
                if bbox:
                    bounding_boxes[out_obj_id] = bbox

            predictor.reset_state(inference_state)

            # Feed bounding boxes back into the predictor
            for out_obj_id, bbox in bounding_boxes.items():
                # predictor.add_new_box(inference_state, frame_nr, out_obj_id, bbox)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=out_obj_id,
                    box=bbox,
                    clear_old_points=True
                )
            # Mark the frame as processed
            inference_state['frames_already_tracked'].update({str(out_frame_idx): {'reverse': False}})

            break  # Process only one frame per iteration


def feed_segments(frame_names, inference_state, predictor, video_segments):
    for frame_nr in range(len(frame_names)):  # Iterate over each frame
        # Propagate only for the current frame
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                        reverse=False):
            if out_frame_idx != frame_nr:
                continue  # Skip frames that are not the current one

            # Store segmentation masks
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            predictor.reset_state(inference_state)
            # Inject the extracted masks into inference state
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                predictor.add_new_mask(inference_state, out_frame_idx, out_obj_id, out_mask.squeeze())
            # Mark the frame as processed
            inference_state['frames_already_tracked'].update({str(out_frame_idx): {'reverse': False}})

            # Break after handling one frame to restart propagation
            break
    return video_segments


def default_SAM2(frame_names, inference_state, input_folder, output_folder, predictor, strategy, video_segments, start_frame=0, restarted=False):
    new_points_added = False

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    reverse=(strategy == "reverse"),
                                                                                    start_frame_idx=start_frame):

        inference_state['asdf'] = 'immer noch nix'
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_folder, frame_names[out_frame_idx])))

        if out_frame_idx == 2 and not restarted:
            new_points_added = True
            break


            inference_state['asdf'] = 'jetzt aber!'
            # new_points_added = True

        # video_segments[out_frame_idx] = {
        #     out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        #     for i, out_obj_id in enumerate(out_obj_ids)
        # }
        video_segments = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                          for i, out_obj_id in enumerate(out_obj_ids)
                          }

        for out_obj_id, out_mask in video_segments.items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(os.path.join(output_folder, f"frame_{out_frame_idx:04d}-{strategy}.png"), bbox_inches="tight",
                    dpi=300)
        gc.collect()
        plt.close("all")

    if new_points_added:
        new_points_added = False
        labels = np.array([0, 1])  # negative prompt for intertwined tree
        points = np.array([[1358, 1126], [1426, 580]])
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=out_frame_idx,
            obj_id=16,
            # box=bboxes[3],  # bbox with double tree
            points=points,
            labels=labels,
            clear_old_points=False
        )
        show_points(points, labels, plt.gca(), marker_size=50)
        default_SAM2(frame_names, inference_state, input_folder, output_folder, predictor, strategy, video_segments, start_frame=out_frame_idx, restarted=True)
        return None
    return video_segments


def timeseries_sam2(input_folder: str, output_folder: str, bbox_file: str, ann_frame_idx: int, max_bboxes: int,
                    strategy: str):
    if strategy not in ["feed_segments", "feed_boxes", "reverse", "default"]:
        raise ValueError(f"Invalid strategy: {strategy}")
    output_folder = output_folder + strategy
    Path(output_folder).mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large",
                                                   device=device,
                                                   non_overlap_masks=True,
                                                   use_high_res_features_in_sam=True,
                                                   )
    frame_names = [
        p for p in os.listdir(input_folder)
        if os.path.splitext(p)[-1] in [".tif"]
    ]
    # frame_names.sort(key=lambda p: int(re.search(r'\d+', os.path.splitext(p)[0]).group()))

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=input_folder)
    predictor.reset_state(inference_state)
    bboxes_gdf = gp.read_file(bbox_file)
    bboxes = bboxes_gdf["geometry"].apply(lambda geom: geom.bounds)  # (minx, miny, maxx, maxy)
    bboxes = bboxes[:max_bboxes]  # for testing, only take x bboxes for now
    # bboxes.pop(10)
    bboxes.reset_index()

    with rasterio.open(input_folder + '/' + frame_names[0]) as src:
        bboxes = bboxes.apply(gis_bbox_to_pixel, args=(src,))

    # show the results on the current (interacted) frame on all objects
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(input_folder, frame_names[ann_frame_idx])))

    for bbox_id, bbox in bboxes.items():
        print_memory_usage()
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=bbox_id,
            box=bbox  # bbox values
        )

        print(bbox_id, bbox)
        show_box(bbox, plt.gca())

    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=3,
    #     box=bboxes[3],  # bbox with double tree
    #     points = np.array([[1358, 1126]]),
    #     labels = np.array([0]) # negative prompt for intertwined tree
    # )

    plt.savefig(os.path.join(output_folder, f"frame_{ann_frame_idx:04d}_bboxes.png"), bbox_inches="tight", dpi=300)
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(input_folder, frame_names[ann_frame_idx])))
    # plt.savefig(os.path.join(output_folder, f"frame_{ann_frame_idx:04d}_bboxes.png"), bbox_inches="tight", dpi=300)
    # plt.show()
    video_segments = {}

    if strategy == "feed_segments":
        video_segments = feed_segments(frame_names, inference_state, predictor, video_segments)
        save_all_frames(frame_names, input_folder, output_folder, strategy, video_segments)

    elif strategy == "feed_boxes":
        video_segments = feed_boxes(ann_frame_idx, frame_names, inference_state, predictor, video_segments)
        save_all_frames(frame_names, input_folder, output_folder, strategy, video_segments)

    elif strategy == "default" or strategy == "reverse":
        # default strategy saves all frames while looping
        inference_state['asdf'] = 'erstmal nichts'
        _ = default_SAM2(frame_names, inference_state, input_folder, output_folder, predictor, strategy,
                                      video_segments)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


if __name__ == "__main__":
    bbox_files = [
        '../../montreal_forest_data/nice_cut/05_28_0_gr0p05_infer.gpkg',
        '../../montreal_forest_data/nice_cut/1007_0_gr0p05_infer.gpkg',
        '/run/media/beerend/LALIB_SSD_2/berend/0046_0_gr0p05_infer.gpkg'
    ]
    ann_frames = [0, 5]
    timeseries_sam2(
        '../../montreal_forest_data/nice_cut/morph/',
        '../../montreal_forest_data/nice_cut/morph_segmented/',
        bbox_files[0],
        ann_frames[0],
        max_bboxes=12,
        strategy="default"
    )
    # timeseries_sam2(
    #     # '../../montreal_forest_data/nice_cut/tiny',
    #     '/run/media/beerend/LALIB_SSD_2/berend/deadtrees1_warped/2022/',
    #     '/run/media/beerend/LALIB_SSD_2/berend/output/deadtrees1/',
    #     bbox_files[2],
    #     ann_frames[0],
    #     max_bboxes=12,
    #     strategy="default"
    # )
    plt.close("all")
