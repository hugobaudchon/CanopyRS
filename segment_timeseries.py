import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gp
import re
from sam2.sam2_video_predictor import SAM2VideoPredictor
import rasterio
import psutil

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


def gis_bbox_to_pixel(bbox, src):
    max_row, min_col = src.index(bbox[0], bbox[1])
    min_row, max_col = src.index(bbox[2], bbox[3])

    return min_col, min_row, max_col, max_row

def timeseries_sam2(input_folder:str, output_folder:str, bbox_file:str, max_bboxes:int):
    ann_frame_idx=0
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

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
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
    bboxes = bboxes[:max_bboxes] # for testing, only take x bboxes for now
    bboxes.reset_index()

    with rasterio.open(input_folder + '/' + frame_names[0]) as src:
        bboxes = bboxes.apply(gis_bbox_to_pixel, args=(src,))
    pid = os.getpid()
    process = psutil.Process(pid)

    def print_memory_usage():
        mem_info = process.memory_info()
        print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

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

        # for i, out_obj_id in enumerate(out_obj_ids):
        #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    # box = np.array([128, 1569, 290, 1739], dtype=np.float32)
    # # box = np.array([132, 1567, 162, 170], dtype=np.float32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=300,
    #     box=box,
    # )
    # box = np.array([1130, 1027, 1440, 1331], dtype=np.float32)
    # # box = np.array([132, 1567, 162, 170], dtype=np.float32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=300,
    #     box=box,
    # )
    plt.savefig(os.path.join(output_folder, f"frame_{ann_frame_idx:04d}_bboxes.png"), bbox_inches="tight", dpi=300)
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(input_folder, frame_names[ann_frame_idx])))
    # plt.savefig(os.path.join(output_folder, f"frame_{ann_frame_idx:04d}_bboxes.png"), bbox_inches="tight", dpi=300)
    # plt.show()

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), 1):
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_folder, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(os.path.join(output_folder, f"frame_{out_frame_idx:04d}.png"), bbox_inches="tight", dpi=300)

if __name__=="__main__":
    timeseries_sam2(
        '../montreal_forest_data/nice_cut/tiny',
        '../montreal_forest_data/nice_cut/output/tiny_warped',
        '../montreal_forest_data/nice_cut/05_28_0_gr0p05_infer.gpkg',
        max_bboxes=12
    )