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
import cv2
import napari


# from sam2.sam2.benchmark import out_frame_idx

# Global variables for drawing
drawing = False
erasing = False
ix, iy = -1, -1
brush_size = 10

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
               linewidth=0.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=0.25)


def gis_bbox_to_pixel(bbox, src):
    max_row, min_col = src.index(bbox[0], bbox[1])
    min_row, max_col = src.index(bbox[2], bbox[3])
    return min_col, min_row, max_col, max_row


def print_memory_usage():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")




def draw_mask(event, x, y, flags, param):
    global drawing, erasing, ix, iy, mask_display, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask_display, (x, y), brush_size, 255, -1)
        elif erasing:
            cv2.circle(mask_display, (x, y), brush_size, 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False

def replace_mask(video_segments, output_folder, input_folder, frame_name):
    global mask_display, brush_size

    # Load the frame image
    image = np.array(Image.open(input_folder + "/" + frame_name).convert("RGB"))

    # Load the old mask
    old_mask = video_segments[3][0].copy().astype(np.uint8)

    # Prepare the mask for editing
    mask_display = old_mask.copy()

    # Set up window and callback
    window_name = "Adjust Mask (Draw=LMB | Erase=RMB | +/-=Brush Size | ESC=Done)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_mask)

    while True:
        # Create overlay image
        overlay = image.copy()
        overlay[mask_display > 0] = [0, 255, 0]  # Green mask overlay

        # Display brush size
        cv2.putText(overlay, f'Brush: {brush_size}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, overlay)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            break
        elif k == ord('+') or k == ord('='):
            brush_size = min(100, brush_size + 2)
        elif k == ord('-') or k == ord('_'):
            brush_size = max(1, brush_size - 2)

    cv2.destroyAllWindows()

    # Replace in video_segments
    video_segments[3] = np.expand_dims(mask_display, 0)
    print("Mask updated:", video_segments[3].shape)

    return video_segments


def adjust_masks(frame_names, inference_state, input_folder, output_folder, predictor, strategy, start_frame=0):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    reverse=(strategy == "reverse"),
                                                                                    start_frame_idx=start_frame,
                                                                                    max_frame_num_to_track=None):

        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_folder, frame_names[out_frame_idx])))

        video_segments = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                          for i, out_obj_id in enumerate(out_obj_ids)
                          }

        for out_obj_id, out_mask in video_segments.items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(os.path.join(output_folder, f"frame_{out_frame_idx:04d}-{strategy}.png"), bbox_inches="tight",
                    dpi=300)
        plt.close("all")

        if out_frame_idx == 2:
            video_segments = replace_mask(video_segments, output_folder, input_folder, frame_names[out_frame_idx])

            predictor.reset_state(inference_state)
            # Inject the extracted masks into inference state
            for out_obj_id, out_mask in video_segments.items():
                predictor.add_new_mask(inference_state, out_frame_idx, out_obj_id, out_mask.squeeze())
            # Mark the frame as processed
            inference_state['frames_already_tracked'].update({str(out_frame_idx): {'reverse': False}})

            plt.figure(figsize=(9, 6))
            plt.title(f"frame {out_frame_idx}_adjusted")
            plt.imshow(Image.open(os.path.join(input_folder, frame_names[out_frame_idx])))

            for out_obj_id, out_mask in video_segments.items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.savefig(os.path.join(output_folder, f"frame_{out_frame_idx:04d}-{strategy}_adjusted.png"), bbox_inches="tight",
                        dpi=300)
            plt.close("all")
            adjust_masks(frame_names, inference_state, input_folder, output_folder, predictor, strategy,
                         start_frame=out_frame_idx+1)
            # Break after handling one frame to restart propagation
            break

def timeseries_sam2(input_folder: str, output_folder: str, bbox_file: str, ann_frame_idx: int, max_bboxes: int):
    strategy = "app"
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

    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=input_folder)
    predictor.reset_state(inference_state)
    bboxes_gdf = gp.read_file(bbox_file)
    bboxes = bboxes_gdf["geometry"].apply(lambda geom: geom.bounds)  # (minx, miny, maxx, maxy)
    bboxes = bboxes[:max_bboxes]  # for testing, only take x bboxes for now
    bboxes.reset_index()

    with rasterio.open(input_folder + '/' + frame_names[0]) as src:
        bboxes = bboxes.apply(gis_bbox_to_pixel, args=(src,))

    # show the results on the current (interacted) frame on all objects
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(input_folder, frame_names[ann_frame_idx])))

    for bbox_id, bbox in bboxes.items():
        print_memory_usage()
        if bbox_id != 3:
            continue
        else:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=bbox_id,
                box=bbox
            )
            show_box(bbox, ax=plt.gca())

    plt.savefig(os.path.join(output_folder, f"frame_{ann_frame_idx:04d}_bboxes.png"), bbox_inches="tight", dpi=300)
    video_segments = {}

    adjust_masks(frame_names, inference_state, input_folder, output_folder, predictor, strategy)


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
    )
    plt.close("all")
