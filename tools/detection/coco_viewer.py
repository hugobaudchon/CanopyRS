#!/usr/bin/env python3
"""
COCO Annotation Viewer
A simple GUI tool to view COCO format annotations with confidence threshold filtering.

Usage: python coco_viewer.py <coco_json_path> <images_folder>
"""

import json
import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import argparse


class COCOViewer:
    def __init__(self, coco_path, images_folder):
        self.coco_path = coco_path
        self.images_folder = images_folder
        self.current_index = 0
        self.confidence_threshold = 0.5
        self.zoom_level = 1.0
        self.original_image = None
        self.processed_image = None

        # Load COCO data
        self.load_coco_data()

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("COCO Annotation Viewer")
        # default window size
        self.root.geometry("1000x1000")

        self.setup_gui()
        self.display_current_image()

        # ensure the image fits in initial window
        self.root.update_idletasks()
        self.zoom_fit()

    def load_coco_data(self):
        with open(self.coco_path, 'r') as f:
            self.coco_data = json.load(f)

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data.get('categories', [])}

        self.annotations_by_image = {}
        for ann in self.coco_data.get('annotations', []):
            self.annotations_by_image.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = [img_id for img_id, info in self.images.items()
                          if os.path.exists(os.path.join(self.images_folder, info['file_name']))]

        if not self.image_ids:
            raise ValueError("No matching images found in the specified folder")

        print(f"Loaded {len(self.image_ids)} images with annotations")

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT)
        self.prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=(0,5))
        self.next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT)
        self.counter_label = ttk.Label(nav_frame, text="")
        self.counter_label.pack(side=tk.LEFT, padx=(20,0))
        self.object_count_label = ttk.Label(nav_frame, text="")
        self.object_count_label.pack(side=tk.LEFT, padx=(10,0))

        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(side=tk.LEFT, padx=(20,0))
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=(5,2))
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=(2,2))
        ttk.Button(zoom_frame, text="Fit", command=self.zoom_fit, width=4).pack(side=tk.LEFT, padx=(2,0))

        # Confidence threshold
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(side=tk.RIGHT)
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=(0,5))
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        self.threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_threshold_change,
            length=200,
            showvalue=True
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=(0,5))

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bindings for navigation, panning, and zooming
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.zoom_fit())
        # Mouse bindings
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', lambda e: self.zoom_in())
        self.canvas.bind('<Button-5>', lambda e: self.zoom_out())
        self.root.focus_set()

    def on_threshold_change(self, val):
        self.confidence_threshold = float(val)
        self.display_current_image()

    def zoom_in(self):
        self.zoom_level *= 1.2
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.2
        self.update_display()

    def zoom_fit(self):
        if not self.processed_image:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.processed_image.size
        if cw > 1 and ch > 1:
            self.zoom_level = min(cw/iw, ch/ih, 1.0)
            self.update_display()

    def on_canvas_click(self, event):
        # Start panning
        self.canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event):
        # Drag to pan
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event):
        # Windows/Mac scroll event
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def update_display(self):
        if not self.processed_image:
            return
        nw = int(self.processed_image.width * self.zoom_level)
        nh = int(self.processed_image.height * self.zoom_level)
        if nw > 0 and nh > 0:
            zoomed = self.processed_image.resize((nw, nh), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(zoomed)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
            self.zoom_fit()

    def next_image(self):
        if self.current_index < len(self.image_ids) - 1:
            self.current_index += 1
            self.display_current_image()
            self.zoom_fit()

    def display_current_image(self):
        if not self.image_ids:
            return
        img_id = self.image_ids[self.current_index]
        path = os.path.join(self.images_folder, self.images[img_id]['file_name'])

        self.counter_label.config(text=f"Image: {self.current_index+1}/{len(self.image_ids)}")
        self.prev_btn.config(state=tk.NORMAL if self.current_index>0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_index<len(self.image_ids)-1 else tk.DISABLED)

        try:
            self.original_image = Image.open(path)
            anns = self.annotations_by_image.get(img_id, [])
            self.processed_image = self.draw_annotations(self.original_image, anns)
            self.update_display()
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text=f"Error loading image: {e}", fill='red')

    def draw_annotations(self, image, annotations):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        colors = ['red','blue','green','yellow','purple','orange','cyan','magenta','lime','pink','brown','gray','olive','navy','teal','maroon']

        drawn = 0
        total = len(annotations)
        for i, ann in enumerate(annotations):
            try:
                score = float(ann.get('score', 1.0))
            except:
                continue
            if score < self.confidence_threshold:
                continue

            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            try:
                x, y, w, h = [float(v) for v in bbox]
            except:
                continue
            if w <= 0 or h <= 0:
                continue
            x2, y2 = x+w, y+h
            iw, ih = image.size
            if x2 < 0 or y2 < 0 or x > iw or y > ih:
                continue

            color = colors[i % len(colors)]
            draw.rectangle([x, y, x2, y2], outline=color, width=2)

            cid = ann.get('category_id')
            if cid in self.categories:
                label = f"{self.categories[cid]} ({score:.2f})"
                try:
                    tb = draw.textbbox((x, max(0, y-20)), label)
                    draw.rectangle(tb, fill=color)
                    draw.text((x, max(0, y-20)), label, fill='white')
                except:
                    draw.text((x, max(0, y-20)), label, fill=color)

            seg = ann.get('segmentation')
            if isinstance(seg, list):
                for poly in seg:
                    pts = [(poly[j], poly[j+1]) for j in range(0, len(poly), 2)]
                    draw.polygon(pts, outline=color, width=1)

            drawn += 1

        self.object_count_label.config(text=f"Objects: {drawn}/{total}")
        return img

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='COCO Annotation Viewer')
    parser.add_argument('coco_json', help='Path to COCO JSON file')
    parser.add_argument('images_folder', help='Path to folder containing images')
    args = parser.parse_args()

    if not os.path.exists(args.coco_json) or not os.path.exists(args.images_folder):
        print(f"Error: JSON or images folder not found")
        return 1
    try:
        COCOViewer(args.coco_json, args.images_folder).run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
