# CanopyRS

A pipeline for processing high-resolution geospatial orthomosaics to detect, segment, and classify trees across various forest biomes.

![CanopyRS banner](assets/canopyrs_banner2.png)

## What is CanopyRS?

CanopyRS takes high-resolution aerial or satellite imagery and runs it through a modular component pipeline to produce per-tree detections, segmentations, and classifications. The pipeline is configurable via YAML, and ships with pre-trained models and preset configurations for common use cases.

## How it works

A CanopyRS pipeline is a sequence of **components**, each responsible for one step:

1. **Tilerizer** — splits a large orthomosaic into overlapping tiles
2. **Detector** — runs object detection on each tile
3. **Segmenter** — refines detections into instance segmentation masks
4. **Aggregator** — merges overlapping detections across tiles using NMS
5. **Classifier** — classifies each detected tree

The pipeline handles all I/O, state management, and background tasks. Components only implement their core logic.

## Quick links

- [Installation](getting-started/installation.md) — get CanopyRS running
- [Quickstart](getting-started/quickstart.md) — run inference in minutes
- [Components](user-guide/components.md) — understand each pipeline stage
- [Presets](user-guide/presets.md) — pre-built configurations for common scenarios
