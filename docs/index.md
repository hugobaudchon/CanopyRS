# CanopyRS

A pipeline for processing high-resolution geospatial orthomosaics to detect, segment, and classify trees across various forest biomes.

![CanopyRS banner](assets/canopyrs_banner2.png)

## What is CanopyRS?

CanopyRS takes high-resolution aerial imagery and runs it through a modular component pipeline to produce per-tree detections, segmentations, and classifications. It supports state-of-the-art model architectures spanning both CNNs (Faster R-CNN, Mask R-CNN, RetinaNet) and transformers (DINO, Mask2Former, SAM 2, SAM 3). The pipeline is configurable via YAML, and ships with pre-trained models and preset configurations for common use cases.

## How it works

A CanopyRS pipeline is a sequence of **components**, each responsible for one step:

1. **Tilerizer** — splits a large orthomosaic into overlapping tiles
2. **Detector** — runs object detection on each tile
3. **Segmenter** — refines detections into instance segmentation masks
4. **Aggregator** — merges overlapping detections across tiles using NMS
5. **Classifier** — classifies each detected tree

The pipeline handles all I/O, state management, and background tasks. Components only implement their core logic.

## Quick links

### Getting started

- [Installation](getting-started/installation.md) — get CanopyRS running
- [Quickstart](getting-started/quickstart.md) — run inference in minutes

### User guide

- [Components](user-guide/components.md) — understand each pipeline stage
- [Configuration](user-guide/configuration.md) — configure pipelines via YAML
- [Presets](user-guide/presets.md) — pre-built configurations for common scenarios
- [Data](user-guide/data.md) — download datasets for training and benchmarking
- [Evaluation](user-guide/evaluation.md) — NMS parameter search and benchmarking
- [Training](user-guide/training.md) — train your own detector models

### API reference

- [Pipeline](api/pipeline.md) — pipeline orchestration
- [Components](api/components/base.md) — component classes
- [DataState](api/data-state.md) — state management
