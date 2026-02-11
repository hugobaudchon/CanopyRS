<h1 align="center">CanopyRS🌴</h1>

<p align="center">
<img src="assets/canopyrs_banner2.png" alt="CanopyRS banner" /><br/>

[//]: # (  <a href="https://github.com/hugobaudchon/CanopyRS/actions">)
[//]: # (    <img src="https://img.shields.io/github/actions/workflow/status/hugobaudchon/CanopyRS/ci.yml?branch=main&label=CI" alt="CI status">)
[//]: # (  </a>)
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python 3.10">
  <img src="https://img.shields.io/badge/CUDA-12.6-green" alt="CUDA 12.6">
  <img src="https://img.shields.io/badge/license-Apache 2.0-green" alt="License: Apache-2.0">
  <a href="https://hugobaudchon.github.io/CanopyRS/">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen" alt="Docs">
  </a>
  <a href="https://arxiv.org/abs/2507.00170">
    <img src="https://img.shields.io/badge/arXiv-2507.00170-red.svg" alt="Read the paper">
  </a>
</p>

Canopy RS (Remote Sensing) is a pipeline designed for processing high-resolution geospatial orthomosaics to detect, segment, and (in the future) classify trees of various forest biomes. It supports state-of-the-art model architectures spanning both CNNs (Faster R-CNN, Mask R-CNN, RetinaNet) and transformers (DINO, Mask2Former, SAM 2, SAM 3).
The pipeline includes components for tiling, detecting, aggregating, and segmenting trees in orthomosaics. These components can be chained together based on the desired application.

**📖 Full documentation: [hugobaudchon.github.io/CanopyRS](https://hugobaudchon.github.io/CanopyRS/)**

## 🎉 News
- **[2024-11-15]**: 🥇 Our team Limelight Rainforest won the $10M XPRIZE Rainforest competition, in part thanks to CanopyRS and SelvaBox!

## 🛠️ Quick Start

See the [Installation guide](https://hugobaudchon.github.io/CanopyRS/getting-started/installation/) and [Quick Start](https://hugobaudchon.github.io/CanopyRS/getting-started/quickstart/) in the documentation.

## 📖 Documentation

The full documentation covers:

- [**Presets & Model Zoo**](https://hugobaudchon.github.io/CanopyRS/user-guide/presets/) — default configs and available pretrained models
- [**Pipeline & Components**](https://hugobaudchon.github.io/CanopyRS/user-guide/pipeline/) — how to configure and run the pipeline
- [**Data**](https://hugobaudchon.github.io/CanopyRS/user-guide/data/) — downloading and using datasets (SelvaBox, SelvaMask, Detectree2, etc.)
- [**Training**](https://hugobaudchon.github.io/CanopyRS/user-guide/training/) — training detectors and segmenters
- [**Evaluation**](https://hugobaudchon.github.io/CanopyRS/user-guide/evaluation/) — benchmarking and finding optimal NMS parameters
- [**API Reference**](https://hugobaudchon.github.io/CanopyRS/api/pipeline/) — programmatic usage

## 📚 Citation
If you use CanopyRS or SelvaBox in your research, please cite our paper (arXiv preprint):

```bibtex
@misc{baudchon2025selvaboxhighresolutiondatasettropical,
      title={SelvaBox: A high-resolution dataset for tropical tree crown detection},
      author={Hugo Baudchon and Arthur Ouaknine and Martin Weiss and Mélisande Teng and Thomas R. Walla and Antoine Caron-Guay and Christopher Pal and Etienne Laliberté},
      year={2025},
      eprint={2507.00170},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.00170},
}
```

If you use other datasets that we have preprocessed, please also cite the original authors of those datasets directly.
