# Standalone Usage

There's no separate standalone API — running one component is just a one-step pipeline. Seed it with the data that component needs (`sources=` a raster, `tiles=` a folder of pre-cut tiles, or `objects=` a GeoPackage of prior detections) and read the result from `pipe.latest(...)` or an export.

For config parameters (tile size, NMS thresholds, score weights, etc.), see [Configuration](configuration.md).

## Tilerizer

```python
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import TilerizerConfig
from canopyrs.engine.data import Tiles

pipe = Pipeline.from_config(
    [('tilerizer', TilerizerConfig(tile_type='tile', tile_size=512, save_tiles_to_disk=True))],
    sources='raster.tif',
    output_dir='./out',
).run()
print(pipe.latest(Tiles))
```

## Detector

```python
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.data import Objects

config = DetectorConfig.from_yaml('canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml')

pipe = Pipeline.from_config(
    [('detector', config)],
    tiles='./tiles',
    output_dir='./out',
).run()
print(pipe.latest(Objects))
```

## Segmenter

```python
from canopyrs.engine.config_parsers import SegmenterConfig

config = SegmenterConfig.from_yaml('canopyrs/config/segmenters/sam3_multi_selvamask_FT.yaml')

# automatic segmenters seed from tiles; prompted ones (e.g. SAM) seed from prior detections (objects=)
pipe = Pipeline.from_config(
    [('segmenter', config)],
    tiles='./tiles',
    output_dir='./out',
).run()
```

## Aggregator

The aggregator georeferences and de-duplicates existing detections, so seed it from a prior run's tiles and objects:

```python
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import AggregatorConfig
from canopyrs.engine.data import Tiles, Objects

prior = Pipeline.from_dir('./detector_run')
pipe = Pipeline.from_config(
    [('aggregator', AggregatorConfig(nms_algorithm='iou', nms_threshold=0.5, score_threshold=0.3))],
    tiles=prior.latest(Tiles),
    objects=prior.latest(Objects),
    output_dir='./out',
).run()
print(pipe.export('gpkg'))
```

## Classifier

The classifier reads one crop per object, so pair a `polygon` tilerizer with it and seed the objects to classify:

```python
from canopyrs.engine.config_parsers import ClassifierConfig, TilerizerConfig

config = ClassifierConfig.from_yaml(
    'canopyrs/config/classifiers/canopyrs_classifier_dinov3_vit_small_512px_quebec.yaml'
)

pipe = Pipeline.from_config(
    [('tilerizer', TilerizerConfig(tile_type='polygon', tile_size=512)),
     ('classifier', config)],
    sources='raster.tif',
    objects='detections.gpkg',
    output_dir='./out',
).run()
```

## How it works

`from_config` builds the components from the registry and validates the wiring in `__init__`, so a bad standalone setup fails immediately with a clear message listing what's missing. `run()` then executes the single component over the seed you provided.
