# Standalone Usage

Every component can be run individually outside a pipeline via its `run_standalone()` classmethod. Each method has a component-specific signature with explicit parameters, so you get clear parameter names and IDE autocomplete.

For details on available config parameters (tile size, NMS thresholds, score weights, etc.), see [Configuration](configuration.md).

## Tilerizer

```python
from canopyrs.engine.components.tilerizer import TilerizerComponent
from canopyrs.engine.config_parsers import TilerizerConfig

result = TilerizerComponent.run_standalone(
    config=TilerizerConfig(tile_type='tile', tile_size=512, ...),
    imagery_path='./raster.tif',
    output_path='./output',
)
print(result.tiles_path)
```

## Detector

```python
from canopyrs.engine.components.detector import DetectorComponent
from canopyrs.engine.config_parsers import DetectorConfig

result = DetectorComponent.run_standalone(
    config=DetectorConfig(model='dino_detrex', ...),
    tiles_path='./tiles',
    output_path='./output',
)
print(result.infer_gdf)
```

Using a config from a preset (see [Model Zoo](model-zoo.md) for available models):

```python
config = DetectorConfig.from_yaml('canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml')

result = DetectorComponent.run_standalone(
    config=config,
    tiles_path='./tiles',
    output_path='./output',
)
```

## Segmenter

```python
from canopyrs.engine.components.segmenter import SegmenterComponent
from canopyrs.engine.config_parsers import SegmenterConfig

result = SegmenterComponent.run_standalone(
    config=SegmenterConfig(model='sam3', ...),
    tiles_path='./tiles',
    output_path='./output',
    infer_coco_path='./coco.json',  # only if model requires box prompts
)
print(result.infer_gdf)
```

Using a config from a preset (see [Model Zoo](model-zoo.md) for available models):

```python
config = SegmenterConfig.from_yaml('canopyrs/config/segmenters/sam3_multi_selvamask_FT.yaml')

result = SegmenterComponent.run_standalone(
    config=config,
    tiles_path='./tiles',
    output_path='./output',
    infer_coco_path='./coco.json',
)
```

## Aggregator

```python
from canopyrs.engine.components.aggregator import AggregatorComponent
from canopyrs.engine.config_parsers import AggregatorConfig

result = AggregatorComponent.run_standalone(
    config=AggregatorConfig(nms_threshold=0.5, ...),
    infer_gdf=my_detections_gdf,
    output_path='./output',
)
print(result.infer_gdf)
```

## Classifier

```python
from canopyrs.engine.components.classifier import ClassifierComponent
from canopyrs.engine.config_parsers import ClassifierConfig

result = ClassifierComponent.run_standalone(
    config=ClassifierConfig(model='resnet50', ...),
    tiles_path='./polygon_tiles',
    infer_coco_path='./coco.json',
    output_path='./output',
)
print(result.infer_gdf)
```

## How it works

Under the hood, `run_standalone()` delegates to the generic `run_component()` helper, which wraps the component in a single-component pipeline. Inputs are validated before execution — if something is missing, you get a clear error message listing what's needed.

If you need more control, you can use `run_component()` directly:

```python
from canopyrs.engine.pipeline import run_component
from canopyrs.engine.components.detector import DetectorComponent

result = run_component(
    component=DetectorComponent(config),
    output_path='./output',
    tiles_path='./tiles',
)
```
