# Pipeline

The `Pipeline` class is the orchestrator. It sequences components, manages state, handles file I/O, and coordinates background tasks.

## Lifecycle

```
Pipeline.__init__()
    ├── Assign component IDs and output paths
    ├── Print flow chart (visualizes state/column flow)
    └── Validate pipeline (catch missing requirements early)

Pipeline.run()
    └── For each component:
        ├── Wait for any required background tasks (e.g. COCO generation)
        ├── Run component.__call__(data_state)
        └── Process result (merge GDF, save files, queue async tasks)
```

## Flow chart

When a pipeline is constructed, it automatically prints a colored flow chart showing which state keys and GDF columns are available, produced, required, or missing at each stage. This makes it straightforward to spot configuration errors before any inference runs.

## State management

All shared state lives in a single `DataState` object. Components read from it and return a `ComponentResult`. The pipeline applies the result to the state — components never mutate `DataState` directly.

Key state keys:

| Key | Description |
|---|---|
| `imagery_path` | Path to the input orthomosaic |
| `tiles_path` | Path to the generated tiles directory |
| `infer_gdf` | The working GeoDataFrame of detections |
| `infer_coco_path` | Path to the current COCO annotation file |
| `product_name` | Derived from the input filename, used for output naming |

## Background tasks

COCO file generation is expensive and is queued as a background process after each component that produces one. The pipeline automatically waits for the relevant COCO file to finish before running any component that requires it — no manual coordination needed.

## Standalone usage

A single component can be run outside a pipeline using the `run_component()` helper:

```python
from canopyrs.engine.components.detector import DetectorComponent
from canopyrs.engine.config_parsers import DetectorConfig

config = DetectorConfig(model='dino_detrex', ...)
detector = DetectorComponent(config)

result = run_component(
    detector,
    output_path='./output',
    tiles_path='./tiles'
)
```
