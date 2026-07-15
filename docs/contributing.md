# Contributing

Contributions are welcome. Here is how to get started.

## Development setup

Follow the [installation guide](getting-started/installation.md), then install docs dependencies:

```bash
pip install -e ".[docs]"
```

## Running tests

Run all fast (unit) tests:

```bash
pytest tests/ -m "not slow"
```

Run everything including slow integration tests (requires the test raster under `assets/` and model weights):

```bash
pytest tests/
```

Run a specific test file:

```bash
pytest tests/engine/test_pipeline.py
```

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Working on the engine

Before touching `canopyrs/engine/`, read [`canopyrs/engine/README.md`](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/engine/README.md) — it explains the whole data model concisely (the two tables, contracts, input matching, pixel loading, ancestry...).

## Adding a new component

1. Create a config parser in `canopyrs/engine/config_parsers/mycomponent.py` (subclass `BaseConfig`), then add it to `canopyrs/engine/config_parsers/__init__.py` so it can be imported from the package
2. Create `canopyrs/engine/components/mycomponent.py`
3. Subclass `Component` and decorate it with `@register_component("mykind")` (the kind used in the pipeline YAML)
4. In `__init__`, declare `requires` and `produces` as data contracts — a type, a `Need(type, columns=..., links=..., crs=...)`, or a `one_of(...)` over alternatives
5. Implement `run(self, *inputs)` (the pipeline passes the required tables in order) and return the produced `Imagery`/`Objects` table(s)
6. Import the module in `canopyrs/engine/components/__init__.py` so its `@register_component` runs
7. Add a docs entry in `docs/user-guide/components.md` and a new page under `docs/api/components/` (then register it in `mkdocs.yml` nav)

## Code style

- Follow existing patterns in the component files
- Keep component logic focused — persistence and file I/O are the pipeline's job
- Declare accurate `requires`/`produces` contracts; the pipeline validates against them and reports clear errors when the wiring is wrong
