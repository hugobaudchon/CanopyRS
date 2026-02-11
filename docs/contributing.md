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

## Adding a new component

1. Create a config parser in `canopyrs/engine/config_parsers/mycomponent.py` (subclass `BaseConfig`), then add it to `canopyrs/engine/config_parsers/__init__.py` so it can be imported from the package
2. Create `canopyrs/engine/components/mycomponent.py`
3. Subclass `BaseComponent`, declare `requires_state`, `requires_columns`, `produces_state`, `produces_columns`
4. Implement `__call__` with the `@validate_requirements` decorator
5. Register it in `Pipeline.from_config()` in `pipeline.py`
6. Add a docs entry in `docs/user-guide/components.md` and a new page under `docs/api/components/` (then register it in `mkdocs.yml` nav)

## Code style

- Follow existing patterns in the component files
- Keep component logic focused — I/O and state updates are the pipeline's job
- Add hints to `state_hints` and `column_hints` for helpful error messages
