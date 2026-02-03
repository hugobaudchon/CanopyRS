# Contributing

Contributions are welcome. Here is how to get started.

## Development setup

Follow the [installation guide](getting-started/installation.md), then install dev dependencies:

```bash
pip install -e ".[docs,dev]"
```

## Running tests

```bash
pytest tests/
```

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Adding a new component

1. Create `canopyrs/engine/components/mycomponent.py`
2. Subclass `BaseComponent`, declare `requires_state`, `requires_columns`, `produces_state`, `produces_columns`
3. Implement `__call__` with the `@validate_requirements` decorator
4. Register it in `Pipeline.from_config()` in `pipeline.py`
5. Add a docs entry in `docs/user-guide/components.md` and `docs/api/components.md`

## Code style

- Follow existing patterns in the component files
- Keep component logic focused — I/O and state updates are the pipeline's job
- Add hints to `state_hints` and `column_hints` for helpful error messages
