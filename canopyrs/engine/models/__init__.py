import importlib
from pathlib import Path

from canopyrs.engine.models.extras import record_import_failure


def auto_import_models():
    """Automatically import all Python files in model subdirectories."""
    current_dir = Path(__file__).parent

    # Define the subdirectories to search
    model_dirs = ['detector', 'segmenter', 'classifier', 'embedder']

    for model_dir in model_dirs:
        model_path = current_dir / model_dir
        if model_path.exists() and model_path.is_dir():
            # Import all .py files in the directory
            for py_file in model_path.glob('*.py'):
                if py_file.name != '__init__.py':
                    module_name = py_file.stem
                    try:
                        importlib.import_module(f'canopyrs.engine.models.{model_dir}.{module_name}')
                    except ImportError as e:
                        # Missing optional framework: the wrapper simply doesn't register.
                        # Recorded (not printed) so registry misses and `canopyrs doctor` can
                        # tell the user exactly which extra installs it.
                        record_import_failure(f'{model_dir}.{module_name}', e)


# Trigger auto-import when this module is imported
auto_import_models()

# Make registries available at package level
from .registry import (
    DETECTOR_REGISTRY,
    SEGMENTER_REGISTRY,
    CLASSIFIER_REGISTRY,
    EMBEDDER_REGISTRY
)
