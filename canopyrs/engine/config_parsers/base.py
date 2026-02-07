from pathlib import Path
from typing import Dict, Any

import yaml
from pydantic import BaseModel

from canopyrs.config import predefined_configs, config_folder_path


def get_config_path(config_name: str):
    # Old style: config_name is a folder under config/ containing pipeline.yaml
    if config_name in predefined_configs:
        config_path = Path(config_folder_path) / config_name / 'pipeline.yaml'
    # Nested path: first segment is a known config folder (e.g. "detectors/dino_swinL_multi_NQOS.yaml")
    elif config_name.split('/')[0] in predefined_configs:
        if config_name.endswith('.yaml'):
            config_path = Path(config_folder_path) / config_name
        else:
            config_path = Path(config_folder_path) / f'{config_name}.yaml'
    # New style: flat pipeline yaml in config/pipelines/ (e.g. "preset_det_multi_NQOS_dino_swinL.yaml")
    elif (Path(config_folder_path) / 'pipelines' / config_name).is_file():
        config_path = Path(config_folder_path) / 'pipelines' / config_name
    # Absolute/relative file path
    elif Path(config_name).is_file() and config_name.endswith('.yaml'):
        config_path = Path(config_name)
    else:
        raise ValueError(
            f"Invalid config '{config_name}'. Pass a pipeline name from config/pipelines/, "
            f"a predefined config folder name, or an absolute path to a .yaml file."
        )
    return config_path


class BaseConfig(BaseModel):
    class Config:
        validate_assignment = True

    @classmethod
    def from_dict(cls, data_dict):
        """Create a config from a dictionary."""
        return cls(**data_dict)

    @classmethod
    def from_yaml(cls, path: str or Path) -> 'BaseConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str or Path) -> None:
        with open(path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration using a dictionary."""
        for key, value in updates.items():
            if key in self.model_fields:
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid key '{key}' for config class '{self.__class__.__name__}'")
