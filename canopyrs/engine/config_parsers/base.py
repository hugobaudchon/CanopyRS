from pathlib import Path
from typing import Dict, Any

import yaml
from pydantic import BaseModel

from canopyrs.config import predefined_configs, config_folder_path


def get_config_path(config_name: str):
    if config_name in predefined_configs:
        config_path = Path(config_folder_path) / config_name / 'pipeline.yaml'
    elif config_name.split('/')[0] in predefined_configs:
        config_path = Path(config_folder_path) / f'{config_name}.yaml'
    elif Path(config_name).is_file() and config_name.endswith('.yaml'):
        config_path = Path(config_name)
    else:
        raise ValueError("Invalid config path.")
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
