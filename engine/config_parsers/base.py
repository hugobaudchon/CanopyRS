from typing import Dict, Any

import yaml
from pydantic import BaseModel


class BaseConfig(BaseModel):
    class Config:
        validate_assignment = True

    @classmethod
    def from_yaml(cls, path: str) -> 'BaseConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration using a dictionary."""
        for key, value in updates.items():
            if key in self.model_fields:
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid key '{key}' for config class '{self.__class__.__name__}'")