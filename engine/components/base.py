from abc import ABC, abstractmethod
from pathlib import Path

from engine.config_parsers import BaseConfig
from engine.data_state import DataState


class BaseComponent(ABC):
    name: str

    def __init__(self, config: BaseConfig, parent_output_path: str, component_id: int):
        self.config = config
        self.parent_output_path = parent_output_path
        self.output_path = Path(parent_output_path) / f'{component_id}_{self.name}'
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self, data_state: DataState) -> DataState:
        pass

    @abstractmethod
    def update_data_state(self, **kwargs):
        pass
