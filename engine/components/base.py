from abc import ABC, abstractmethod
from pathlib import Path

from engine.config_parsers import BaseConfig
from engine.data_state import DataState
from engine.utils import green_print


class BaseComponent(ABC):
    name: str

    def __init__(self, config: BaseConfig, parent_output_path: str, component_id: int):
        self.config = config
        self.parent_output_path = parent_output_path
        self.output_path = Path(parent_output_path) / f'{component_id}_{self.name}'
        self.output_path.mkdir(parents=True, exist_ok=True)

        green_print(f"Running component '{self.name}'")

    def run(self, data_state: DataState) -> DataState:
        return self(data_state)

    @abstractmethod
    def __call__(self, data_state: DataState) -> DataState:
        pass

    @abstractmethod
    def update_data_state(self, **kwargs):
        pass
