import time
from abc import ABC, abstractmethod
from pathlib import Path

from canopyrs.engine.config_parsers import BaseConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import green_print, get_component_folder_name


class BaseComponent(ABC):
    name: str

    def __init__(self, config: BaseConfig, parent_output_path: str, component_id: int):
        self.config = config
        self.parent_output_path = parent_output_path
        self.component_id = component_id
        self.output_path = Path(parent_output_path) / get_component_folder_name(self.component_id, self.name)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # same global temp path for all components
        self.temp_path = Path(self.parent_output_path) / 'temp'

        green_print(f"Running component '{self.name}'", add_return=True)

    def register_outputs_base(self, data_state: DataState) -> DataState:
        """Register this component's outputs with the data_state."""
        data_state.register_component_folder(self.name, self.component_id, self.output_path)
        return data_state

    def run(self, data_state: DataState) -> DataState:
        start_time = time.time()
        data_state = self(data_state)
        green_print(f"Finished in {time.time() - start_time:.1f} seconds")
        return data_state

    @abstractmethod
    def __call__(self, data_state: DataState) -> DataState:
        pass

    @abstractmethod
    def update_data_state(self, **kwargs):
        pass
