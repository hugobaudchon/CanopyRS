from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd

from canopyrs.engine.utils import get_component_folder_name, object_id_column_name


# Each field declares a "persist" role used by the resume/initialize machinery:
#   - "input":     restored when resuming AND when initializing a new pipeline (the input slice).
#                  infer_gdf belongs here too, but being a GeoDataFrame it is stored/recovered via
#                  a reused gpkg reference (gdf_ref) rather than serialized into state.json.
#   - "registry":  output registries; restored on resume only
#   - "transient": never persisted (runtime objects, output location)
@dataclass
class DataState:
    imagery_path: str = field(default=None, metadata={"persist": "input"})
    parent_output_path: str = field(default=None, metadata={"persist": "transient"})
    product_name: str = field(default=None, metadata={"persist": "input"})  # Derived from imagery filename or "tiled_input" if only tiles

    tiles_path: str = field(default=None, metadata={"persist": "input"})

    infer_coco_path: str = field(default=None, metadata={"persist": "input"})
    infer_gdf: gpd.GeoDataFrame = field(default=None, metadata={"persist": "input"})  # input slice, but stored via gdf_ref (gpkg), not state.json
    infer_gdf_columns_to_pass: set = field(default_factory=set, metadata={"persist": "input"})
    infer_gdf_columns_to_delete_on_save: List = field(default_factory=list, metadata={"persist": "input"})

    background_executor: Optional = field(default=None, metadata={"persist": "transient"})
    side_processes: List = field(default_factory=list, metadata={"persist": "transient"})

    component_output_folders: Dict = field(default_factory=dict, metadata={"persist": "registry"})
    component_output_files: Dict = field(default_factory=dict, metadata={"persist": "registry"})

    def apply_snapshot(self, snapshot, *, restore_registries: bool) -> None:
        """
        Restore persisted fields from a resume/initialize snapshot.

        Args:
            snapshot: a persistence.Snapshot exposing `.fields` (dict of restored field values)
                      and `.gdf` (the recovered GeoDataFrame, or None).
            restore_registries: if True (resume), also restore the output registries; if False
                                (initialize), restore only the "input" slice.
        """
        roles = ("input", "registry") if restore_registries else ("input",)
        for f in fields(self):
            if f.metadata.get("persist") in roles and f.name in snapshot.fields:
                setattr(self, f.name, snapshot.fields[f.name])
        gdf = snapshot.gdf  # read the gpkg once
        if gdf is not None:
            self.infer_gdf = gdf

    def update_infer_gdf(self, infer_gdf: gpd.GeoDataFrame) -> None:
        assert isinstance(infer_gdf, gpd.GeoDataFrame)
        assert object_id_column_name in infer_gdf.columns, f"Columns of the infer_gdf must contain a '{object_id_column_name}'."
        self.infer_gdf = infer_gdf

    def register_component_folder(self, component_name: str, component_id: int, folder_path: Path) -> None:
        """
        Register the output folder for a component.
        """
        key = get_component_folder_name(component_id, component_name)
        self.component_output_folders[key] = folder_path

    def register_output_file(self, component_name: str, component_id: int, file_type: str, file_path: Path) -> None:
        """
        Register an output file created by a component.
        """
        key = get_component_folder_name(component_id, component_name)

        if key not in self.component_output_files:
            self.component_output_files[key] = {}

        self.component_output_files[key][file_type] = file_path

    def get_component_folder(self, component_name: str, component_id: int) -> Optional[Path]:
        """Get the output folder for a specific component."""
        key = get_component_folder_name(component_id, component_name)
        return self.component_output_folders.get(key)

    def get_output_file(self, component_name: str, component_id: int, file_type: str) -> Optional[Path]:
        """Get a specific output file from a component."""
        key = get_component_folder_name(component_id, component_name)
        if key in self.component_output_files:
            return self.component_output_files[key].get(file_type)
        return None

    def get_latest_output_by_type(self, file_type: str) -> Optional[Path]:
        """Get the most recent output file of a specific type from any component."""
        latest_id = -1
        latest_path = None

        for key, files in self.component_output_files.items():
            if file_type in files:
                component_id = int(key.split('_')[0])
                if component_id > latest_id:
                    latest_id = component_id
                    latest_path = files[file_type]

        return latest_path

    def get_all_outputs(self) -> Dict:
        """Get all registered output files organized by component."""
        return self.component_output_files

    def clean_side_processes(self, key: str = None):
        """
        Wait for and process side processes.

        Args:
            key: If provided, only process side processes for this attribute.
                 If None, process all side processes.
        """
        to_process = []
        to_keep = []

        for side_process in self.side_processes:
            if key is None or (isinstance(side_process, tuple) and side_process[0] == key):
                to_process.append(side_process)
            else:
                to_keep.append(side_process)

        self.side_processes = to_keep

        for side_process in to_process:
            if isinstance(side_process, tuple):
                attribute_name = side_process[0]
                future_or_result = side_process[1]

                # Check if this is a Future object with a .result() method
                if hasattr(future_or_result, 'result'):
                    result = future_or_result.result()
                else:
                    result = future_or_result  # It's already a result

                # Update the data_state attribute
                if attribute_name:
                    setattr(self, attribute_name, result)

                # If there's registration info, register the output file
                if len(side_process) > 2 and isinstance(side_process[2], dict):
                    reg_info = side_process[2]

                    # If an expected_path was provided, use it
                    if 'expected_path' in reg_info:
                        file_path = Path(reg_info['expected_path'])
                    # Otherwise try to get a path from the result
                    elif isinstance(result, (str, Path)):
                        file_path = Path(result)
                    else:
                        file_path = None

                    if file_path:
                        # Register the component folder first
                        self.register_component_folder(
                            reg_info['component_name'],
                            reg_info['component_id'],
                            file_path.parent
                        )
                        # Then register the file
                        self.register_output_file(
                            reg_info['component_name'],
                            reg_info['component_id'],
                            reg_info['file_type'],
                            file_path
                        )

        # Clear processed side processes
        self.side_processes = []

        return self


# Guard: every DataState field must declare a "persist" role so the resume/initialize machinery
# stays in sync as the dataclass evolves. Adding a field without metadata fails fast here.
assert all("persist" in f.metadata for f in fields(DataState)), \
    "every DataState field must declare metadata={'persist': 'input'|'registry'|'transient'}"
