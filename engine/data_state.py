from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd

@dataclass
class DataState:
    imagery_path: str = None
    parent_output_path: str = None

    tiles_path: str = None
    tiles_names: list = None

    infer_coco_path: str = None
    infer_gdf: gpd.GeoDataFrame = None
    infer_gdf_columns_to_pass: set = field(default_factory=set)
    infer_gdf_columns_to_delete_on_save: List = field(default_factory=list)

    ground_truth_coco_path: str = None
    ground_truth_gdf: gpd.GeoDataFrame = None
    ground_truth_gdf_columns_to_pass: set = field(default_factory=set)

    side_processes: List = field(default_factory=list)
    
    component_output_folders: Dict = field(default_factory=dict)
    component_output_files: Dict = field(default_factory=dict)

    def register_component_folder(self, component_name: str, component_id: int, folder_path: Path) -> None:
        """
        Register the output folder for a component.
        """
        key = f"{component_id}_{component_name}"
        self.component_output_folders[key] = folder_path
    
    def register_output_file(self, component_name: str, component_id: int, file_type: str, file_path: Path) -> None:
        """
        Register an output file created by a component.
        """
        key = f"{component_id}_{component_name}"
        
        if key not in self.component_output_files:
            self.component_output_files[key] = {}
            
        self.component_output_files[key][file_type] = file_path
    
    def get_component_folder(self, component_name: str, component_id: int) -> Optional[Path]:
        """Get the output folder for a specific component."""
        key = f"{component_id}_{component_name}"
        return self.component_output_folders.get(key)
    
    def get_output_file(self, component_name: str, component_id: int, file_type: str) -> Optional[Path]:
        """Get a specific output file from a component."""
        key = f"{component_id}_{component_name}"
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
