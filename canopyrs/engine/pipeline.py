"""
Pipeline with centralized I/O and merge handling.

Design principles:
1. Pipeline handles all I/O (saving gpkg, COCO generation, state updates)
2. Pipeline handles GDF merging with clear rules
3. Components only flatten outputs and do component-specific validation
4. Pre-run validation catches config errors before expensive processing
5. Helper function for standalone component usage

Merge rules (in _merge_result_gdf):
- No existing GDF → set directly, assign object_ids if missing
- Result has geometry → becomes base, merge in other columns from existing
- Result has no geometry → merge attributes into existing
- Merge key priority: object_id > tile_path (if unique) > error
"""

import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Set, Optional, Union

import geopandas as gpd
import pandas as pd

from geodataset.utils import (
    GeoPackageNameConvention, validate_and_convert_product_name, strip_all_extensions_and_path
)

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, ComponentValidationError
from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.pipeline_visualizer import PipelineFlowVisualizer
from canopyrs.engine.raster_validation import validate_input_raster_or_tiles, RasterValidationError
from canopyrs.engine.utils import (
    generate_future_coco, green_print, get_component_folder_name,
    parse_tilerizer_aoi_config, object_id_column_name, tile_path_column_name
)

class PipelineValidationError(ComponentValidationError):
    """Raised when pipeline validation fails."""
    pass

class Pipeline:
    """
    Orchestrates component execution with centralized I/O handling.

    Responsibilities:
    - Component instantiation and ordering
    - Pre-run validation of entire pipeline
    - State management (updating DataState from ComponentResult)
    - File I/O (saving gpkg, COCO generation)
    - Output registration
    - Background task management
    """

    def __init__(
        self,
        components: List[BaseComponent],
        data_state: DataState,
        output_path: Path,
    ):
        """
        Initialize pipeline.

        Args:
            components: List of component instances (already configured)
            data_state: Initial data state
            output_path: Base output directory
        """
        self.components = components
        self.data_state = data_state
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup background executor for async COCO generation
        self.background_executor = ProcessPoolExecutor(max_workers=1)
        self.data_state.background_executor = self.background_executor

        # Assign component IDs and output paths
        for i, component in enumerate(self.components):
            component.component_id = i
            component.output_path = self.output_path / f"{i}_{component.name}"

        # Validate pipeline configuration immediately to catch errors early
        self._print_flow_chart()
        self._validate_pipeline(raise_on_error=True)

    @classmethod
    def from_config(cls, io_config: InferIOConfig, config: PipelineConfig) -> 'Pipeline':
        """
        Create a Pipeline from configuration objects.

        This matches the interface of the original pipeline.py for backward compatibility.

        Args:
            io_config: Input/output configuration
            config: Pipeline configuration with component configs

        Returns:
            Configured Pipeline instance
        """
        # Import component classes here to avoid circular imports
        from canopyrs.engine.components.aggregator import AggregatorComponent
        from canopyrs.engine.components.detector import DetectorComponent
        from canopyrs.engine.components.segmenter import SegmenterComponent
        from canopyrs.engine.components.tilerizer import TilerizerComponent
        from canopyrs.engine.components.classifier import ClassifierComponent

        output_path = Path(io_config.output_folder)

        # Initialize AOI configuration (Area of Interest, used by the Tilerizer)
        infer_aois_config = parse_tilerizer_aoi_config(
            aoi_config=io_config.aoi_config,
            aoi_type=io_config.aoi_type,
            aois={INFER_AOI_NAME: io_config.aoi}
        )

        # Instantiate components from config
        components = []
        for component_id, (component_type, component_config) in enumerate(config.components_configs):
            if component_type == 'tilerizer':
                component = TilerizerComponent(
                    component_config, output_path, component_id, infer_aois_config
                )
            elif component_type == 'detector':
                component = DetectorComponent(component_config, output_path, component_id)
            elif component_type == 'aggregator':
                component = AggregatorComponent(component_config, output_path, component_id)
            elif component_type == 'segmenter':
                component = SegmenterComponent(component_config, output_path, component_id)
            elif component_type == 'classifier':
                component = ClassifierComponent(component_config, output_path, component_id)
            else:
                raise ValueError(f'Invalid component type: {component_type}')
            components.append(component)

        # Initialize data state from the io (input/output) config
        infer_gdf = gpd.read_file(io_config.input_gpkg) if io_config.input_gpkg else None
        infer_gdf_columns_to_pass = (
            set(io_config.infer_gdf_columns_to_pass)
            if io_config.infer_gdf_columns_to_pass else set()
        )

        # If an infer_gdf from a previous pipeline run is provided,
        # make sure to pass the special columns if present
        for special_column_name in [object_id_column_name, tile_path_column_name]:
            if infer_gdf is not None and special_column_name in infer_gdf.columns:
                infer_gdf_columns_to_pass.add(special_column_name)

        # Derive product name from imagery path, or use default for tiled input
        if io_config.input_imagery:
            product_name = validate_and_convert_product_name(
                strip_all_extensions_and_path(Path(io_config.input_imagery))
            )
        else:
            product_name = "tiled_input"

        data_state = DataState(
            imagery_path=io_config.input_imagery,
            parent_output_path=io_config.output_folder,
            product_name=product_name,
            tiles_path=io_config.tiles_path,
            infer_coco_path=io_config.input_coco,
            infer_gdf=infer_gdf,
            infer_gdf_columns_to_pass=infer_gdf_columns_to_pass,
        )

        green_print("Pipeline initialized")

        return cls(
            components=components,
            data_state=data_state,
            output_path=output_path,
        )

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def run(self, strict_rgb_validation: bool = True) -> DataState:
        """
        Run the pipeline.

        Args:
            strict_rgb_validation: If True, enforce strict RGB band validation

        Returns:
            Final DataState with all outputs
        """

        # Validate input raster/tiles bands
        self._validate_input_data(strict_rgb_validation)

        try:
            for component in self.components:
                green_print(f"Running {component.name}...")

                self._wait_for_required_state(component)

                # Run component - returns ComponentResult
                component.output_path.mkdir(parents=True, exist_ok=True)
                result = component(self.data_state)

                # Pipeline handles all I/O and state updates
                self._process_result(component, result)

            # Final cleanup of async tasks
            self.data_state.clean_side_processes()

            # Save final GDF to root output folder if one was produced
            if self.data_state.infer_gdf is not None:
                final_gpkg_path = self._save_final_gpkg()
                num_polygons = len(self.data_state.infer_gdf)
                green_print(f"Final GDF containing {num_polygons} polygons saved to: {final_gpkg_path}")

            green_print("Pipeline finished")

        finally:
            self.background_executor.shutdown(wait=True)

        return self.data_state

    def __call__(self) -> DataState:
        """Alias for run()."""
        return self.run()

    def _wait_for_required_state(self, component: BaseComponent) -> None:
        """Wait for any required state still being produced by background processes."""
        if self.data_state.side_processes is None or len(self.data_state.side_processes) == 0:
            return
        for key in component.requires_state:
            if getattr(self.data_state, key, None) is None:
                self.data_state.clean_side_processes(key)

    def _process_result(self, component: BaseComponent, result: ComponentResult):
        """
        Process ComponentResult: update state and handle I/O.

        This is where all I/O and merging is centralized.
        """
        # Register component folder
        self.data_state.register_component_folder(
            component.name, component.component_id, component.output_path
        )

        # Apply state updates (e.g., tiles_path, infer_coco_path)
        for key, value in result.state_updates.items():
            setattr(self.data_state, key, value)

        # Merge GDF if provided (this replaces the old update_infer_gdf call)
        if result.gdf is not None and len(result.gdf) > 0:
            merged_gdf = self._merge_result_gdf(result.gdf)
            self.data_state.infer_gdf = merged_gdf

        # Update columns to pass based on what's actually in the merged GDF
        # (not just what the component claims to produce, since merge may add columns)
        if self.data_state.infer_gdf is not None:
            # Use all non-geometry columns from the merged GDF
            actual_columns = set(self.data_state.infer_gdf.columns) - {Col.GEOMETRY}
            self.data_state.infer_gdf_columns_to_pass = actual_columns

        # Register any files the component already wrote itself (e.g. geodataset internals)
        for file_type, file_path in result.output_files.items():
            if file_path is not None:
                self.data_state.register_output_file(
                    component.name, component.component_id, file_type, Path(file_path)
                )

        # Save GeoPackage if requested (use merged GDF from data_state)
        if result.save_gpkg and self.data_state.infer_gdf is not None and len(self.data_state.infer_gdf) > 0:
            gpkg_path = self._save_gpkg(component, self.data_state.infer_gdf, result.gpkg_name_suffix)
            file_type = 'gpkg' if result.gpkg_name_suffix == 'aggregated' else 'pre_aggregated_gpkg'
            self.data_state.register_output_file(
                component.name, component.component_id, file_type, gpkg_path
            )

        # Queue COCO generation if requested (use merged GDF from data_state)
        if result.save_coco and self.data_state.infer_gdf is not None and len(self.data_state.infer_gdf) > 0:
            future_coco = self._queue_coco_generation(component, result)
            if future_coco:
                self.data_state.side_processes.append(future_coco)

    def _merge_result_gdf(self, result_gdf: Union[gpd.GeoDataFrame, pd.DataFrame]) -> gpd.GeoDataFrame:
        """
        Merge component output with existing infer_gdf.

        Rules:
        1. No existing GDF → set directly, assign object_ids if missing
        2. Result has geometry + valid merge key → becomes base, merge in other columns
        3. Result has geometry + no merge key → full replacement (e.g., aggregator)
        4. Result has no geometry → merge attributes into existing
        5. Merge key priority: object_id > tile_path (if unique)

        Args:
            result_gdf: Output from component (GeoDataFrame or DataFrame)

        Returns:
            Merged GeoDataFrame
        """
        existing_gdf = self.data_state.infer_gdf

        # Case 1: No existing GDF - set directly
        if existing_gdf is None:
            return self._set_as_new_gdf(result_gdf)

        # Check if result has geometry
        has_geometry = Col.GEOMETRY in result_gdf.columns and result_gdf[Col.GEOMETRY].notna().any()

        # Try to determine merge key
        merge_key = self._determine_merge_key(result_gdf, existing_gdf, raise_on_error=False)

        # Case 2: Result has geometry
        if has_geometry:
            if merge_key:
                # Merge with existing to get other columns
                return self._merge_with_new_geometry(result_gdf, existing_gdf, merge_key)
            else:
                # No merge key - full replacement
                return self._set_as_new_gdf(result_gdf)

        # Case 3: Result has no geometry - must merge into existing
        if not merge_key:
            raise ValueError(
                "Cannot merge DataFrame into existing GDF: no valid merge key found. "
                f"Result columns: {list(result_gdf.columns)}"
            )
        
        # no new geometry, merge attributes into existing
        return self._merge_into_existing(result_gdf, existing_gdf, merge_key)

    def _set_as_new_gdf(self, result_gdf: Union[gpd.GeoDataFrame, pd.DataFrame]) -> gpd.GeoDataFrame:
        """Set result as new infer_gdf, assigning object_ids if missing."""
        # Assign object_ids if not present
        if Col.OBJECT_ID not in result_gdf.columns:
            result_gdf = result_gdf.copy()
            result_gdf[Col.OBJECT_ID] = range(len(result_gdf))

        # Ensure it's a GeoDataFrame
        if isinstance(result_gdf, gpd.GeoDataFrame):
            return result_gdf

        # Convert DataFrame to GeoDataFrame (no geometry)
        if Col.GEOMETRY in result_gdf.columns:
            return gpd.GeoDataFrame(result_gdf, geometry=Col.GEOMETRY)
        return gpd.GeoDataFrame(result_gdf)

    def _determine_merge_key(
        self,
        result_gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
        existing_gdf: gpd.GeoDataFrame,
        raise_on_error: bool = True
    ) -> Optional[str]:
        """Determine which column to use for merging."""
        # Try object_id first
        if (Col.OBJECT_ID in result_gdf.columns and
            Col.OBJECT_ID in existing_gdf.columns and
            result_gdf[Col.OBJECT_ID].notna().all()):
            return Col.OBJECT_ID

        # Fall back to tile_path if unique in result
        if (Col.TILE_PATH in result_gdf.columns and
            Col.TILE_PATH in existing_gdf.columns and
            result_gdf[Col.TILE_PATH].is_unique):
            return Col.TILE_PATH

        if raise_on_error:
            raise ValueError(
                "Cannot merge GDFs: need valid object_id or unique tile_path. "
                f"Result columns: {list(result_gdf.columns)}, "
                f"Existing columns: {list(existing_gdf.columns)}"
            )
        return None

    def _merge_with_new_geometry(
        self,
        result_gdf: gpd.GeoDataFrame,
        existing_gdf: gpd.GeoDataFrame,
        merge_key: str
    ) -> gpd.GeoDataFrame:
        """Merge when result has new geometry (e.g., segmenter masks replace detector boxes)."""
        # Get columns from existing that aren't in result (except geometry)
        existing_cols_to_keep = [merge_key]
        for col in existing_gdf.columns:
            if col not in result_gdf.columns and col != Col.GEOMETRY:
                existing_cols_to_keep.append(col)

        # Merge: result is base, pull in other columns from existing
        merged = result_gdf.merge(
            existing_gdf[existing_cols_to_keep],
            on=merge_key,
            how='left'
        )

        # Warn about unmatched rows
        unmatched = merged[merge_key].isna().sum() if merge_key in merged.columns else 0
        if unmatched > 0:
            warnings.warn(f"{unmatched} rows in result had no match in existing GDF")

        # Check if merge_key is unique in merged. If merge key is OBJECT_ID, warn and assign new unique object_ids to duplicates. If merge key is TILE_PATH, raise error.
        if merged[merge_key].duplicated().any():
            if merge_key == Col.OBJECT_ID:
                warnings.warn(f"Duplicate OBJECT_IDs found in merged GDF. Assigning new unique OBJECT_IDs. This can happen if a component duplicated objects, like a labeled tilerizer with overlap > 0.")
                merged = merged.copy()
                # Identify duplicates (keep='first' ensures the first occurrence retains its ID)
                duplicates_mask = merged.duplicated(subset=[Col.OBJECT_ID], keep='first')
                # Find the current highest ID to start incrementing from
                current_max_id = merged[Col.OBJECT_ID].max()
                num_duplicates = duplicates_mask.sum()
                # Generate a range of new IDs
                new_ids = range(current_max_id + 1, current_max_id + 1 + num_duplicates)
                # Assign new IDs only to the rows identified as duplicates
                merged.loc[duplicates_mask, Col.OBJECT_ID] = new_ids
            else: 
                raise ValueError(f"Duplicate TILE_PATHs found in merged GDF after merging. This should not happen as TILE_PATH is the merge_key and is expected to be unique.")

        return gpd.GeoDataFrame(merged, geometry=Col.GEOMETRY, crs=result_gdf.crs)

    def _merge_into_existing(
        self,
        result_gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
        existing_gdf: gpd.GeoDataFrame,
        merge_key: str
    ) -> gpd.GeoDataFrame:
        """Merge attributes into existing GDF (e.g., classifier adds scores)."""
        # Drop geometry from result if present (we keep existing geometry)
        result_cols = [col for col in result_gdf.columns if col != Col.GEOMETRY]
        result_data = result_gdf[result_cols]

        # Merge: existing is base, add new columns from result
        merged = existing_gdf.merge(
            result_data,
            on=merge_key,
            how='left'
        )

        # Warn about unmatched rows
        new_cols = [col for col in result_cols if col != merge_key]
        if new_cols:
            unmatched = merged[new_cols[0]].isna().sum()
            if unmatched > 0:
                warnings.warn(f"{unmatched} rows in existing GDF had no match in result")

        return gpd.GeoDataFrame(merged, geometry=Col.GEOMETRY, crs=existing_gdf.crs)

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def _save_gpkg(self, component: BaseComponent, gdf: gpd.GeoDataFrame, suffix: str) -> Path:
        """Save GeoDataFrame to GeoPackage."""
        gpkg_name = self._generate_gpkg_name(suffix)
        gpkg_path = component.output_path / gpkg_name
        gdf.to_file(gpkg_path, driver='GPKG')
        return gpkg_path

    def _save_final_gpkg(self) -> Path:
        """Save final GeoDataFrame to root output folder."""
        gpkg_name = self._generate_gpkg_name("final")
        gpkg_path = self.output_path / gpkg_name
        self.data_state.infer_gdf.to_file(gpkg_path, driver='GPKG')
        return gpkg_path

    def _generate_gpkg_name(self, suffix: str) -> str:
        """Generate GeoPackage filename using product name from data state."""
        product_name = self.data_state.product_name or "output"
        fold = f"{INFER_AOI_NAME}{suffix}" if suffix else INFER_AOI_NAME
        return GeoPackageNameConvention.create_name(
            product_name=product_name,
            fold=fold,
            scale_factor=1.0,
            ground_resolution=None
        )

    def _queue_coco_generation(self, component: BaseComponent, result: ComponentResult):
        """Queue async COCO generation."""
        return generate_future_coco(
            future_key=StateKey.INFER_COCO_PATH,
            executor=self.background_executor,
            component_name=component.name,
            component_id=component.component_id,
            description=f"{component.name} inference",
            gdf=self.data_state.infer_gdf,  # Use merged GDF from data_state
            tiles_paths_column=Col.TILE_PATH,
            polygons_column=Col.GEOMETRY,
            scores_column=result.coco_scores_column,
            categories_column=result.coco_categories_column,
            other_attributes_columns=result.produced_columns - {Col.GEOMETRY, Col.TILE_PATH},
            output_path=component.output_path,
            use_rle_for_labels=False,
            n_workers=4,
            coco_categories_list=None
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_input_data(self, strict_rgb_validation: bool = True) -> None:
        """
        Validate input raster/tiles at pipeline start.

        Uses utility functions from raster_validation module to check RGB band properties.

        Args:
            strict_rgb_validation: If True, raise error for missing color interpretation.
                                    If False, only warn if color interpretation is not R,G,B.

        Raises:
            PipelineValidationError: If validation fails
        """
        try:
            if self.data_state.imagery_path:
                green_print("Validating input raster bands...")
                validate_input_raster_or_tiles(
                    imagery_path=self.data_state.imagery_path,
                    strict_color_interp=strict_rgb_validation
                )
                green_print("Input raster validation passed")
            elif self.data_state.tiles_path:
                green_print("Validating input tiles...")
                validate_input_raster_or_tiles(
                    tiles_path=self.data_state.tiles_path,
                    strict_color_interp=strict_rgb_validation
                )
                green_print("Tile validation passed")
        except RasterValidationError as e:
            # Convert to PipelineValidationError for consistency
            raise PipelineValidationError(str(e))

    def _validate_pipeline(self, raise_on_error: bool = True) -> List[str]:
        """
        Validate entire pipeline before running.

        Simulates state flow through components to catch errors early.

        Args:
            raise_on_error: If True, raise PipelineValidationError on first error

        Returns:
            List of all validation errors (empty if valid)
        """
        all_errors = []

        # Start with initial state
        available_state = self._get_initial_state_keys()
        available_columns = self._get_initial_columns()

        # Simulate running through each component
        for i, component in enumerate(self.components):
            # Validate component
            errors = component.validate(
                available_state=available_state,
                available_columns=available_columns,
                raise_on_error=False,
            )

            if errors:
                all_errors.append(f"Component {i} ({component.name}):")
                all_errors.extend(f"  {e}" for e in errors)

            # Update available state/columns with what this component produces
            available_state = available_state | component.produces_state
            available_columns = available_columns | component.produces_columns

        if all_errors and raise_on_error:
            error_msg = "Pipeline validation failed:\n" + "\n".join(all_errors)
            raise PipelineValidationError(error_msg)

        return all_errors

    def _print_flow_chart(self) -> None:
        """
        Print a flow chart showing state/column availability through the pipeline.

        This visualizes the data flow through all components, showing what each
        component requires, produces, and what passes through.
        """
        visualizer = PipelineFlowVisualizer(
            components=self.components,
            initial_state_keys=self._get_initial_state_keys(),
            initial_columns=self._get_initial_columns(),
        )
        visualizer.print()

    def _get_initial_state_keys(self) -> Set[str]:
        """Get state keys available at pipeline start."""
        available = set()
        if self.data_state.imagery_path:
            available.add(StateKey.IMAGERY_PATH)
        if self.data_state.tiles_path:
            available.add(StateKey.TILES_PATH)
        if self.data_state.infer_gdf is not None:
            available.add(StateKey.INFER_GDF)
        if self.data_state.infer_coco_path:
            available.add(StateKey.INFER_COCO_PATH)
        if self.data_state.product_name:
            available.add(StateKey.PRODUCT_NAME)
        return available

    def _get_initial_columns(self) -> Set[str]:
        """Get GDF columns available at pipeline start."""
        if self.data_state.infer_gdf is not None:
            return set(self.data_state.infer_gdf.columns)
        return set()


# =============================================================================
# Standalone Helper
# =============================================================================

def run_component(
    component: BaseComponent,
    output_path: str = None,
    imagery_path: str = None,
    tiles_path: str = None,
    infer_gdf: gpd.GeoDataFrame = None,
    infer_coco_path: str = None,
    product_name: str = None,
    **kwargs
) -> DataState:
    """
    Run a single component standalone (wraps it in a Pipeline).

    This is a convenience function for users who want to run a single
    component without manually creating a Pipeline. For a more discoverable
    API with explicit signatures, use each component's ``run_standalone()``
    classmethod instead.

    Args:
        component: The component to run
        output_path: Where to save outputs (required)
        imagery_path: Path to imagery (for tilerizer)
        tiles_path: Path to tiles (for detector, segmenter)
        infer_gdf: Input GeoDataFrame (for aggregator, classifier)
        infer_coco_path: Path to COCO file (for segmenter, classifier)
        product_name: Name for output files (derived from imagery if not provided)
        **kwargs: Additional DataState attributes

    Returns:
        DataState with component outputs

    Example:
        from canopyrs.engine.components.detector import DetectorComponent
        from canopyrs.engine.config_parsers import DetectorConfig

        config = DetectorConfig(model='faster_rcnn_detectron2', ...)
        detector = DetectorComponent(config)

        result = run_component(
            detector,
            output_path='./output',
            tiles_path='./tiles'
        )
        print(result.infer_gdf)
    """
    if output_path is None:
        raise ValueError("output_path is required for run_component()")

    # Validate inputs against component requirements before creating Pipeline
    available_state = {
        key for key, value in {
            StateKey.IMAGERY_PATH: imagery_path,
            StateKey.TILES_PATH: tiles_path,
            StateKey.INFER_GDF: infer_gdf,
            StateKey.INFER_COCO_PATH: infer_coco_path,
            StateKey.PRODUCT_NAME: product_name,
        }.items() if value is not None
    }
    available_columns = set(infer_gdf.columns) if infer_gdf is not None else set()

    errors = component.validate(
        available_state=available_state,
        available_columns=available_columns,
        raise_on_error=False,
    )
    if errors:
        error_msg = (
            f"Cannot run '{component.name}' standalone - missing inputs:\n"
            + "\n".join(f"  * {e}" for e in errors)
            + f"\n\n{component.describe()}"
        )
        raise ComponentValidationError(error_msg)

    # Derive product name if not provided
    if product_name is None:
        if imagery_path:
            product_name = validate_and_convert_product_name(
                strip_all_extensions_and_path(Path(imagery_path))
            )
        else:
            product_name = "tiled_input"

    # Create DataState from provided inputs
    data_state = DataState(
        imagery_path=imagery_path,
        tiles_path=tiles_path,
        product_name=product_name,
        infer_gdf=infer_gdf,
        infer_coco_path=infer_coco_path,
        parent_output_path=output_path,
        **kwargs
    )

    # Create and run pipeline with single component
    pipeline = Pipeline(
        components=[component],
        data_state=data_state,
        output_path=output_path,
    )

    return pipeline.run()
