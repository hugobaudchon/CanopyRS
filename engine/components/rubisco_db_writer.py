import os
from pathlib import Path

from engine.components.base import BaseComponent
from engine.config_parsers.rubisco_db_writer import RubiscoDbWriterConfig
from engine.data_state import DataState
from rubisco_db.db_writer import InferRunWriter, PredictionsWriter
from rubisco_db.utils.logging import get_db_engine


class RubiscoDbWriterComponent(BaseComponent):
    name = 'rubisco_db_writer'

    def __init__(self, config: RubiscoDbWriterConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def __call__(self, data_state: DataState) -> DataState:
        if not self.config.enabled:
            print("RubiscoDbWriter is disabled. Set 'enabled: true' in the configuration to enable it.")
            return data_state

        # Debug: Print all available component output files
        print("DEBUG: Available component output files:")
        for key, files in data_state.component_output_files.items():
            print(f"  {key}: {files}")

        # Strategy: 
        # 1. If classifier is enabled, use classifier output (contains both boxes and masks with proper classes)
        # 2. Otherwise, fall back to aggregator outputs

        final_predictions_gpkg = None

        # First, check if we have classifier outputs (preferred)
        if self.config.model_id_classifier is not None:
            # Look for classifier component outputs
            classifier_files = []
            for key, files in data_state.component_output_files.items():
                if 'classifier' in key and 'gpkg' in files:
                    component_id = int(key.split('_')[0])
                    classifier_files.append((component_id, files['gpkg']))

            # Use the latest classifier output (highest component ID)
            if classifier_files:
                classifier_files.sort(key=lambda x: x[0], reverse=True)  # Sort by component_id descending
                final_predictions_gpkg = classifier_files[0][1]
                print(f"Using classifier predictions: {final_predictions_gpkg}")

        # Fallback to aggregator logic if no classifier output found
        if final_predictions_gpkg is None:
            # Keep track of aggregator ids to identify first (detector) and second (segmenter) aggregators
            aggregator_ids = []

            for key, files in data_state.component_output_files.items():
                if 'aggregator' in key and 'gpkg' in files:
                    component_id = int(key.split('_')[0])
                    aggregator_ids.append((component_id, files['gpkg']))

            # Sort by component ID 
            aggregator_ids.sort()

            # Use the latest aggregator output, or first if only one exists
            if len(aggregator_ids) >= 1:
                # If we have multiple aggregators, use the latest one (likely after classifier/segmenter)
                final_predictions_gpkg = aggregator_ids[-1][1]
                print(f"Using latest aggregator predictions: {final_predictions_gpkg}")
            else:
                print("No classifier or aggregator GPKG file found. Database writing skipped.")
                return data_state


        # Verify the file exists
        if not Path(final_predictions_gpkg).exists():
            print(f"ERROR: Predictions file not found at {final_predictions_gpkg}")
            return data_state

        print(f"Writing configuration and predictions to database...")
        print(f"Using final predictions GeoPackage: {final_predictions_gpkg}")

        try:
            # Get engine connection
            engine = get_db_engine()

            # Write configuration 
            # Make sure the pipeline folder actually exists
            pipeline_folder = self.config.config_folder_path

            if not os.path.exists(pipeline_folder):
                # Try to find config in the current directory structure
                project_root = Path(__file__).resolve().parents[2]  # Go up to project root
                pipeline_folder = str(project_root / "config" / pipeline_folder)
                print(f"Config folder not found at {self.config.config_folder_path}, trying {pipeline_folder}")

                if not os.path.exists(pipeline_folder):
                    print(f"ERROR: Cannot find configuration folder at {pipeline_folder}")
                    return data_state

            inference_writer = InferRunWriter(engine)
            print(f"Writing config from {pipeline_folder} with model_id_detector={self.config.model_id_detector}")

            infer_run_id = inference_writer.write_config(
                pipeline_folder=pipeline_folder,
                model_id_detector=self.config.model_id_detector,
                model_id_classifier=self.config.model_id_classifier
            )

            if infer_run_id is None:
                print("ERROR: Failed to write configuration to database - infer_run_id is None")
                return data_state

            print(f"Configuration written to database with inference ID: {infer_run_id}")

            # Write predictions
            predictions_writer = PredictionsWriter(engine)
            added_boxes, added_masks = predictions_writer.write_predictions(
                infer_run_id=infer_run_id,
                boxes_path=str(final_predictions_gpkg),  # Same file for both
                masks_path=str(final_predictions_gpkg),  # Same file for both
                has_classifier=self.config.model_id_classifier is not None
            )

            print(f"Predictions written to database: {added_boxes} boxes, {added_masks} masks")

            # Register outputs in data_state
            return self.update_data_state(data_state, infer_run_id)

        except Exception as e:
            print(f"ERROR writing to database: {str(e)}")
            import traceback
            traceback.print_exc()
            return data_state

    def update_data_state(self, data_state: DataState, infer_run_id: int) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)

        # Record the inference ID in a text file for reference
        infer_run_id_path = self.output_path / "infer_run_id.txt"
        with open(infer_run_id_path, 'w') as f:
            f.write(f"Inference ID: {infer_run_id}\n")

        data_state.register_output_file(self.name, self.component_id, 'infer_run_id', infer_run_id_path)

        return data_state
