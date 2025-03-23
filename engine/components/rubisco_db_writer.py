import os
from pathlib import Path

from engine.components.base import BaseComponent
from engine.config_parsers.rubisco_db_writer import RubiscoDbWriterConfig
from engine.data_state import DataState
from rubisco_db.db_writer import InferenceRunWriter, PredictionsWriter
from rubisco_db.utils.logging import get_db_engine


class RubiscoDbWriterComponent(BaseComponent):
    name = 'rubisco_db_writer'

    def __init__(self, config: RubiscoDbWriterConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        
    def __call__(self, data_state: DataState) -> DataState:
        if not self.config.enabled:
            print("RubiscoDbWriter is disabled. Set 'enabled: true' in the configuration to enable it.")
            return data_state

        # Find detector aggregator path (first aggregator) and segmenter aggregator path (second aggregator)
        detector_aggregator_gpkg = None
        segmenter_aggregator_gpkg = None
        
        # Keep track of aggregator ids to identify first (detector) and second (segmenter) aggregators
        aggregator_ids = []
        
        for key, files in data_state.component_output_files.items():
            if 'aggregator' in key and 'gpkg' in files:
                component_id = int(key.split('_')[0])
                aggregator_ids.append((component_id, files['gpkg']))
        
        # Sort by component ID 
        aggregator_ids.sort()
        
        # First aggregator is detector, second is segmenter
        if len(aggregator_ids) >= 1:
            detector_aggregator_gpkg = aggregator_ids[0][1]
        if len(aggregator_ids) >= 2:
            segmenter_aggregator_gpkg = aggregator_ids[1][1]
        
        if detector_aggregator_gpkg is None:
            print("No detector aggregator GPKG file found. Database writing skipped.")
            return data_state
            
        # Get output paths to verify they exist
        print(f"Writing configuration and predictions to database...")
        print(f"Using detector GeoPackage: {detector_aggregator_gpkg}")
        if segmenter_aggregator_gpkg:
            print(f"Using segmenter GeoPackage: {segmenter_aggregator_gpkg}")
        else:
            print("No segmenter GeoPackage found. Only detector results will be stored.")
        
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
            
            inference_writer = InferenceRunWriter(engine)
            print(f"Writing config from {pipeline_folder} with model_id_detector={self.config.model_id_detector}")
            
            inference_id = inference_writer.write_config(
                pipeline_folder=pipeline_folder,
                model_id_detector=self.config.model_id_detector,
                model_id_classifier=self.config.model_id_classifier
            )
            
            if inference_id is None:
                print("ERROR: Failed to write configuration to database - inference_id is None")
                return data_state
                
            print(f"Configuration written to database with inference ID: {inference_id}")
            
            # Write predictions
            predictions_writer = PredictionsWriter(engine)
            added_boxes, added_masks = predictions_writer.write_predictions(
                inference_id=inference_id,
                boxes_path=str(detector_aggregator_gpkg),
                masks_path=str(segmenter_aggregator_gpkg) if segmenter_aggregator_gpkg else None,
                has_classifier=self.config.model_id_classifier is not None
            )
            
            print(f"Predictions written to database: {added_boxes} boxes, {added_masks} masks")
            
            # Register outputs in data_state
            return self.update_data_state(data_state, inference_id)
            
        except Exception as e:
            print(f"ERROR writing to database: {str(e)}")
            import traceback
            traceback.print_exc()
            return data_state
    
    def update_data_state(self, data_state: DataState, inference_id: int) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)
        
        # Record the inference ID in a text file for reference
        inference_id_path = self.output_path / "inference_id.txt"
        with open(inference_id_path, 'w') as f:
            f.write(f"Inference ID: {inference_id}\n")
            
        data_state.register_output_file(self.name, self.component_id, 'inference_id', inference_id_path)
        
        return data_state
