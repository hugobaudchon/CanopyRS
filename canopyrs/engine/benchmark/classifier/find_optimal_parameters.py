# engine/benchmark/classifier/find_optimal_parameters.py
import warnings
import pandas as pd

def find_optimal_classifier_params_placeholder(*args, **kwargs) -> pd.DataFrame:
    """
    Placeholder for finding optimal parameters for classifier post-processing.
    Currently, this is not used as evaluation is primarily tile-level COCO.
    """
    warnings.warn("Optimal classifier parameter search is not yet implemented.")
    return pd.DataFrame([{"status": "not_implemented", "reason": "find_optimal_classifier_params_placeholder"}])