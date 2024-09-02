"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import data_load, word_embeddings, data_processing, model_data_creation, flaml_model


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["data_load"] = data_load.create_pipeline()
    pipelines["word_embeddings"] = word_embeddings.create_pipeline()
    pipelines["data_processing"] = data_processing.create_pipeline()
    pipelines["model_data_creation"] = model_data_creation.create_pipeline()
    
    # Custom pipeline runs for the models
    flaml_model_pipeline = flaml_model.create_pipeline()

    # Add the model pipelines to the pipelines dictionary
    pipelines["flaml_model"] = flaml_model_pipeline

    # Custom pipeline runs
    pipelines['full_data_load'] = data_load.create_pipeline() + word_embeddings.create_pipeline() # Import from blob and all external calls
    pipelines['build_model_data_from_loaded'] = data_processing.create_pipeline() + model_data_creation.create_pipeline() # Run all data processing without reloading data
    pipelines['load_and_build_model_data'] = pipelines.get('full_data_load') + pipelines.get('build_model_data_from_loaded') # Run all data processing with reloading data
    pipelines['run_all_models'] = flaml_model_pipeline # Run all models -- add to this as more models are created

    return pipelines
