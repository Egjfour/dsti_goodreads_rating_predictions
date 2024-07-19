"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines import data_load, word_embeddings, data_processing


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
    return pipelines
