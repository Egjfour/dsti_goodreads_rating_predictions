# pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import create_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=create_embeddings, inputs=['books_loaded'], outputs='books_embedded', name='apply_word_embeddings')
    ])
