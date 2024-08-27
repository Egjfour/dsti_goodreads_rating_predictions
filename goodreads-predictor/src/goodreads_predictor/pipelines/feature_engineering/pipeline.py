"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import apply_book_attributes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=apply_book_attributes, inputs=['filtered_books'], outputs=['books_features', 'feature_cutoffs'], name='add_engineered_features')
    ])
