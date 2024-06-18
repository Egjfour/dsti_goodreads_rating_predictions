"""
This is a boilerplate pipeline 'data_load'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline import node
from .nodes import copy


def create_pipeline(**kwargs) -> Pipeline:
    """
    Define the nodes to run
    """
    return pipeline([node(func = copy, inputs = ['books_raw'], outputs = 'books_loaded', name='load_raw_book_csv')])
