"""
This is a boilerplate pipeline 'data_load'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline import node
from .nodes import copy, query_slugbooks_price_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Define the nodes to run
    """
    return pipeline([
          node(func = copy, inputs = ['books_raw'], outputs = 'books_loaded', name='load_raw_book_csv')
        # , node(func = query_slugbooks_price_data, inputs = ['books_loaded'], outputs = 'price_by_isbn', name='identify_book_prices') # RUN THIS FIRST
        , node(func = query_slugbooks_price_data, inputs = ['books_loaded', 'price_by_isbn_input'], outputs = 'price_by_isbn', name='identify_book_prices')
        ])
