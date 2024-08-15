"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    apply_filters_and_consolidate,
    join_data,
    create_filters_waterfall_plot,
    aggregate_exclusions_data,
    create_data_filters,
    create_descriptions_lookup
    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = create_descriptions_lookup, inputs=['open_library_book_api_info'],
             outputs='descriptions_lookup', name='create_descriptions_lookup'),
        node(func = join_data, inputs=['books_loaded', 'descriptions_lookup', 'price_by_isbn', 'book_genres'],
             outputs='joined_data', name='join_book_data'),
        node(func = create_data_filters, inputs=['joined_data'], outputs='books_filters', name='create_data_filters'),
        node(func = apply_filters_and_consolidate, inputs=['books_filters'],
             outputs='filtered_books', name='apply_filters_and_consolidate'),
        node(func = aggregate_exclusions_data, inputs=['books_filters'],
             outputs=['exclusions_summary', 'walk_data'], name='aggregate_exclusions_data'),
        node(func = create_filters_waterfall_plot,
             inputs=['walk_data', "params:color_gr_purple", "params:color_gr_green",
                     "params:color_gr_brown", "params:color_gr_tan_background"],
             outputs='scope_waterfall_plot', name='create_scope_walk')
    ])
