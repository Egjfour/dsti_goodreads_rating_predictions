"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import apply_book_attributes, merge_description_embeddings, perform_clustering_analysis



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        
        node(func=merge_description_embeddings, inputs=['filtered_books', 'description_embeddings'], outputs='merged_df', name='merge_description_embeddings'),
        node(func=perform_clustering_analysis, inputs=['merged_df'], outputs='books_clustered', name='perform_clustering_analysis'),
        node(func=apply_book_attributes, inputs=['books_clustered'], outputs=['books_features', 'feature_cutoffs'], name='add_engineered_features'),
    ])
