# pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import create_embeddings, init_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=init_model, inputs=None, outputs='embedding_model', name='init_embedding_model')

    ,   node(func=create_embeddings, inputs=['embedding_model', 'books_loaded', 'params:title_column', 'params:original_data_key_column'],
             outputs='title_embeddings_original', name='apply_word_embeddings_titles_original')

    ,   node(func=create_embeddings, inputs=['embedding_model', 'books_loaded', 'params:author_column', 'params:original_data_key_column'],
             outputs='authors_embeddings', name='apply_word_embeddings_authors')

    ,   node(func=create_embeddings, inputs=['embedding_model', 'descriptions_lookup', 'params:description_column', 'params:original_data_key_column'],
             outputs='description_embeddings', name='apply_word_embeddings_descriptions')
    ])
