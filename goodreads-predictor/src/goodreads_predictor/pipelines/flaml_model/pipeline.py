"""
This is a boilerplate pipeline 'flaml_model'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model, predict_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=train_model, inputs=["model_train", "model_valid", "params:FEATURES", "params:TARGET", "params:model_config"], outputs="flaml_model_trained", name = 'train_flaml_model'),
        node(func=predict_test, inputs=["flaml_model_trained", "model_test", "params:FEATURES", "params:TARGET"], outputs="flaml_model_test_results", name = 'get_flaml_model_preds'),
    ])
