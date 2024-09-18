"""
This is a boilerplate pipeline 'pycaret_models'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import train_and_predict_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_and_predict_models,
                inputs=["train_data", "test_data", "params:features", "params:target", "params:model_config"],
                outputs="model_predictions",
                name="train_and_predict_models_node",
            ),
        ]
    )
