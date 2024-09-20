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
                inputs=["model_train", "model_test", "params:features_reg_mdl", "params:target_reg_mdl", "params:model_config_reg_mdl"],
                outputs=["model_predictions_bayesian_ridge", "model_predictions_ridge"],
                name="train_and_predict_models_node",
            ),
        ]
    )
