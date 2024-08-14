"""
This is a boilerplate pipeline 'model_data_creation'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import create_factor_lumper, create_train_test_split, apply_factor_lumping, save_model_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_factor_lumper,
            inputs=["params:threshold", "params:is_top_n", "params:is_percentage"],
            outputs="factor_lumper_init",
            name = "create_factor_lumper"
        ),
        node(
            func=create_train_test_split,
            inputs=["filtered_books", "params:target_col", "params:test_size"],
            outputs=["train_raw", "valid_raw", "test_raw"],
            name = "create_train_test_split"
        ),
        node(
            func=apply_factor_lumping,
            inputs=["train_raw", "valid_raw", "test_raw", "factor_lumper_init"],
            outputs=["train_lumped", "valid_lumped", "test_lumped", "factor_lumper"]
        ),
        node(
            func=save_model_data,
            inputs=["train_lumped", "valid_lumped", "test_lumped"],
            outputs=['model_train', 'model_test', 'model_valid'],
            name = "saving_train_valid_test_split"
        )
    ])
