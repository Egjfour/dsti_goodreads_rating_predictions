"""
This is a boilerplate pipeline 'measurement'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import report_model_metrics, create_prediction_scatterplot
from functools import partial

# Define the experiment name for each of the plot functions using functools.partial
create_flaml_regressor_scatterplot = partial(create_prediction_scatterplot, experiment_name="FLAML Regressor")
create_flaml_regressor_scatterplot.__name__ = "create_flaml_regressor_scatterplot" # Set the name of the function to be used in the pipeline for logging purposes

# Same thing for the other models we tested
create_baysian_ridge_scatterplot = partial(create_prediction_scatterplot, experiment_name="Bayesian Ridge")
create_baysian_ridge_scatterplot.__name__ = "create_baysian_ridge_scatterplot"

create_ridge_scatterplot = partial(create_prediction_scatterplot, experiment_name="Ridge")
create_ridge_scatterplot.__name__ = "create_ridge_scatterplot"


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=report_model_metrics, inputs=["params:experiment_names", "flaml_model_test_results", "model_predictions_bayesian_ridge", "model_predictions_ridge"],\
             outputs="all_model_results", name="build_model_metrics"),

        # Create scatterplots of the test data predicted and actual values for each of the different models
        node(func=create_flaml_regressor_scatterplot, inputs={"results": "flaml_model_test_results",
                                                              "color_background": "params:color_gr_tan_background",
                                                              "color_dots": "params:color_gr_brown"},
             outputs="flaml_regressor_scatterplot", name="create_scatterplot_flmal_regressor"),

        # Create scatterplots of the test data predicted and actual values for each of the different models
        node(func=create_baysian_ridge_scatterplot, inputs={"results": "model_predictions_bayesian_ridge",
                                                              "color_background": "params:color_gr_tan_background",
                                                              "color_dots": "params:color_gr_brown"},
             outputs="bayesian_ridge_scatterplot", name="create_scatterplot_baysian_ridge"),

        # Create scatterplots of the test data predicted and actual values for each of the different models
        node(func=create_ridge_scatterplot, inputs={"results": "model_predictions_ridge",
                                                              "color_background": "params:color_gr_tan_background",
                                                              "color_dots": "params:color_gr_brown"},
             outputs="ridge_scatterplot", name="create_scatterplot_ridge")
    ])
