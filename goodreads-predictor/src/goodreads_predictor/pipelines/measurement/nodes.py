"""
This is a boilerplate pipeline 'measurement'
generated using Kedro 0.19.6
"""
from typing import List
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def report_model_metrics(experiment_names: List[str], *experiments) -> pd.DataFrame:
    """
    Reports the model metrics for all the experiments.

    Args:
        experiment_names (List[str]): The list of experiment names.
        experiments (List[pd.DataFrame]): The list of DataFrames containing the metrics for each experiment.
            Set this up as *args to allow variable num args in the pipeline

    Returns:
        pd.DataFrame: A DataFrame containing the desired model metrics for all the experiments.
    """
    all_results = pd.DataFrame()
    for results, name in zip(experiments, experiment_names):
        actual = results["Actual"]
        predicted = results["Predicted"]
        mse = mean_squared_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted, squared=False)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        results = pd.DataFrame({"Experiment": [name], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "R2": [r2]})
        all_results = pd.concat([all_results, results])
    
    return all_results

def create_prediction_scatterplot(results: pd.DataFrame, experiment_name: str, color_background: str, color_dots: str) -> None:
    """
    Creates a scatterplot of the predicted vs. actual values for all the experiments.

    Args:
        experiments (List[pd.DataFrame]): The list of DataFrames containing the metrics for each experiment.
        experiment_names (List[str]): The list of experiment names.

    Returns:
        None
    """
    # Update the styling of the plots
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titlepad"] = 20
    plt.rcParams["figure.facecolor"] = color_background

    # Create the scatter plot
    plt.scatter(results["Actual"], results["Predicted"], color=color_dots)
    plt.plot([2.5, 5.5], [2.5, 5.5], color="black", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted Values - Model: {}".format(experiment_name))
    return plt
