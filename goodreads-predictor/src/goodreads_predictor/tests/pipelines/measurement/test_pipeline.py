"""
This is a boilerplate test file for pipeline 'measurement'
generated using Kedro 0.19.6.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
from goodreads_predictor.pipelines.measurement.nodes import *

@pytest.fixture
def experiment_names():
    return ["FLAML Regressor"]

@pytest.fixture
def experiment_names_multiple():
    return ["FLAML Regressor", "Classification Model"]

@pytest.fixture
def flaml_model_test_results():
    return pd.DataFrame({"Actual": [3, 4, 5], "Predicted": [3.2, 4.1, 4.9]})

@pytest.fixture
def classification_model_test_results():
    return pd.DataFrame({"Actual": [3,4,5], "Predicted": [3.5, 4.25, 4.75]})

class TestMeasurementPipeline:
    @staticmethod
    def test_report_metrics_singular(flaml_model_test_results, experiment_names):
        results = report_model_metrics(experiment_names, flaml_model_test_results)
        assert results.shape[0] == 1
        assert results["Experiment"].iloc[0] == "FLAML Regressor"

    @staticmethod
    def test_report_metrics_multiple(flaml_model_test_results, classification_model_test_results, experiment_names_multiple):
        results = report_model_metrics(experiment_names_multiple, flaml_model_test_results, classification_model_test_results)
        assert results.shape[0] == 2
        assert results["Experiment"].iloc[0] == "FLAML Regressor"
        assert results["Experiment"].iloc[1] == "Classification Model"

    @staticmethod
    def test_all_desired_metrics_included(flaml_model_test_results, experiment_names):
        results = report_model_metrics(experiment_names, flaml_model_test_results)
        assert "MSE" in results.columns
        assert "RMSE" in results.columns
        assert "MAE" in results.columns
        assert "R2" in results.columns

    @staticmethod
    def test_error_handling_for_experiment_names_mismatch(flaml_model_test_results, experiment_names_multiple):
        with pytest.raises(ValueError):
            report_model_metrics(experiment_names_multiple, flaml_model_test_results)

    @staticmethod
    def test_create_scatterplot_creates_scatterplot(flaml_model_test_results, experiment_names):
        create_prediction_scatterplot(flaml_model_test_results, "FLAML Regressor", "tan", "brown")
        # Check that the plot was created
        assert plt.gcf()
