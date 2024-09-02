"""
This is a boilerplate pipeline 'flaml_model'
generated using Kedro 0.19.6
"""
from typing import Any, Dict, List
import pandas as pd
from flaml import AutoML

def train_model(train_data: pd.DataFrame, valid_data: pd.DataFrame, features: List[str], target: str, config: Dict[str, Any]) -> AutoML:
    """
    Trains a model using the FLAML AutoML framework.

    Args:
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        features (List[str]): The list of feature column names.
        target (str): The target column name.
        config (Dict[str, Any]): The configuration parameters for the AutoML model.

    Returns:
        AutoML: The trained AutoML model.
    """
    X_train = train_data[features]
    y_train = train_data[target]

    X_valid = valid_data[features]
    y_valid = valid_data[target]

    automl = AutoML()
    automl.fit(X_train, y_train, X_val = X_valid, y_val = y_valid, **config, verbose = -1)

    return automl

def predict_test(model: AutoML, test_data: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """
    Predicts the target variable using the given model on the test data.

    Args:
        model (AutoML): The trained model used for prediction.
        test_data (pd.DataFrame): The test data containing the features and target variable.
        features (List[str]): The list of feature column names used for prediction.
        target (str): The name of the target variable column.

    Returns:
        pd.DataFrame: A DataFrame containing the actual and predicted values of the target variable.
    """
    X_test = test_data[features]
    y_test = test_data[target]

    preds = model.predict(X_test)

    return pd.DataFrame({"Actual": y_test, "Predicted": preds})
