"""
This is a boilerplate pipeline 'model_data_creation'
generated using Kedro 0.19.6
"""
from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit # We want to make sure that we have an adequate amount of each class in the train and test sets
from goodreads_predictor.utils.factor_lumper import FactorLumper

# Set a seed to ensure the data split always return the same result
SEED = 123
np.random.seed(SEED)

def create_factor_lumper(threshold: int, is_top_n: bool, is_percentage: bool) -> FactorLumper:
    """
    Create a FactorLumper object with the given parameters.
    We use this helper to make it work within a Kedro pipeline.

    Args:
        threshold (int): The threshold value for lumping factors.
        is_top_n (bool): If True, lump the top N factors based on the threshold value.
        is_percentage (bool): If True, treat the threshold value as a percentage.

    Returns:
        FactorLumper: The created FactorLumper object.
    """
    return FactorLumper(threshold=threshold, is_top_n=is_top_n, is_percentage=is_percentage)

def create_train_test_split(data: pd.DataFrame, target_col: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train-test split based on the target column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_col (str): The target column name.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing train and test DataFrames.
    """
    if not isinstance(data[target_col][0], str):
        data['stratify_col'] = data[target_col] > data[target_col].median()
    else:
        data['stratify_col'] = data[target_col]

    # Split the data into train and test sets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_index, test_index_init = list(splitter.split(data, data['stratify_col']))[0]
    train = data.iloc[train_index].reset_index(drop=True)
    test_init = data.iloc[test_index_init].reset_index(drop=True)

    # Split the test set into test and validation sets. This time we will hard-code a 50/50 split
    splitter_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    test_index, valid_index = list(splitter_valid.split(test_init, test_init['stratify_col']))[0]
    test = test_init.iloc[test_index].reset_index(drop=True)
    valid = test_init.iloc[valid_index].reset_index(drop=True)

    return train, valid, test


def apply_factor_lumping(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame,
                         factor_lumper: FactorLumper) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FactorLumper]:
    """
    Prepare the input data for the model by lumping factors

    Args:
        train_data (pd.DataFrame): The training data DataFrame.
        valid_data (pd.DataFrame): The validation data DataFrame.
        test_data (pd.DataFrame): The test data DataFrame.
        factor_lumper (FactorLumper): The factor lumper object used for lumping factors.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FactorLumper]: A tuple containing the prepared training data,
        validation data, and test data DataFrames as well as the fitted factor lumper object.
    """
    # Lump factors for Categorical Variables
    # First fit on the training data, then transform the validation and test data
    cat_data = train_data.select_dtypes(include=['object'])
    num_data = train_data.select_dtypes(exclude=['object'])
    cat_data = factor_lumper.fit_transform(cat_data)
    train_data = pd.concat([cat_data, num_data], axis=1)

    for dt in [valid_data, test_data]:
        cat_data = dt.select_dtypes(include=['object'])
        num_data = dt.select_dtypes(exclude=['object'])
        cat_data = factor_lumper.transform(cat_data)
        dt = pd.concat([cat_data, num_data], axis=1)

    return train_data, valid_data, test_data, factor_lumper

def save_model_data(*dataframes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Passthrough function that will let us define a constant final node for the pipeline

    Args:
        *dataframes: The dataframes for train/valid/test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The dataframes for train/valid/test.
    """
    return dataframes
