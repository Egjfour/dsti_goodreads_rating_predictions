"""
FactorLumper class which will be used to transform categorical attributes
by lumping together levels that are infrequently occuring within the dataset
"""
from typing_extensions import Union, Self, Dict, List
import json
import pandas as pd
import numpy as np
# pylint: disable=invalid-name,unused-private-member,line-too-long

class FactorLumper:
    """
    A class for lumping factor levels based on a threshold.

    Args:
        threshold (Union[float, int], optional): The threshold value for lumping factor levels. Defaults to 30.
        is_top_n (bool, optional): If True, lump the factor levels based on the top N levels. Defaults to False.
        is_percentage (bool, optional): If True, lump the factor levels based on a percentage threshold. Defaults to False.

    Raises:
        ValueError: Raised if both is_top_n and is_percentage are True.

    Attributes:
        is_top_n (bool): Indicates if the factor levels are lumped based on the top N levels.
        is_percentage (bool): Indicates if the factor levels are lumped based on a percentage threshold.
        threshold (Union[float, int]): The threshold value for lumping factor levels.
        __valid_factor_levels (dict): A dictionary to store the valid factor levels.
        __is_fitted (bool): Indicates if the factor lumper has been fit on the training data.
        __is_pandas_fitted (bool): Indicates if the data provided during fitting was a pandas dataframe.
    """
    def __init__(self,
                 threshold: Union[float, int] = 30,
                 is_top_n: bool = False,
                 is_percentage: bool = False):
        if is_top_n and is_percentage:
            raise ValueError("is_top_n and is_percentage cannot be True at the same time")
        self.is_top_n = is_top_n
        self.is_percentage = is_percentage
        self.threshold = threshold
        self.__valid_factor_levels = {} # This is where we will store the valid factor levels
        self.__is_fitted = False # This will let us know if the factor lumper has been fit on the training data
        self.__is_pandas_fitted = False # This will let us know if the data provided during fitting was a pandas dataframe

    @classmethod
    def from_prefitted(cls, valid_factor_levels: dict, **kwargs) -> Self:
        """
        This method will allow us to create a FactorLumper object from a dictionary of valid factor levels which will be useful for deployment with new datasets as well as test

        Args:
            valid_factor_levels (dict): A dictionary of valid factor levels
            **kwargs: Additional keyword arguments to pass to the default constructor

        Returns:
            FactorLumper: A FactorLumper object that has been fitted with the valid factor levels
        """
        # Initialize the class noramlly
        fl = cls(**kwargs)

        # Override the dictionary with one that has already been fitted
        fl.__valid_factor_levels = valid_factor_levels

        # Check if the keys are strings, if so, we know the data was a pandas dataframe since those are column names
        if isinstance(list(valid_factor_levels.keys())[0], str):
            fl.__is_pandas_fitted = True

        # Set the is_fitted flag to True
        fl.__is_fitted = True

        return fl

    # Define a property for threshold which will let us valate the consturctor arguments
    @property
    def threshold(self) -> Union[float, int]:
        """
        Getter for the threshold property
        """
        return self.__threshold
    
    @threshold.setter
    def threshold(self, value):
        if value < 0:
            raise ValueError("Threshold must be a positive number")
        if self.is_percentage and value > 1:
            raise ValueError("Percentage threshold should be between 0 and 1")
        if not self.is_percentage and (not int(value) == value or value < 1):
            raise ValueError("Threshold must be a integer greater than 1")
        self.__threshold = value

    # Private method that will actually do the work to identify which levels are valid
    def __get_valid_levels(self, x: Union[pd.Series, np.array]) -> List[str]:
        """
        Get the valid levels from the input array. Take into consideration the threshold and whether it is a percentage or not
        NOTE: This method will lowercase and strip all the strings as well

        Args:
            x (Union[pd.Series, np.array]): The input array.
        Returns:
            List[str]: The list of valid levels.
        """
        # Convert the input to a numpy array if it is not already
        if isinstance(x, pd.Series):
            x = x.values

        # Get the unique levels and their counts
        unique, counts = np.unique(np.char.strip(np.char.lower(x.astype(str))), return_counts=True)

        # Reset the threshold if it is a percentage
        if self.is_percentage:
            threshold = self.__threshold * len(x)
        else:
            threshold = self.__threshold

        # Grab the top n levels if that is what is specified
        if self.is_top_n:
            valid_levels = unique[counts.argsort()[-threshold:]]
        else:
            # Otherwise, grab levels that have counts greater than the threshold
            valid_levels = unique[counts >= threshold]
        return valid_levels

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Fit the factor lumper on the input data using the private method to get the valid levels in a list
        This method will store the valid levels in the __valid_factor_levels dictionary

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data to fit on

        Returns:
            None
        """
        if isinstance(X, pd.DataFrame):
            self.__is_pandas_fitted = True
            for col in X.columns:
                self.__valid_factor_levels[col] = self.__get_valid_levels(X[col])
        elif isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                self.__valid_factor_levels[i] = self.__get_valid_levels(X[:, i])
        else:
            raise ValueError("Input data type not supported")
        
        self.__is_fitted = True
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforms the input data by replacing invalid factor levels with "other".
        NOTE: This method will lowercase and strip all the strings as well

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data to be transformed. It can be either a pandas DataFrame or a numpy ndarray.

        Returns:
            Union[pd.DataFrame, np.ndarray]: The transformed data with invalid factor levels replaced by "other".
        """
        if not self.__is_fitted:
            raise ValueError("FactorLumper has not been trained yet")
        if not self.__is_pandas_fitted and isinstance(X, pd.DataFrame):
            raise ValueError("FactorLumper was not fitted on a pandas dataframe. Cannot use one for transformation")
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                X[col] = X[col].apply(lambda x: x.lower().strip() if x.lower().strip() in self.__valid_factor_levels[col] else "other")
        elif isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                X[:, i] = np.array([x.lower().strip() if x.lower().strip() in self.__valid_factor_levels[i] else "other" for x in X[:, i]])
        else:
            raise ValueError("Input data type not supported")
        
        return X
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fits the factor lumper model to the input data and transforms it.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input data to fit and transform.

        Returns:
            Union[pd.DataFrame, np.ndarray]: The transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def __call__(self, x: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Wrapper to allow the FactorLumper instance to be called like a function for transformation ONLY

        Args:
            x (Union[pd.DataFrame, np.ndarray]): The input data to be transformed. It can be either a pandas DataFrame or a numpy ndarray.
        
        Returns:
            Union[pd.DataFrame, np.ndarray]: The transformed data with invalid factor levels replaced by "other".
        """
        # Calling the factor lumper expects the data is already fitted
        return self.transform(x)
    
    def __str__(self):
        """
        Pretty-print the valid factor levels dictionary as a JSON string
        For brevity, we will only print the counts of the valid factor levels for each column

        Args:
            None

        Returns:
            str: The pretty-printed JSON string
        """
        if not self.__is_fitted:
            return "FactorLumper has not been fitted yet. Please pass a dataframe or two-dimensional numpy array to the fit method"
        
        # Consolidate the lists stored in the dictionary to a count for printablity
        counts = {k: len(v) for k, v in self.__valid_factor_levels.items()}

        # Pretty-print using JSON
        return json.dumps(counts, indent=4)
    
    def to_dict(self) -> Dict[Union[str, int], List[str]]:
        """
        Returns the valid factor levels dictionary

        Args:
            None

        Returns:
            Dict[Union[str, int], List[str]]: The valid factor levels dictionary
        """
        return self.__valid_factor_levels
    
    def to_json(self) -> str:
        """
        Returns the valid factor levels dictionary as a JSON string (useful for Kedro outputs)

        Args:
            None

        Returns:
            str: The valid factor levels dictionary as a JSON string
        """
        return json.dumps(self.__valid_factor_levels)
    
    def check_fitted(self) -> bool:
        """
        Check if the factor lumper has been fitted

        Args:
            None

        Returns:
            bool: True if the factor lumper has been fitted, False otherwise
        """
        return self.__is_fitted
