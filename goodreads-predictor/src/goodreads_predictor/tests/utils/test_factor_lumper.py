#pylint disable=import-error,missing-class-docstring,missing-module-docstring,line-too-long
import pytest
import json
import pandas as pd
import numpy as np
from goodreads_predictor.utils.factor_lumper import FactorLumper #pylint: disable=import-error

@pytest.fixture
def simulated_pandas_data():
    return pd.DataFrame({
        "col1": ["A" for _ in range(100)] + ["B" for _ in range(10)] + ["C" for _ in range(40)] + ["D" for _ in range(30)] + ["E" for _ in range(20)],
        "col2": ["A" for _ in range(40)] + ["B" for _ in range(80)] + ["C" for _ in range(15)] + ["D" for _ in range(15)] + ["E" for _ in range(50)]
    })

@pytest.fixture
def simulated_numpy_data():
    return np.array([
        ["A" for _ in range(100)] + ["B" for _ in range(10)] + ["C" for _ in range(40)] + ["D" for _ in range(30)] + ["E" for _ in range(20)],
        ["A" for _ in range(40)] + ["B" for _ in range(80)] + ["C" for _ in range(15)] + ["D" for _ in range(15)] + ["E" for _ in range(50)]
    ]).T

@pytest.fixture
def simulated_transform_data():
    return pd.DataFrame({
        "col1": ["A" for _ in range(5)] + ["B" for _ in range(3)] + ["F" for _ in range(2)],
        "col2": ["B" for _ in range(7)] + ["C" for _ in range(3)] 
    })


class TestFactorLumperInitialization:
    @staticmethod
    def test_initialization_defaults():
        """
        Test the initialization of the FactorLumper class with default values using the standard constructor
        """
        # Arrange
        test_factor_lumper = FactorLumper()

        # Assert
        assert test_factor_lumper.threshold == 30
        assert not test_factor_lumper.is_top_n
        assert not test_factor_lumper.is_percentage

    @staticmethod
    def test_initialization_with_threshold():
        """
        Test the initialization of the FactorLumper class with a threshold using the standard constructor
        """
        # Arrange
        test_factor_lumper = FactorLumper(threshold=20)

        # Assert
        assert test_factor_lumper.threshold == 20
        assert not test_factor_lumper.is_top_n
        assert not test_factor_lumper.is_percentage

    @staticmethod
    def test_initialization_failure_with_top_n_and_percentage():
        """
        Test the initialization of the FactorLumper class with both top_n and percentage set to True.
        This should raise a ValueError.
        """
        # Assert
        with pytest.raises(ValueError):
            FactorLumper(threshold = 10, is_top_n=True, is_percentage=True)
            
    @staticmethod
    def test_initialization_failure_threshold_not_int():
        """
        Test the initialization of the FactorLumper class with a threshold that is not an integer
        when is_percentage is False.
        This should raise a ValueError.
        """
        # Assert
        with pytest.raises(ValueError):
            FactorLumper(threshold = 10.5)

    @staticmethod
    def test_initialization_failure_threshold_not_percentage():
        """
        Test the initialization of the FactorLumper class with a threshold that is not a percentage
        when is_percentage is True.
        This should raise a ValueError.
        """
        # Assert
        with pytest.raises(ValueError):
            FactorLumper(threshold = 10, is_percentage=True)

    @staticmethod
    def test_initialization_failure_negative_threshold():
        """
        Test the initialization of the FactorLumper class with a negative threshold
        This should raise a ValueError.
        """
        # Assert
        with pytest.raises(ValueError):
            FactorLumper(threshold = -10)

    @staticmethod
    def test_load_from_prefitted():
        """
        Test the initialization of the FactorLumper class using the from_prefitted method
        We should see that the factor lumper dictionary is loaded with what was provided
        """
        # Arrange
        expected = {"col1": np.array(["A", "B"]), "col2": np.array(["E"])}

        # Act
        new_factor_lumper = FactorLumper.from_prefitted(valid_factor_levels={"col1": np.array(["A", "B"]), "col2": np.array(["E"])})
        result = new_factor_lumper.to_dict()

        # Assert
        for k, arr in expected.items():
            assert all(result[k] == arr)
        assert new_factor_lumper.__dict__['_FactorLumper__is_fitted'] # This should be set to True after loading the prefitted dictionary


class TestFactorLumpFit:
    @staticmethod
    def test_fit_pandas_defaults(simulated_pandas_data):
        """
        Test the fit method of the FactorLumper class using a pandas DataFrame
        Assumes that we are using a count-based lumping strategy with a threshold of 30
        Assumes that we also have a to_dict method that returns a dictionary with the column names as keys
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        expected = {
            'col1': ["a", "c", "d"],
            'col2': ["a", "b", "e"]
        }

        # Act
        test_factor_lumper.fit(simulated_pandas_data)
        result = test_factor_lumper.to_dict()

        # Assert
        for k, arr in expected.items():
            assert result.get(k, None) is not None
            assert all(result[k] == arr)

    @staticmethod
    def test_fit_numpy_defaults(simulated_numpy_data):
        """
        Test the fit method of the FactorLumper class using a numpy array
        Assumes that we are using a count-based lumping strategy with a threshold of 30
        Assumes that we also have a to_dict method that returns a dictionary with the column indices as keys
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        expected = {
            0: ["a", "c", "d"],
            1: ["a", "b", "e"]
        }

        # Act
        test_factor_lumper.fit(simulated_numpy_data)
        result = test_factor_lumper.to_dict()

        # Assert
        for k, arr in expected.items():
            assert result.get(k, None) is not None
            assert all(result[k] == arr)

    @staticmethod
    def test_fit_top_n(simulated_pandas_data):
        """
        Test the fit method of the FactorLumper class using a top n threshold returns the correct results
        """
        # Arrange
        test_factor_lumper = FactorLumper(threshold=2, is_top_n=True)
        expected = {
            'col1': ["c", "a"],
            'col2': ["e", "b"]
        }

        # Act
        test_factor_lumper.fit(simulated_pandas_data)
        result = test_factor_lumper.to_dict()

        # Assert
        for k, arr in expected.items():
            assert all(result[k] == arr)
        
    @staticmethod
    def test_fit_percentage(simulated_pandas_data):
        """
        Test the fit method of the FactorLumper class using a percentage threshold returns the correct results
        """
        # Arrange
        test_factor_lumper = FactorLumper(threshold=0.1, is_percentage=True)
        expected = {
            'col1': ["a", "c", "d", "e"],
            'col2': ["a", "b", "e"]
        }

        # Act
        test_factor_lumper.fit(simulated_pandas_data)
        result = test_factor_lumper.to_dict()

        # Assert
        for k, arr in expected.items():
            assert all(result[k] == arr)
        

class TestFactorLumpTransform:
    @staticmethod
    def test_transform_fails_not_fitted():
        """
        Test the transform method of the FactorLumper class when the object is not fitted
        This should raise a ValueError
        """
        # Arrange
        test_factor_lumper = FactorLumper()

        # Assert
        with pytest.raises(ValueError):
            test_factor_lumper.transform(pd.DataFrame())

    @staticmethod
    def test_transform_pandas(simulated_pandas_data, simulated_transform_data):
        """
        Test the transform method of the FactorLumper class using a pandas DataFrame
        Assumes that we are using a count-based lumping strategy with a threshold of 30
        Assumes that we also have a to_dict method that returns a dictionary with the column names as keys
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        expected = {
            "col1": ["a" for _ in range(5)] + ["other" for _ in range(3)] + ["other" for _ in range(2)],
            "col2": ["b" for _ in range(7)] + ["other" for _ in range(3)] 
        }

        # Act
        test_factor_lumper.fit(simulated_pandas_data)
        result = test_factor_lumper.transform(simulated_transform_data)

        # Assert
        assert isinstance(result, pd.DataFrame)
        for k, arr in expected.items():
            assert all(result[k] == arr)

    @staticmethod
    def test_transform_fails_column_mismatch(simulated_pandas_data, simulated_transform_data):
        """
        Test the transform method of the FactorLumper class when the data to transform has different columns
        than the data used to fit the object
        This should raise a ValueError
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        test_factor_lumper.fit(simulated_pandas_data)

        # Assert
        with pytest.raises(KeyError):
            test_factor_lumper.transform(pd.DataFrame({"col3": ["A" for _ in range(10)]}))

    @staticmethod
    def test_transform_fails_data_type_mismatch(simulated_numpy_data, simulated_transform_data):
        """
        Test the transform method of the FactorLumper class when the data to transform has a different data type
        than the data used to fit the object
        This should raise a ValueError
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        test_factor_lumper.fit(simulated_numpy_data)

        # Assert
        with pytest.raises(ValueError):
            test_factor_lumper.transform(simulated_transform_data)

    @staticmethod
    def test_call_override_invokes_transform(simulated_pandas_data, simulated_transform_data):
        """
        Test that the __call__ method of the FactorLumper class invokes the transform method
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        test_factor_lumper.fit(simulated_pandas_data)

        # Act
        result = test_factor_lumper(simulated_transform_data)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert all(result.columns == simulated_transform_data.columns)

    @staticmethod
    def test_fit_transform_returns_same_dtype(simulated_pandas_data):
        """
        Test the fit method of the FactorLumper class using a pandas DataFrame
        The transform method should return the same data type as the input data
        """
        # Arrange
        test_factor_lumper = FactorLumper()

        # Act
        result = test_factor_lumper.fit_transform(simulated_pandas_data)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert test_factor_lumper.__dict__['_FactorLumper__is_fitted'] # Need to access the attribute dict to get the private variable for testing

class TestMagicMethods:
    @staticmethod
    def test_str_method_not_yet_fitted(simulated_pandas_data):
        """
        Test the __str__ method of the FactorLumper class
        """
        # Arrange
        test_factor_lumper = FactorLumper()

        # Act
        result = str(test_factor_lumper)

        # Assert
        assert result == "FactorLumper has not been fitted yet. Please pass a dataframe or two-dimensional numpy array to the fit method"

    @staticmethod
    def test_str_method_fitted(simulated_pandas_data):
        """
        Test the __str__ method of the FactorLumper class
        """
        # Arrange
        test_factor_lumper = FactorLumper()
        expected = {
            'col1': 3,
            'col2': 3
        }
        expected_str = json.dumps(expected, indent = 4)        

        # Act
        test_factor_lumper.fit(simulated_pandas_data)
        result = str(test_factor_lumper)

        # Assert
        assert result == expected_str
