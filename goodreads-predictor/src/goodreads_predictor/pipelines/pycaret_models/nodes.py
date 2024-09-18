"""
This is a boilerplate pipeline 'pycaret_models'
generated using Kedro 0.19.6
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import BayesianRidge, Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold
import scipy 
from scipy.stats import uniform, loguniform
from typing import Any, Dict, List, Tuple

def preprocess_data(train_data: pd.DataFrame,test_data:pd.DataFrame ,features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    return X_train, y_train, X_test, y_test

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    return pd.concat([X.drop(columns=categorical_columns), encoded_df], axis=1)


def train_bayesian_ridge(X_train, y_train, param_distributions:Dict[str,Any], cv):
    model = BayesianRidge()
    random_search = RandomizedSearchCV(model, param_distributions, cv=cv, scoring='r2', n_jobs=2, random_state=42)
    result = random_search.fit(X_train, y_train)
    return result.best_estimator_

def predict_bayesian_ridge(model, X_test):
    return model.predict(X_test)


def train_ridge(X_train, y_train, param_grid, cv):
    modelR = Ridge()
    search_ridge = RandomizedSearchCV(modelR, param_grid, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
    result = search_ridge.fit(X_train, y_train)
    return result.best_estimator_

def predict_ridge(modelR, X_test):
    return modelR.predict(X_test)


def train_and_predict_models(train_data: pd.DataFrame, test_data: pd.DataFrame, features: List[str], target: str, config: Dict[str, Any]) -> pd.DataFrame:
    X_train, y_train = preprocess_data(train_data, features, target)
    X_test, y_test = preprocess_data(test_data, features, target)

    X_train_encoded = encode_categorical_features(X_train)
    X_test_encoded = encode_categorical_features(X_test)

  # Train models
    bayesian_model = train_bayesian_ridge(X_train, y_train, config['bayesian_params'], config['cv'])
    ridge_model = train_ridge(X_train, y_train, config['ridge_params'], config['cv'])

    # Make predictions
    bayesian_preds = predict_bayesian_ridge(bayesian_model, X_test)
    ridge_preds = predict_ridge(ridge_model, X_test)

    results = pd.DataFrame({
        "Actual": y_test,
        "BayesianRidge_Predicted": bayesian_preds,
        "Ridge_Predicted": ridge_preds
    })

    return results