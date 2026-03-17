import pytest
import pandas as pd


@pytest.fixture
def dummy_data():
    """
    Sample dataset for testing our ML model.
    """
    return pd.DataFrame({
        "age": [25, 30, 40],
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "education": ["HS-grad", "Bachelors", "Masters"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Tech-support", "Sales", "Exec-managerial"],
        "relationship": ["Not-in-family", "Unmarried", "Husband"],
        "race": ["White", "Black", "Asian-Pac-Islander"],
        "sex": ["Male", "Female", "Male"],
        "native-country": ["United-States", "United-States", "India"],
        "salary": [" <=50K", " >50K", " <=50K"]
    })

@pytest.fixture
def categorical_features():
    """
    List of the categorical features we have in our dataset as a simulation for the unit tests.
    """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]