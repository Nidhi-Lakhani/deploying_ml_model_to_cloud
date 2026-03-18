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

@pytest.fixture
def person_high_income():
    """
    Dummy person for our prediction with high income. 
    """
    return {
        "age": 47,
        "workclass": "Private",
        "fnlgt": 209642,
        "education": "Masters",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 12000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }


@pytest.fixture
def person_low_income():
    """
    Dummy person for our prediction with low income. 
    """
    return {
        'age': 18,
        'workclass': 'State-gov',
        'fnlgt': 1000,
        'education': 'HS-grad',
        'education-num': 1,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 10,
        'native-country': 'United-States'
    }