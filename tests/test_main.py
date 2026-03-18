import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_welcome_msg():
    """
    Test to check the welcome message is displayed as intended.
    """
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == 'Prediction model for whether a person earns over $50k or not'


def test_lower_50k(person_low_income):
    """
    Test to check predictions for lower income person.
    """
    resp = client.post('/predict', json=person_low_income)
    assert resp.status_code == 200
    assert resp.json() == {'prediction': '<=50K'}


def test_greater_50k(person_high_income):
    """
    Test to check predictions for higher income person.
    """
    resp = client.post('/predict', json=person_high_income)
    assert resp.status_code == 200
    assert resp.json() == {'prediction': '>50K'}
