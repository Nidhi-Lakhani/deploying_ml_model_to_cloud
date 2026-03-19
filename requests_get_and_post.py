import requests
import logging

logging.basicConfig(level=logging.INFO)


render_url = 'https://ml-to-cloud.onrender.com/predict'

# using the same hih earner example from conftest.py
example_data = {
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

logging.info(f"POSTing {example_data} to {render_url}")
req = requests.post(render_url, json=example_data)
logging.info(f"Response status code: {req.status_code}")
logging.info(f"Response message: {req.json()}")
