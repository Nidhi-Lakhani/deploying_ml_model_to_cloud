import logging 

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal

from joblib import load
from ml_scripts.pre_processing import process_data

logging.basicConfig(level=logging.INFO)

app = FastAPI()

model = load("model_artifacts/lr_model.joblib")
encoder = load("model_artifacts/onehot_encoder.joblib")

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Defining Pydantic body
class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: Literal["Male", "Female"]
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

    model_config = {
        "json_schema_extra": {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }
    }



@app.get("/")
async def read_root():
    return "Prediction model for whether a person earns over $50k or not"


@app.post("/predict")
async def predict(body: Person):
    logging.info("Running API POST call now")
    # using model_dump as dict is deprecated
    input_dict = body.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])
    X, _, _ = process_data(
        df,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder
    )

    pred = model.predict(X)[0]
    result = ">50K" if pred == 1 else "<=50K"
    logging.info(f"Got response: {result}")

    return {"prediction": result}