import numpy as np
from sklearn.linear_model import LogisticRegression

from ml_scripts.pre_processing import process_data
from ml_scripts.model import train_model, model_inference, compute_metrics


def test_process_data_returns_correct_types(dummy_data, categorical_features):
    """
    Test to check whether process_data() returns the data types as expected.
    """
    X, y, encoder = process_data(dummy_data, categorical_features=categorical_features, label="salary")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


def test_train_model_returns_model(dummy_data, categorical_features):
    """
    Test to check whether train_model() returs the correct model instance (LogiesticRegression).
    """
    X, y, encoder = process_data(dummy_data, categorical_features=categorical_features, label="salary")
    model = train_model(X, y)

    assert isinstance(model, LogisticRegression)
    

def test_compute_metrics_returns_floats(dummy_data, categorical_features):
    """
    Test to that compute_metrics function returns the correct data types of the metrics.
    """
    X, y, encoder = process_data(dummy_data, categorical_features=categorical_features, label="salary")
    model = train_model(X, y)
    preds = model.predict(X)
    precision, recall, f1 = compute_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


def test_model_inference_length(dummy_data, categorical_features):
    """
    Test to check if the predictions by our model are the same size (value counts) as our input.
    """
    X, y, encoder = process_data(dummy_data, categorical_features=categorical_features, label="salary")
    model = train_model(X, y)
    preds = model_inference(model, X)

    assert len(preds) == X.shape[0]

