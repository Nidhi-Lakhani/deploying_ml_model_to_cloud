import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from pre_processing import process_data
from model import train_model, compute_metrics, save_artifacts, compute_slice_metrics

logging.basicConfig(level=logging.INFO)

# Define categorical features
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


def train_and_save_model(data_path):
    data = pd.read_csv(data_path)
    
    logging.info("Splitting data")
    train, test = train_test_split(
        data, test_size=0.3, stratify=data['salary'], random_state=42
    )

    logging.info("Pre-processing input data")
    X_train, y_train, encoder = process_data(
        train, categorical_features=categorical_features, label="salary", training=True
    )
    X_test, y_test, _ = process_data(
        test, categorical_features=categorical_features, label="salary",
        training=False, encoder=encoder
    )

    logging.info("Model training started")
    model = train_model(X_train, y_train)

    logging.info("Making predictions")
    y_pred = model.predict(X_test)

    precision, recall, f1 = compute_metrics(y_test, y_pred)
    logging.info(f"Overall Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    logging.info("Computing per slice metrics")
    # Convert test labels to numeric to match model predictions
    test_numeric = test.copy()
    test_numeric['salary'] = test_numeric['salary'].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    test_numeric['pred'] = y_pred

    # Compute metrics per slice
    for feature in categorical_features:
        compute_slice_metrics(
            df=test_numeric,
            feature=feature,
            y_true="salary",
            y_preds="pred"
        )

        save_artifacts(model, encoder)


if __name__ == "__main__":
    train_and_save_model(data_path="data/census.csv")