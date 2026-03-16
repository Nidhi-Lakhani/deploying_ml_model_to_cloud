import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import dump

from pre_processing import process_data

logging.basicConfig(level=logging.INFO)

data_slice_path = 'model_artifacts/slice_output.txt'
model_path = 'model_artifacts/lr_model.joblib'
encoder_path = 'model_artifacts/onehot_encoder.joblib'


def train_model(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (np.array): training data
        y_train (np.array): labels

    Returns:
        model (LogisticRegression): trained LR model
    """
    logging.info("Training model")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def model_inference(model, X):
    """
    Run model inference.

    Args:
        model (LogisticRegression): a trained LR model
        X (np.array): data we need predictions on
    
    Returns:
        preds (np.array): model predictions
    """
    logging.info("Making predictions")
    preds = model.predict(X)
    return preds


def compute_metrics(y_true, preds):
    """
    Compute model performance metrics.

    Args:
        y_true (np.array): correct labels of the validation/test set
        preds (np.array): labels the model predicted

    Returns:
        precision (float): precision metric of the model
        recall (float): recall metric of the model
        f1 (float): f1-score of the model
    """
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    return precision, recall, f1


def compute_slice_metrics(
    df,
    feature,
    y_true,
    y_preds,
    output_file=data_slice_path,
):
    """
    Compute model metrics for each slice of a feature in the dataset.

    Args:
        df (DataFrame): dataframe containing the features and labels
        feature (str): feature name to slice on
        y_true (str): column name for true labels
        y_preds (str): column name for model predictions
        output_file (str): file to save slice metrics to

    Returns:
        None
    """
    results = []

    for value in df[feature].unique():
        slice_df = df[df[feature] == value]
        y_true_slice = slice_df[y_true].values
        y_preds_slice = slice_df[y_preds].values

        logging.info("Computing model metrics")
        precision, recall, f1 = compute_metrics(y_true_slice, y_preds_slice)

        logging.info("Appending the metrics to results dict")
        line = (
            f"{feature}={value} | "
            f"Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, "
            f"F1: {f1:.3f}"
        )

        results.append(line)

    logging.info(f"Saving the data slice metrics file to {data_slice_path}")
    with open(output_file, "w") as f:
        for r in results:
            f.write(r + "\n")


def save_artifacts(model, encoder):
    """
    Saves the trained model and encoder.

    Args:
        model (LogisticRegression): trained LR model
        encoder (OneHotEncoder): trained encoder
    """
    logging.info(f"Saving model and encoder to {model_path} and {encoder_path}")
    dump(model, model_path)
    dump(encoder, encoder_path)