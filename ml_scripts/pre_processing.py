import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def process_data(
    df,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
):
    """
    Function to pre-process data by performing one-hot encoding of categorical features
    
    Args:
        df (DataFrame): Pandas dataframe containing the features and label
        categorical_features (list): List of columns needed to be encoded
        label (str): Name of the label column
        training (bool): Boolean to indicate true when training, false during inference
        encoder (OneHotEncoder): Trained encoder for use during inference pipeline

    Returns:
        X (NDArray): Processed feature matrix.
        y (NDArray): Processed labels.
        encoder (OneHotEncoder): Fitted encoder
    """

    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = df[label]
        y = y.apply(lambda x: 1 if x.strip() == ">50K" else 0).values
        X = df.drop(columns=[label])
    else:
        y = np.array([])
        X = df.copy()

    # Split categorical and continuous columns
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features).values

    if training:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)

    # Combine numerical and encoded categorical features
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder