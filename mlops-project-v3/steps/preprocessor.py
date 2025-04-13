import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocessor(data: pd.DataFrame):
    """
    Preprocess the data by handling outliers and splitting into train and test sets.

    Args:
        data (pd.DataFrame): The dataset to preprocess.

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: X_train, X_test, y_train, y_test
    """
    # Outlier treatment
    for column in data.select_dtypes(include=["float64", "int64"]).columns:
        data = data[(data[column] <= data[column].mean() + 3 * data[column].std()) &
                    (data[column] >= data[column].mean() - 3 * data[column].std())]

    # Split features and target
    X = data.drop(['date', 'score'], axis=1)
    y = data['score']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test