from sklearn.preprocessing import StandardScaler
import numpy as np

def scaler(X_train: np.ndarray, X_test: np.ndarray):
    """
    Scale the training and test data using StandardScaler.

    Args:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Test data.

    Returns:
        np.ndarray, np.ndarray: Scaled training and test data.
    """
    # Initialize the scaler
    standard_scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled