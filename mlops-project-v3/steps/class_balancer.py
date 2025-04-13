from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import numpy as np

def balance_classes(X_train: np.ndarray, y_train: np.ndarray, method: str = "oversample", algorithm: str = "random"):
    """
    Balance the class distribution in the training data using various resampling algorithms.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        method (str): Balancing method - "oversample" or "undersample".
        algorithm (str): Resampling algorithm to use. Options:
            - For oversampling: "random", "smote"
            - For undersampling: "random", "nearmiss"

    Returns:
        np.ndarray, np.ndarray: Balanced training features and labels.
    """
    if method == "oversample":
        if algorithm == "random":
            sampler = RandomOverSampler(random_state=42)
        elif algorithm == "smote":
            sampler = SMOTE(random_state=42)
        else:
            raise ValueError("Invalid algorithm for oversampling. Choose 'random' or 'smote'.")
    elif method == "undersample":
        if algorithm == "random":
            sampler = RandomUnderSampler(random_state=42)
        elif algorithm == "nearmiss":
            sampler = NearMiss()
        else:
            raise ValueError("Invalid algorithm for undersampling. Choose 'random' or 'nearmiss'.")
    else:
        raise ValueError("Invalid method. Choose 'oversample' or 'undersample'.")

    # Apply the selected resampling algorithm
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced