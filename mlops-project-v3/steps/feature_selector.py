from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def feature_selector(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, method: str = "anova", k: int = 10):
    """
    Perform feature selection or dimensionality reduction using various algorithms.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training labels.
        method (str): Feature selection or dimensionality reduction method. Options:
            - "anova": ANOVA F-test
            - "rfe": Recursive Feature Elimination
            - "dti": Decision Tree Importance
            - "correlation": Correlation Coefficient
            - "pca": Principal Component Analysis
            - "lda": Linear Discriminant Analysis
        k (int): Number of top features or components to select.

    Returns:
        np.ndarray, np.ndarray: Transformed training and test features.
    """
    if method == "anova":
        # ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    elif method == "rfe":
        # Recursive Feature Elimination (RFE) with Logistic Regression
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    elif method == "dti":
        # Decision Tree Importance
        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X_train, y_train)
        feature_importances = tree.feature_importances_
        top_features = np.argsort(feature_importances)[-k:]  # Select top k features
        X_train_selected = X_train[:, top_features]
        X_test_selected = X_test[:, top_features]

    elif method == "correlation":
        # Correlation Coefficient
        correlation_matrix = np.corrcoef(X_train.T, y_train)[-1, :-1]
        top_features = np.argsort(np.abs(correlation_matrix))[-k:]
        X_train_selected = X_train[:, top_features]
        X_test_selected = X_test[:, top_features]

    elif method == "pca":
        # Principal Component Analysis (PCA)
        pca = PCA(n_components=k, random_state=42)
        X_train_selected = pca.fit_transform(X_train)
        X_test_selected = pca.transform(X_test)

    elif method == "lda":
        # Linear Discriminant Analysis (LDA)
        lda = LDA(n_components=k)
        X_train_selected = lda.fit_transform(X_train, y_train)
        X_test_selected = lda.transform(X_test)

    else:
        raise ValueError("Invalid method. Choose 'anova', 'rfe', 'dti', 'correlation', 'pca', or 'lda'.")

    return X_train_selected, X_test_selected