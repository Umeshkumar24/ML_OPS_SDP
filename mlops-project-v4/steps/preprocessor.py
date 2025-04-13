from zenml.steps import step
from sklearn.model_selection import train_test_split
import pandas as pd

@step
def preprocessor(data: pd.DataFrame):
    """
    Preprocess the data and split into train and test sets.
    """
    X = data.drop(['date', 'score'], axis=1)
    y = data['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test