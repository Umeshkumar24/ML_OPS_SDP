import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def split_data(data, target_column, test_size=0.2, random_state=None):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test