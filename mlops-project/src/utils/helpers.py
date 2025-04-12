def load_data(file_path):
    # Function to load data from a given file path
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Function to preprocess the data
    # Example: handle missing values, encode categorical variables, etc.
    data.fillna(method='ffill', inplace=True)
    return data

def evaluate_model(model, X_test, y_test):
    # Function to evaluate the model's performance
    from sklearn.metrics import accuracy_score
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)