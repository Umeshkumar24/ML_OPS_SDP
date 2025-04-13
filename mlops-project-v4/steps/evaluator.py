from zenml.steps import step
from sklearn.metrics import classification_report, confusion_matrix

@step
def evaluator(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    predictions = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))