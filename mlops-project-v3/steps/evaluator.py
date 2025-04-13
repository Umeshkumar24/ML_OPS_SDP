from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import numpy as np
import os
import matplotlib.pyplot as plt

def evaluator(model, X_test, y_test, output_dir: str = "results"):
    """
    Evaluate the model on the test set and save results.

    Args:
        model: Trained model to evaluate.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        output_dir (str): Directory to save evaluation results.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get predictions from the model
    raw_predictions = model.predict(X_test)  # Probabilities for each class
    predictions = np.argmax(raw_predictions, axis=1)  # Convert to class labels

    # Ensure y_test is in the correct format (1D array of class labels)
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.flatten()

    # Generate classification report and confusion matrix
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    # Log evaluation metrics
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])

    # Save classification report to a file
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(classification_report(y_test, predictions, zero_division=0))

    # Save confusion matrix to a file
    cm_file = os.path.join(output_dir, "confusion_matrix.txt")
    with open(cm_file, "w") as f:
        f.write(str(cm))

    # Save confusion matrix as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    heatmap_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(heatmap_file)
    plt.close()

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Log artifacts to MLflow
    mlflow.log_artifact(report_file)
    mlflow.log_artifact(cm_file)
    mlflow.log_artifact(heatmap_file)

    # Print additional metrics
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted', zero_division=0)}")
    print(f"Recall: {recall_score(y_test, predictions, average='weighted', zero_division=0)}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted', zero_division=0)}")

    print(f"Evaluation results saved in '{output_dir}' directory.")