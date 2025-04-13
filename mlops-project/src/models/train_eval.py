import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1)

    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss}, MAE: {mae}")

    # Make predictions
    predictions = model.predict(X_test, batch_size=32)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes))

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, predicted_classes)}")
    print(f"Precision: {precision_score(y_test, predicted_classes, average='weighted')}")
    print(f"Recall: {recall_score(y_test, predicted_classes, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, predicted_classes, average='weighted')}")

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

