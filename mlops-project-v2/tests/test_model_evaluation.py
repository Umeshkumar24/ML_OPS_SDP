import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.model.model_evaluation import evaluate_model

def test_evaluate_model():
    # Create a mock model and data for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(6, activation='softmax', input_shape=(10,))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate mock predictions and true labels
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)

    # Assertions to check if the evaluation metrics are calculated correctly
    assert accuracy == accuracy_score(y_true, y_pred), "Accuracy calculation is incorrect"
    assert precision == precision_score(y_true, y_pred, average='weighted'), "Precision calculation is incorrect"
    assert recall == recall_score(y_true, y_pred, average='weighted'), "Recall calculation is incorrect"
    assert f1 == f1_score(y_true, y_pred, average='weighted'), "F1 Score calculation is incorrect"

def test_confusion_matrix():
    # Create mock true labels and predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Check if the confusion matrix is of the correct shape
    assert cm.shape == (3, 3), "Confusion matrix shape is incorrect"
    assert cm[0, 0] == 2, "Confusion matrix values are incorrect"
    assert cm[1, 1] == 2, "Confusion matrix values are incorrect"
    assert cm[2, 2] == 1, "Confusion matrix values are incorrect"