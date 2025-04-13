import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import numpy as np

def model_trainer(X_train, y_train, model_type: str = "dnn"):
    """
    Train a machine learning or deep learning model based on the specified model type.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        model_type (str): The type of model to train. Options:
            - "dnn": Deep Neural Network
            - "lstm": Long Short-Term Memory Network
            - "logistic_regression"
            - "random_forest"
            - "gradient_boosting"
            - "svm"
            - "knn"
            - "decision_tree"

    Returns:
        model: Trained model.
    """
    if model_type == "dnn":
        print(f"Input shape for training: {X_train.shape}")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Starting DNN model training...")
        history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2, verbose=1)
        print("DNN model training finished.")

        # Log training metrics
        for epoch, acc in enumerate(history.history['accuracy']):
            mlflow.log_metric("accuracy", acc, step=epoch)
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("loss", loss, step=epoch)

        return model

    elif model_type == "lstm":
        print(f"Input shape for training: {X_train.shape}")
        X_train_reshaped = np.expand_dims(X_train, axis=2)  # Reshape for LSTM input
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, activation='tanh', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Starting LSTM model training...")
        history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
        print("LSTM model training finished.")

        # Log training metrics
        for epoch, acc in enumerate(history.history['accuracy']):
            mlflow.log_metric("accuracy", acc, step=epoch)
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("loss", loss, step=epoch)

        return model

    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose from 'dnn', 'lstm', 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm', 'knn', or 'decision_tree'.")

    # Train the machine learning model
    print(f"Starting {model_type} model training...")
    model.fit(X_train, y_train)
    print(f"{model_type} model training finished.")

    # Log the model to MLflow
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_type)

    return model