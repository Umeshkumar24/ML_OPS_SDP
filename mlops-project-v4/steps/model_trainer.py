from zenml.steps import step
import tensorflow as tf

@step
def model_trainer(X_train, y_train) -> tf.keras.Model: # Added return type hint
    """
    Train a TensorFlow/Keras model for binary classification.
    """
    print(f"Input shape for training: {X_train.shape}") # Good to log shape
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        # Consider if these sigmoid layers are necessary, often relu is used throughout hidden layers
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        # Corrected final layer for binary classification
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    # Corrected loss for binary classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting model training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    print("Model training finished.")
    return model