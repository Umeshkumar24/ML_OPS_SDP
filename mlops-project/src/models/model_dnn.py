from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def build_model(X_train, y_train):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        # Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# def train_model(X_train, y_train):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(32, activation='sigmoid'),
#         tf.keras.layers.Dense(16, activation='sigmoid'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(6, activation='softmax')
#     ])
    
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.2, verbose=1)
#     return model, history

# def main():
#     data = load_data('data/raw/filtered_file.csv')
#     X, y = preprocess_data(data)
    
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     X_train_scaled = scale_features(X_train)
#     X_test_scaled = scale_features(X_test)
    
#     model, history = train_model(X_train_scaled, y_train)
#     save_model(model, 'draught_model1.h5')
    
#     evaluate_model(model, X_test_scaled, y_test)

# if __name__ == "__main__":
#     main()
